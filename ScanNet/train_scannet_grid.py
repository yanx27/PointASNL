import json
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import time
from tqdm import tqdm
from scannet_dataset_grid import ScannetDataset
from sklearn.metrics import confusion_matrix
from metrics import IoU_from_confusions

seg_label_to_cat = {0: 'unannotated', 1: 'wall', 2: 'floor', 3: 'chair', 4: 'tabel', 5: 'desk', 6: 'bed', 7: 'bookshelf', 8: 'sofa', 9: 'sink',
                    10: 'bathtub', 11: 'tollet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window', 16: 'shower curtain', 17: 'refrigerator',
                    18: 'picture', 19: 'cabinet', 20: 'otherfurniture'}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/Scannet', help='Root for dataset')
parser.add_argument('--model', default='pointasnl_sem_seg_res', help='Model name [default: pointasnl_sem_seg]')
parser.add_argument('--log_dir', default=None, help='Log dir [default: None]')
parser.add_argument('--pretrain_dir', default=None, help='Pretrain dir [default: None]')
parser.add_argument('--num_point', type=int, default=8192, help='Point number [default: 8192]')
parser.add_argument('--num_buffer', type=int, default=1024, help='Buffer point number, work only if in_radius is 0 [default: 1024]')
parser.add_argument('--in_radius', type=float, default=0, help='Radius of chopped area, work only if it larger than 0 [default: 0]')
parser.add_argument('--epoch_sample', type=int, default=4800, help='Number of steps per epochs [default: 4800]')
parser.add_argument('--validation_size', type=int, default=100, help='Number of validation examples per epoch [default: 100]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 500]')
parser.add_argument('--from_epoch', type=int, default=0, help='Epoch to run from (for restoring from checkpoints) [default: 0]')
parser.add_argument('--snapshot_gap', type=int, default=20, help='Gap for voting test [default: 20]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size during training [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 400000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--with_rgb', type=str, default='True', help='Whether use rgb feature [default: True]')
parser.add_argument('--input_threads', type=int, default=8, help='Number of CPU threads for the input pipeline [default: 8]')
parser.add_argument('--first_subsampling_dl', type=float, default=0.04, help='Voxel size for grid sampling [default: 0.04]')
parser.add_argument('--trainval', action='store_true', help='Train with both train and valid sets [default: False]')
parser.add_argument('--debug', action='store_true')

FLAGS = parser.parse_args()

if FLAGS.with_rgb == 'True':
    FLAGS.with_rgb = True
else:
    FLAGS.with_rgb = False

FLAGS.epoch_steps = FLAGS.epoch_sample // FLAGS.batch_size

if FLAGS.debug:
    FLAGS.epoch_steps = 50
    FLAGS.snapshot_gap = 1
    FLAGS.batch_size = 2

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
FROM_EPOCH = FLAGS.from_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

if not os.path.exists('log/'): os.mkdir('log/')
if FLAGS.log_dir is None:
    LOG_DIR = 'log/test/'
else:
    LOG_DIR = 'log/' + FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = '../models/' + FLAGS.model + '.py'
PointASNL = '../utils/' + 'pointasnl_util.py'
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (PointASNL, LOG_DIR))
os.system('cp train_scannet_grid.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
    json.dump(FLAGS.__dict__, f, indent=2)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
feature_channel = 3 if FLAGS.with_rgb else 0
NUM_CLASSES = 21
validation_probs = None
val_proportions = None

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            dataset = ScannetDataset(FLAGS.data, FLAGS.num_point, FLAGS.input_threads,
                                     load_test=False,
                                     buffer=FLAGS.num_buffer,
                                     debug=FLAGS.debug,
                                     trainval=FLAGS.trainval)
            dataset.load_subsampled_clouds(FLAGS.first_subsampling_dl)
            gen_function_train, gen_types, gen_shapes = dataset.get_batch_gen('training', FLAGS)
            map_func = dataset.get_tf_mapping(FLAGS)
            train_data = tf.data.Dataset.from_generator(gen_function_train, gen_types, gen_shapes)
            train_data = train_data.map(map_func=map_func, num_parallel_calls=dataset.num_threads)
            train_data = train_data.batch(FLAGS.batch_size, drop_remainder=True)
            train_data = train_data.prefetch(10)
            gen_function_val, _, _ = dataset.get_batch_gen('validation', FLAGS)

            val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

            # Transform inputs
            val_data = val_data.map(map_func=map_func, num_parallel_calls=dataset.num_threads)
            val_data = val_data.batch(FLAGS.batch_size, drop_remainder=True)
            val_data = val_data.prefetch(10)

            # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            flat_inputs = iter.get_next()

            # create the initialisation operations
            train_init_op = iter.make_initializer(train_data)
            val_init_op = iter.make_initializer(val_data)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(FROM_EPOCH * FLAGS.epoch_steps)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss ---")
            # Get model and loss
            if not FLAGS.with_rgb:
                points = flat_inputs[0]
            else:
                points = tf.concat([flat_inputs[0], flat_inputs[1][:, :, :3]], axis=-1)

            point_labels = flat_inputs[2]
            sample_weights = flat_inputs[3]
            pred, end_points = MODEL.get_model(points, is_training_pl, NUM_CLASSES,
                                               bn_decay=bn_decay,
                                               feature_channel=feature_channel)
            loss = MODEL.get_loss(pred, point_labels, end_points, smpw=sample_weights)

            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(point_labels))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        config_proto.allow_soft_placement = True
        config_proto.log_device_placement = False
        sess = tf.Session(config=config_proto)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        if FLAGS.pretrain_dir is not None:
            saver.restore(sess, FLAGS.pretrain_dir)
            print('Loading model from %s ...'%FLAGS.pretrain_dir)
        else:
            init = tf.global_variables_initializer()  # Init variables
            sess.run(init, {is_training_pl: True})
            print('Training from scratch ...')

        ops = {'is_training_pl': is_training_pl,
                'pred': pred,
                'loss': loss,
                'train_op': train_op,
                'merged': merged,
                'step': batch,
                'labels': point_labels,
                'point_inds': flat_inputs[-2],
                'cloud_inds': flat_inputs[-1],
                'end_points': end_points}

        best_iou = 0
        best_iou_whole = 0

        for epoch in range(FROM_EPOCH, MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            start_time = time.time()
            train_one_epoch(sess, ops, train_writer, train_init_op)
            end_time = time.time()
            log_string('one epoch time: %.4f' % (end_time - start_time))
            iou, iou_whole = eval_one_epoch(pred, ops, sess, val_init_op, dataset, epoch % FLAGS.snapshot_gap == 0)

            if iou > best_iou:
                best_iou = iou

            if iou_whole > best_iou_whole:
                best_iou_whole = iou_whole
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % epoch))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "latest_model.ckpt"))
            log_string("Model saved in file: %s\n" % save_path)
            log_string('Best chopped class avg mIOU is: %.3f' % best_iou)
            if best_iou_whole > 0:
                log_string('Best voting whole scene class avg mIOU is: %.3f  \n' % best_iou_whole)


def train_one_epoch(sess, ops, train_writer, train_init_op):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string('---- TRAINING ----')
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_iou_deno = 0
    sess.run(train_init_op)
    num_steps = 0
    with tqdm(total=FLAGS.epoch_steps) as pbar:
        while True:
            try:
                feed_dict = {ops['is_training_pl']: is_training, }
                summary, step, _, loss_val, pred_val, batch_label = \
                    sess.run([ops['merged'], ops['step'],ops['train_op'], ops['loss'],
                    ops['pred'],ops['labels']],feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum(pred_val == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                iou_deno = 0
                for l in range(NUM_CLASSES):
                    iou_deno += np.sum((pred_val == l) | (batch_label == l))
                total_iou_deno += iou_deno
                loss_sum += loss_val
                num_steps += 1
                pbar.update(1)
            except tf.errors.OutOfRangeError:
                sess.run(train_init_op)
                break

    log_string('Training Loss: %f' % (loss_sum / num_steps))
    log_string('Training Accuracy: %f' % (total_correct / float(total_seen) * 100))
    log_string('Training IoU: %f' % (total_correct / float(total_iou_deno) * 100))

def eval_one_epoch(pred, inputs, sess, val_init_op, dataset, vote=False):
    """ ops: dict mapping from string to tf ops """
    log_string('---- EVALUATION ----')
    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95

    prob_logits = tf.nn.softmax(pred[:, :, 1:])

    # Do not validate if dataset has no validation cloud
    if dataset.validation_split not in dataset.all_splits:
        return 0, 0

    sess.run(val_init_op)
    nc_tot = NUM_CLASSES
    nc_model = NUM_CLASSES - 1

    # Initiate global prediction over validation clouds
    global validation_probs
    global val_proportions

    if validation_probs is None:
        validation_probs = [np.zeros((l.shape[0], nc_model)) for l in dataset.input_labels['validation']]
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

    predictions = []
    targets = []
    for _ in tqdm(range(FLAGS.validation_size),total=FLAGS.validation_size):
        try:
            ops = (prob_logits,
                   inputs['labels'],
                   inputs['point_inds'],
                   inputs['cloud_inds'])

            stacked_probs, labels, point_inds, cloud_inds = sess.run(ops, {inputs['is_training_pl']: False})

            # Stack all validation predictions for each class separately
            for b in range(stacked_probs.shape[0]):
                probs = stacked_probs[b]
                inds = point_inds[b]
                c_i = cloud_inds[b]

                validation_probs[c_i][inds] = val_smooth * validation_probs[c_i][inds] + (1 - val_smooth) * probs
                predictions += [probs]
                targets += [dataset.input_labels['validation'][c_i][inds]]

        except tf.errors.OutOfRangeError:
            break

    # Confusions for our subparts of validation set
    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):
        for l_ind, label_value in enumerate(dataset.label_values):
            if label_value in dataset.ignored_labels:
                probs = np.insert(probs, l_ind, 0, axis=1)

        preds = dataset.label_values[np.argmax(probs, axis=1)]
        Confs[i, :, :] = confusion_matrix(truth, preds, dataset.label_values)

    C = np.sum(Confs, axis=0).astype(np.float32)

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)

    # Balance with real validation proportions
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

    IoUs = IoU_from_confusions(C)
    mIoU = 100 * np.mean(IoUs)

    log_string('Eval point avg class IoU: %.3f' % (mIoU))

    # Voting test
    mIoU_vote = 0
    if vote:
        log_string('---- VOTING EVALUATION ----')
        files = dataset.train_files
        Confs = np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int32)
        i_val = 0

        with tqdm(total=len(dataset.input_labels['validation'])) as pbar:
            for i, file_path in enumerate(files):
                if dataset.all_splits[i] == dataset.validation_split:
                    # Get probs on our own ply points
                    sub_probs = validation_probs[i_val]

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(dataset.label_values):
                        if label_value in dataset.ignored_labels:
                            sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                    # Get the predicted labels
                    sub_preds = dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                    # Reproject preds on the evaluations points
                    preds = (sub_preds[dataset.validation_proj[i_val]]).astype(np.int32)
                    labels = dataset.validation_labels[i_val].astype(np.int32)

                    Confs += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

                    i_val += 1
                    pbar.update(1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
            if label_value in dataset.ignored_labels:
                Confs = np.delete(Confs, l_ind, axis=0)
                Confs = np.delete(Confs, l_ind, axis=1)

        IoUs = IoU_from_confusions(Confs)
        mIoU_vote = 100 * np.mean(IoUs)

        iou_per_class_str = '------- IoU --------\n'
        for l in range(1, NUM_CLASSES):
            iou_per_class_str += 'class %s IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                100 * IoUs[l - 1])

        log_string(iou_per_class_str)

        log_string('Eval voting avg class IoU: %.3f \n' % (mIoU_vote))

    return mIoU, mIoU_vote

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()


