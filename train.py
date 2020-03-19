import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import utils.provider as provider
import modelnet_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', default='data/modelnet40_normal_resampled/', help='Data path')
parser.add_argument('--model', default='pointasnl_cls', help='Model name [default: pointasnl_cls]')
parser.add_argument('--exp_dir', default=None, help='Experiment dir [default: None]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=500000, help='Decay step for lr decay [default: 500000]')
parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.1]')
parser.add_argument('--normal', type=str, default='True', help='Whether use normal information [default: True]')
parser.add_argument('--rotation', action='store_true', help='Whether use rotation as augmentation [default: False]')
parser.add_argument('--uniform', action='store_true', help='Whether use uniform sampling [default: False]')
parser.add_argument('--AS', action='store_true', help='Whether use adaptive sampling [default: False]')
parser.add_argument('--debug', action='store_true')
FLAGS = parser.parse_args()

if FLAGS.normal == 'True':
    FLAGS.normal = True
else:
    FLAGS.normal = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
EXP_PATH = FLAGS.exp_dir
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')

if not os.path.exists('log/'): os.mkdir('log/')
if EXP_PATH is None:
    LOG_DIR = './log/' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
else:
    LOG_DIR = './log/' + EXP_PATH

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
PointASNL = './utils/pointasnl_util.py'
os.system('cp %s %s' % (PointASNL, LOG_DIR))
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 40

# Modelnet  train/test split
assert (NUM_POINT <= 10000)

DATA_PATH = FLAGS.data
TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

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
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, FLAGS.normal)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl,
                                               is_training_pl,
                                               num_class=NUM_CLASSES,
                                               use_normal=FLAGS.normal,
                                               bn_decay=bn_decay,
                                               adaptive_sample=FLAGS.AS)

            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        if EXP_PATH is not None and os.path.exists(EXP_PATH + '/latest_model.ckpt'):
            MODEL_PATH = EXP_PATH + '/latest_model.ckpt'
            saver.restore(sess, MODEL_PATH)
        else:
            init = tf.global_variables_initializer()  # Init variables
            sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = 0
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            acc = eval_one_epoch(sess, ops, test_writer)

            if acc > best_acc:
                best_acc = acc
                saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(LOG_DIR, "latest_model.ckpt"))
            log_string("Model saved in file: %s" % save_path)

            log_string('Best accuracy is: %.4f\n' % best_acc)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batch = int(len(TRAIN_DATASET) / BATCH_SIZE)

    with tqdm(total=num_batch) as pbar:
        while TRAIN_DATASET.has_next_batch():
            batch_data, batch_label = TRAIN_DATASET.next_batch()

            if FLAGS.rotation:
                if FLAGS.normal:
                    batch_data = provider.rotate_point_cloud_with_normal(batch_data)
                    batch_data = provider.rotate_perturbation_point_cloud_with_normal(batch_data)
                else:
                    batch_data = provider.rotate_point_cloud(batch_data)
                    batch_data = provider.rotate_perturbation_point_cloud(batch_data)

            batch_data[:, :, 0:3] = provider.random_scale_point_cloud(batch_data[:, :, 0:3])
            batch_data[:, :, 0:3] = provider.shift_point_cloud(batch_data[:, :, 0:3])
            batch_data = provider.shuffle_points(batch_data)
            batch_data = provider.random_point_dropout(batch_data)

            bsize = batch_data.shape[0]
            cur_batch_data[0:bsize, ...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val

            if FLAGS.debug:
                break

            pbar.update(1)

    log_string('Current Learning Rate %.6f' % sess.run(get_learning_rate(step)))
    log_string('Training loss: %f' % (loss_sum / num_batch))
    log_string('Training accuracy: %f\n' % (total_correct / float(total_seen)))
    TRAIN_DATASET.reset()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    num_batch = int(len(TEST_DATASET) / BATCH_SIZE)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    with tqdm(total=num_batch) as pbar:
        while TEST_DATASET.has_next_batch():
            batch_data, batch_label = TEST_DATASET.next_batch()
            bsize = batch_data.shape[0]

            # for the last batch in the epoch, the bsize:end are from last batch
            cur_batch_data[0:bsize, ...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_val
            batch_idx += 1
            for i in range(0, bsize):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)

            if FLAGS.debug:
                break

            pbar.update(1)

    log_string('Eval mean loss: %f' % (loss_sum / num_batch))
    log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('Eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct / float(total_seen)

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
