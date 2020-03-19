import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import modelnet_dataset
import provider
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', default='data/modelnet40_normal_resampled/', help='Data path')
parser.add_argument('--model', default='pointasnl_cls', help='Model name. [default: pointasnl_cls]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [1024/512/256/128/64] [default: 1024]')
parser.add_argument('--model_path', required=True, help='Model checkpoint file path')
parser.add_argument('--dump_dir', default='log/dump/', help='Dump folder path [dump]')
parser.add_argument('--normal', type=str, default='True', help='Whether use normal information')
parser.add_argument('--num_votes', type=int, default=5, help='Aggregate classification scores from multiple test [default: 5]')
parser.add_argument('--AS', action='store_true', help='Whether use adaptive sampling [default: False]')
parser.add_argument('--noise', action='store_true', help='Noisy Point Number [1/10/50/100]')

FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

NOISE_POINT = [1,10,50,100]
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in open(os.path.join(FLAGS.data,'modelnet40_shape_names.txt'))]
HOSTNAME = socket.gethostname()

# Official train/test split
assert (NUM_POINT <= 10000)
DATA_PATH = FLAGS.data
TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test',
                                                normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):

    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, use_normal=FLAGS.normal)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, use_normal=FLAGS.normal, adaptive_sample=FLAGS.AS)
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss}

    log_string('*** Evaluation ***')
    acc = eval_one_epoch(sess, ops, num_votes)
    if FLAGS.noise:
        noise_acc = []
        txt = 'Noise    Accuracy\n'
        txt += ' 000       %.3f\n' %acc
        for noise_num in NOISE_POINT:
            log_string('\n*** Evaluation with %d Noisy Points ***' % noise_num)
            tem_acc = (eval_one_epoch(sess, ops, num_votes, NUM_NOISY_POINT=noise_num))
            noise_acc.append(tem_acc)
            txt += ' %03d       %.3f\n' % (noise_num, tem_acc)
        log_string(txt)

def eval_one_epoch(sess, ops, num_votes=1, NUM_NOISY_POINT=0):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    num_batch = int(len(TEST_DATASET) / BATCH_SIZE)

    total_correct = 0
    total_object = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    with tqdm(total=num_batch) as pbar:
        while TEST_DATASET.has_next_batch():
            batch_data, batch_label = TEST_DATASET.next_batch()
            # for the last batch in the epoch, the bsize:end are from last batch
            bsize = batch_data.shape[0]

            # noisy robustness
            if NUM_NOISY_POINT > 0:
                noisy_point = np.random.random((bsize, NUM_NOISY_POINT, 3))
                noisy_point = provider.normalize_data(noisy_point)
                batch_data[:bsize, :NUM_NOISY_POINT, :3] = noisy_point

            loss_vote = 0
            cur_batch_data[0:bsize,...] = batch_data
            cur_batch_label[0:bsize] = batch_label

            batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
            for vote_idx in range(num_votes):
                # Shuffle point order to achieve different farthest samplings
                shuffled_indices = np.arange(NUM_POINT)
                np.random.shuffle(shuffled_indices)

                feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                             ops['labels_pl']: cur_batch_label,
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                batch_pred_sum += pred_val
                loss_vote += loss_val
            loss_vote /= num_votes

            pred_val = np.argmax(batch_pred_sum, 1)
            correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
            total_correct += correct
            total_seen += bsize
            loss_sum += loss_vote
            batch_idx += 1
            total_object += BATCH_SIZE
            for i in range(bsize):
                l = batch_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)

            pbar.update(1)

    log_string('Eval mean loss: %f' % (loss_sum / float(total_object)))
    log_string('Eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('Eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    TEST_DATASET.reset()
    return total_correct / float(total_seen)

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
