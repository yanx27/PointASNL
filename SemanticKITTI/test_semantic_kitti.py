import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from datetime import datetime
import importlib
import socket
import tensorflow as tf
import numpy as np
import argparse
import provider
import semantic_kitti_dataset
import time

seg_label_to_cat = {0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck', 5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist', 9: 'road',
                    10: 'parking', 11: 'sidewalk', 12: 'other-ground', 13: 'building', 14: 'fence', 15: 'vegetation', 16: 'trunk', 17: 'terrain', 18: 'pole', 19: 'traffic-sign'}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', default='../data/kitti/dataset/', help='Root for dataset')
parser.add_argument('--model', default='pointasnl_sem_seg', help='Model name [default: model]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during evaluation [default: 6]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [256/512/1024/2048] [default: 8192]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='log/dump/', help='dump folder path [dump]')
parser.add_argument('--num_votes', type=int, default=5, help='Aggregate classification scores from multiple rotations [default: 5]')
parser.add_argument('--with_remission', action="store_true", help='Whether use remission feature [default: False]')
parser.add_argument('--dataset', type=str, default='valid', help='[valid/test]')
parser.add_argument('--random_rotate', action="store_true", help='randomly rotate the point cloud')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATASET = FLAGS.dataset
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
WITH_REMISSION = FLAGS.with_remission
MODEL = importlib.import_module(FLAGS.model)  # import network module
DUMP_DIR = FLAGS.dump_dir + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)
SAVE_DIR = os.path.join(DUMP_DIR,'submit')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

feature_channel = 1 if FLAGS.with_remission else 0
NUM_CLASSES = 20
HOSTNAME = socket.gethostname()
DATA_PATH = FLAGS.data

print("start loading %s whole scene data ..." % DATASET)
TEST_DATASET_WHOLE_SCENE = semantic_kitti_dataset.SemanticKittiDatasetSlidingWindow(root=DATA_PATH, sample_points=NUM_POINT, split=DATASET, with_remission=WITH_REMISSION, block_size=10, stride=4)

g_label2color = TEST_DATASET_WHOLE_SCENE.color_map

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    with tf.device('/gpu:' + str(GPU_INDEX)):
        if WITH_REMISSION:
            pointclouds_pl = tf.placeholder(
                tf.float32, shape=(BATCH_SIZE, NUM_POINT, 4))
        else:
            pointclouds_pl = tf.placeholder(
                tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))

        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points = MODEL.get_model(
            pointclouds_pl, is_training_pl, NUM_CLASSES, feature_channel=feature_channel)
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
           'is_training_pl': is_training_pl,
           'pred': pred}

    eval_one_epoch(sess, ops, num_votes)

def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def eval_one_epoch(sess, ops, num_votes=1):
    is_training = False

    num_scans = len(TEST_DATASET_WHOLE_SCENE)

    log_string(str(datetime.now()))
    log_string('----PREPARING PREDICTIONS----')

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    labelweights = np.zeros(NUM_CLASSES)
    full_points_names = TEST_DATASET_WHOLE_SCENE.points_name
    for batch_idx in range(num_scans):
        start = time.time()
        t_seen_class = [0 for _ in range(NUM_CLASSES)]
        t_correct_class = [0 for _ in range(NUM_CLASSES)]
        t_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        full_points_name = full_points_names[batch_idx]

        components = full_points_name.split('/')
        sequence = components[-3]
        points_name = components[-1]
        label_name = points_name.replace('bin','label')
        log_string("Inference sequence %s-%s [%d/%d] ..." % (sequence, label_name.split('.')[0], batch_idx, num_scans))
        full_save_dir = os.path.join(SAVE_DIR, 'sequences',sequence, 'predictions')
        os.makedirs(full_save_dir, exist_ok=True)
        full_label_name = os.path.join(full_save_dir,label_name)
        whole_scene_label = None

        for vote_idx in range(num_votes):
            print("Voting [%d/%d] ..." % (vote_idx+1, num_votes))
            if DATASET == 'test':
                scene_data, scene_point_index, whole_scene_data = TEST_DATASET_WHOLE_SCENE[batch_idx]
            else:
                scene_data, scene_point_index, whole_scene_data, whole_scene_label = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            if WITH_REMISSION:
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 4))
            else:
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]

                if FLAGS.random_rotate:
                    batch_data[:, :, :3] = provider.rotate_point_cloud_z(batch_data[:, :, :3])

                feed_dict = {ops['pointclouds_pl']: batch_data,
                             ops['is_training_pl']: is_training}
                pred_val = sess.run(ops['pred'], feed_dict=feed_dict)  # BxNxNUM_CLASSES
                batch_pred_label = np.argmax(pred_val[:, :, 1:], 2) + 1  # BxN

                if sbatch == 0:
                    vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...])

        final_preds = np.argmax(vote_label_pool, axis=1)
        final_preds = final_preds.astype(np.uint32)
        print('writing %s' % full_label_name)
        final_preds.tofile(full_label_name)
        print('Visualuze %s-%s' % (sequence,label_name.split('.')[0]))
        end = time.time()
        print('Use Time %.2f s'%(end-start))

        fout = open(os.path.join(DUMP_DIR, '%s_%s_pred.obj'%(sequence,label_name)), 'w')
        if DATASET != 'test':
            fout_gt = open(os.path.join(DUMP_DIR, '%s_%s_gt.obj'%(sequence,label_name)), 'w')
        for i in range(whole_scene_data.shape[0]):
            color = g_label2color[final_preds[i]]
            fout.write('v %f %f %f %d %d %d\n' % (
                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1], color[2]))
            if DATASET != 'test':
                color_gt = g_label2color[whole_scene_label[i]]
                fout_gt.write('v %f %f %f %d %d %d\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0], color_gt[1], color_gt[2]))
        fout.close()
        if DATASET != 'test':
            fout_gt.close()
            correct = np.sum(final_preds == whole_scene_label)
            seen = len(whole_scene_label)
            total_correct += correct
            total_seen += seen
            tmp, _ = np.histogram(whole_scene_label, range(NUM_CLASSES + 1))
            labelweights += tmp
            for l in range(NUM_CLASSES):
                tem_seen = np.sum(whole_scene_label == l)
                t_seen_class[l] += tem_seen
                total_seen_class[l] += tem_seen
                temp_correct = np.sum((final_preds == l) & (whole_scene_label == l))
                temp_iou_deno_class = np.sum((final_preds == l) | (whole_scene_label == l))
                total_correct_class[l] += temp_correct
                t_correct_class[l] += temp_correct
                total_iou_deno_class[l] += temp_iou_deno_class
                t_iou_deno_class[l] += temp_iou_deno_class
            iou = np.array(t_correct_class[1:]) / (np.array(t_iou_deno_class[1:], dtype=np.float) + 1e-6)
            arr = np.array(t_seen_class[1:])
            mIoU = np.mean(iou[arr != 0])
            log_string('Mean IoU of %s-%s: %.4f'%(sequence, label_name.split('.')[0], mIoU))
        print('---------------------')
        if DATASET != 'test':
            if batch_idx % 10 == 0:
                mIoU = np.mean(np.array(total_correct_class[1:]) / (np.array(total_iou_deno_class[1:], dtype=np.float) + 1e-6))
                log_string('eval point avg class IoU: %f' % mIoU)
                log_string('eval whole scene point accuracy: %f' % (total_correct / float(total_seen)))
                log_string('eval whole scene point avg class acc: %f' % (
                    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
                labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))

                iou_per_class_str = '------- IoU --------\n'
                for l in range(1,NUM_CLASSES):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                        total_correct_class[l] / float(total_iou_deno_class[l]))
                log_string(iou_per_class_str)
                print('---------------------')
    print("Done!")


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
