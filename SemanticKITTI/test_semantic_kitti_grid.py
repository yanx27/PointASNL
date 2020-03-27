# *_*coding:utf-8 *_*
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import json
import pickle
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import socket
import importlib
from pathlib import Path
from os import makedirs
from os.path import exists, join, isfile, dirname
from semantic_kitti_dataset_grid import SemanticKITTIDataset

data_config = 'semantic-kitti.yaml'
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map_inv"]

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/semantic_kitti/dataset/sequences', help='Root for dataset')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate scores from multiple test [default: 1]')
parser.add_argument('--test_area', type=str, default='08', help='options: 08,11,12,13,14,15,16,17,18,19,20,21')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

config = parser.parse_args()
with open(Path(FLAGS.model_path).parent / 'args.txt', 'r') as f:
    config.__dict__ = json.load(f)

BATCH_SIZE = FLAGS.batch_size
FLAGS.first_subsampling_dl = config.first_subsampling_dl
FLAGS.prepare_data = False
FLAGS.in_radius = config.in_radius
FLAGS.num_buffer = config.num_buffer
FLAGS.num_point = config.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(config.model)  # import network module
NUM_CLASSES = 20

HOSTNAME = socket.gethostname()
feature_channel = 0

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

class TimeLiner:
    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelTester:
    def __init__(self, logits, restore_snap=None):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        self.Log_file = open(Path(FLAGS.model_path).parent / 'test.txt', 'a')

        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        cProto.allow_soft_placement = True
        cProto.log_device_placement = False
        self.sess = tf.Session(config=cProto)

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
        else:
            self.sess.run(tf.global_variables_initializer())

        self.prob_logits = tf.nn.softmax(logits)
        self.test_probs = 0
        self.idx = 0

    def test(self, flat_inputs, dataset, is_training_pl):

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)
        self.test_probs = [np.zeros(shape=[len(l), NUM_CLASSES], dtype=np.float16)
                           for l in dataset.possibility]

        test_path = join(dirname(dataset.dataset_path), 'test', 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, '08')) if not exists(join(test_path, '08')) else None
        makedirs(join(test_path, '08', 'predictions')) if not exists(
            join(test_path, '08', 'predictions')) else None
        for seq_id in range(11, 22, 1):
            makedirs(join(test_path, str(seq_id))) if not exists(join(test_path, str(seq_id))) else None
            makedirs(join(test_path, str(seq_id), 'predictions')) if not exists(
                join(test_path, str(seq_id), 'predictions')) else None
        test_smooth = 0.98
        epoch_ind = 0

        with tqdm() as pbar:
            while True:
                try:
                    ops = (self.prob_logits,
                           flat_inputs[1],
                           flat_inputs[3],
                           flat_inputs[4])
                    stacked_probs, labels, point_inds, cloud_inds = self.sess.run(ops, {is_training_pl: False})

                    pbar.set_description(
                        "Step %08d, Min Potential %.6f" % (self.idx, np.min(dataset.min_possibility)))
                    pbar.update(1)

                    self.idx += 1
                    stacked_probs = np.reshape(stacked_probs, [FLAGS.batch_size,
                                                               config.num_point,
                                                               NUM_CLASSES])
                    for j in range(np.shape(stacked_probs)[0]):
                        probs = stacked_probs[j, :, :]
                        inds = point_inds[j, :]
                        c_i = cloud_inds[j][0]
                        self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

                except tf.errors.OutOfRangeError:
                    new_min = np.min(dataset.min_possibility)
                    log_out('\nEpoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_ind, new_min), self.Log_file)
                    if np.min(dataset.min_possibility) > FLAGS.num_votes:
                        log_out(' Min possibility = {:.1f}'.format(np.min(dataset.min_possibility)), self.Log_file)
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        for j in range(len(self.test_probs)):
                            test_file_name = dataset.test_list[j]
                            frame = test_file_name.split('/')[-1][:-4]
                            proj_path = join(dataset.dataset_path, dataset.test_scan_number, 'proj')
                            proj_file = join(proj_path, str(frame) + '_proj.pkl')
                            if isfile(proj_file):
                                with open(proj_file, 'rb') as f:
                                    proj_inds = pickle.load(f)
                            probs = self.test_probs[j][proj_inds[0], :]
                            pred = np.argmax(probs, 1)
                            store_path = join(test_path, dataset.test_scan_number, 'predictions',
                                              str(frame) + '.label')
                            pred = pred.astype(np.uint32)
                            upper_half = pred >> 16  # get upper half for instances
                            lower_half = pred & 0xFFFF  # get lower half for semantics
                            lower_half = remap_lut[lower_half]  # do the remapping of semantics
                            pred = (upper_half << 16) + lower_half  # reconstruct full label
                            pred = pred.astype(np.uint32)
                            pred.tofile(store_path)
                        log_out(str(dataset.test_scan_number) + ' finished', self.Log_file)

                        sys.exit()
                    self.sess.run(dataset.test_init_op)
                    epoch_ind += 1
                    continue

def val():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            dataset = SemanticKITTIDataset(FLAGS, FLAGS.test_area)
            dataset.init_input_pipeline()
            # create a iterator of the correct shape and type
            flat_inputs = dataset.flat_inputs
            is_training_pl = tf.placeholder(tf.bool, shape=())

            points = flat_inputs[0]
            pred, end_points = MODEL.get_model(points, is_training_pl, NUM_CLASSES,
                                               feature_channel=feature_channel)

        tester = ModelTester(pred, FLAGS.model_path)

        tester.test(flat_inputs, dataset, is_training_pl)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    val()