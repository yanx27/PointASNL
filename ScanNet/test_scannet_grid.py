# *_*coding:utf-8 *_*
import os
import sys
from os import makedirs
from os.path import exists, join
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from ply_helper import read_ply, write_ply
from sklearn.metrics import confusion_matrix
from metrics import IoU_from_confusions
import json
import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import time
from pathlib import Path
from scannet_dataset_grid import ScannetDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data', type=str, default='../data/Scannet', help='Root for dataset')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--num_votes', type=int, default=100, help='Aggregate scores from multiple test [default: 100]')
parser.add_argument('--split', type=str, default='validation', help='[validation/test]')
parser.add_argument('--saving', action='store_true', help='Whether save test results')
parser.add_argument('--debug', action='store_true', help='Whether save test results')
FLAGS = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

config = parser.parse_args()
with open(Path(FLAGS.model_path).parent / 'args.txt', 'r') as f:
    config.__dict__ = json.load(f)

config.validation_size = 500
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = config.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
WITH_RGB = config.with_rgb
MODEL = importlib.import_module(config.model)  # import network module
NUM_CLASSES = 21

HOSTNAME = socket.gethostname()
feature_channel = 3 if WITH_RGB else 0

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
    def __init__(self, pred, num_classes, saver, restore_snap=None):

        self.saver = saver
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        cProto.allow_soft_placement = True
        cProto.log_device_placement = False
        self.sess = tf.Session(config=cProto)

        if (restore_snap is not None):
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
        else:
            self.sess.run(tf.global_variables_initializer())

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(pred[:, :, 1:])
        self.num_classes = num_classes

    def test_cloud_segmentation(self, input, dataset, test_init_op, num_votes=100, saving=FLAGS.saving):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(test_init_op)

        # Initiate global prediction over test clouds
        nc_model = self.num_classes - 1
        self.test_probs = [np.zeros((l.data.shape[0], nc_model), dtype=np.float32) for l in dataset.input_trees['test']]

        # Test saving path
        if saving:
            saving_path = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            test_path = join('test', saving_path.split('/')[-1])
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'predictions')):
                makedirs(join(test_path, 'predictions'))
            if not exists(join(test_path, 'probs')):
                makedirs(join(test_path, 'probs'))
        else:
            test_path = None

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       input['labels'],
                       input['point_inds'],
                       input['cloud_inds'])
                stacked_probs, labels, point_inds, cloud_inds = \
                    self.sess.run(ops, {input['is_training_pl']: False})

                t += [time.time()]

                # Stack all predictions for each class separately
                for b in range(stacked_probs.shape[0]):
                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind, i0, 1000 * (mean_dt[0]), 1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['test'])))

                i0 += 1

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['test'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))
                print([np.mean(pots) for pots in dataset.potentials['test']])

                if last_min + 2 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_files
                    i_test = 0
                    for i, file_path in enumerate(files):

                        # Get file
                        points = dataset.load_evaluation_points(file_path)

                        # Reproject probs
                        probs = self.test_probs[i_test][dataset.test_proj[i_test], :]

                        # Insert false columns for ignored labels
                        probs2 = probs.copy()
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.int32)

                        # Project potentials on original points
                        pots = dataset.potentials['test'][i_test][dataset.test_proj[i_test]]

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds, pots],
                                  ['x', 'y', 'z', 'preds', 'pots'])
                        test_name2 = join(test_path, 'probs', cloud_name)
                        prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values
                                      if label not in dataset.ignored_labels]
                        write_ply(test_name2,
                                  [points, probs],
                                  ['x', 'y', 'z'] + prob_names)

                        # Save ascii preds
                        ascii_name = join(test_path, 'predictions', cloud_name[:-4] + '.txt')
                        np.savetxt(ascii_name, preds, fmt='%d')
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(test_init_op)
                epoch_ind += 1
                i0 = 0
                continue

        return

    def test_cloud_segmentation_on_val(self, input, dataset, val_init_op, num_votes=100, saving=True):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(val_init_op)

        # Initiate global prediction over test clouds
        nc_model = self.num_classes - 1
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32)
                           for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

        # Test saving path
        if saving:
            saving_path = time.strftime('Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            test_path = join('test', saving_path)
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions')):
                makedirs(join(test_path, 'val_predictions'))
            if not exists(join(test_path, 'val_probs')):
                makedirs(join(test_path, 'val_probs'))
        else:
            test_path = None

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        while last_min < num_votes:
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       input['labels'],
                       input['point_inds'],
                       input['cloud_inds'])
                stacked_probs, labels, point_inds, cloud_inds = self.sess.run(ops, {input['is_training_pl']: False})
                t += [time.time()]

                # Stack all validation predictions for each class separately
                for b in range(stacked_probs.shape[0]):
                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 10.0:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind, i0, 1000 * (mean_dt[0]), 1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))
                i0 += 1

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f}'.format(epoch_ind, new_min))

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    print('\nConfusion on sub clouds')
                    Confs = []
                    for i_test in range(dataset.num_validation):

                        # Insert false columns for ignored labels
                        probs = self.test_probs[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = dataset.input_labels['validation'][i_test]

                        # Confs
                        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n')

                    if int(np.ceil(new_min)) % 4 == 0:

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                        t1 = time.time()
                        files = dataset.train_files
                        i_val = 0
                        proj_probs = []
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:
                                # Reproject probs on the evaluations points
                                probs = self.test_probs[i_val][dataset.validation_proj[i_val], :]
                                proj_probs += [probs]
                                i_val += 1

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        print('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i_test in range(dataset.num_validation):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    proj_probs[i_test] = np.insert(proj_probs[i_test], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                            # Confusion
                            targets = dataset.validation_labels[i_test]
                            Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')

                        # Save predictions
                        print('Saving clouds')
                        t1 = time.time()
                        files = dataset.train_files
                        i_test = 0
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:
                                # Get points
                                points = dataset.load_evaluation_points(file_path)

                                # Get the predicted labels
                                preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                                # Project potentials on original points
                                pots = dataset.potentials['validation'][i_test][dataset.validation_proj[i_test]]

                                # Save plys
                                cloud_name = file_path.split('/')[-1]
                                test_name = join(test_path, 'val_predictions', cloud_name)
                                write_ply(test_name,
                                          [points, preds, pots, dataset.validation_labels[i_test]],
                                          ['x', 'y', 'z', 'preds', 'pots', 'gt'])
                                test_name2 = join(test_path, 'val_probs', cloud_name)
                                prob_names = ['_'.join(dataset.label_to_names[label].split())
                                              for label in dataset.label_values]
                                write_ply(test_name2,
                                          [points, proj_probs[i_test]],
                                          ['x', 'y', 'z'] + prob_names)
                                i_test += 1
                        t2 = time.time()
                        print('Done in {:.1f} s\n'.format(t2 - t1))

                self.sess.run(val_init_op)
                epoch_ind += 1
                i0 = 0
                continue
        return

def val():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            dataset = ScannetDataset(FLAGS.data, NUM_POINT, config.input_threads, load_test=FLAGS.split=='test', buffer=config.num_buffer, debug=FLAGS.debug)
            dl0 = config.first_subsampling_dl
            dataset.load_subsampled_clouds(dl0)
            map_func = dataset.get_tf_mapping(config)
            gen_function_val, gen_types, gen_shapes = dataset.get_batch_gen(FLAGS.split, config)

            val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

            # Transform inputs
            val_data = val_data.map(map_func=map_func, num_parallel_calls=dataset.num_threads)
            val_data = val_data.batch(FLAGS.batch_size, drop_remainder=True)
            val_data = val_data.prefetch(10)

            # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
            flat_inputs = iter.get_next()

            # create the initialisation operations
            val_init_op = iter.make_initializer(val_data)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            if not WITH_RGB:
                points = flat_inputs[0]
            else:
                points = tf.concat([flat_inputs[0], flat_inputs[1][:, :, :3]], axis=-1)
            point_labels = flat_inputs[2]
            pred, end_points = MODEL.get_model(points, is_training_pl, NUM_CLASSES,
                                               feature_channel=feature_channel)
            saver = tf.train.Saver()

        input = {
            'is_training_pl': is_training_pl,
            'pred': pred,
            'labels': point_labels,
            'point_inds': flat_inputs[-2],
            'cloud_inds': flat_inputs[-1]}

        tester = ModelTester(pred, NUM_CLASSES, saver, MODEL_PATH)

        if FLAGS.split == "validation":
            tester.test_cloud_segmentation_on_val(input, dataset, val_init_op)
        else:
            tester.test_cloud_segmentation(input, dataset, val_init_op)

if __name__ == "__main__":
    val()