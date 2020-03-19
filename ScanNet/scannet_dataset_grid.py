import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import json
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
from ply_helper import read_ply, write_ply
from mesh import rasterize_mesh
from os import makedirs, listdir
from os.path import exists, join, isfile
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


class ScannetDataset:
    """
    Class to handle ScanNet dataset for segmentation task.
    """
    def __init__(self, path, npoint, input_threads=8, load_test=False, buffer=1024, debug=False, trainval=False):
        self.debug = debug
        self.npoint = npoint
        self.buffer = buffer
        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])
        self.num_threads = input_threads
        self.path = path
        self.trainval = trainval

        self.label_weights = np.array([1.1808748, 1., 1.0941308, 1.9492522, 2.2317414, 1.6149306, 2.3081288,
                                       2.040714, 1.8799158, 1.9753349, 2.3331642, 3.950435, 3.9714756, 2.5003498,
                                       2.4034925, 3.8694403, 4.572348, 4.5791054, 4.88347, 4.448638, 2.0478268])
        # Path of the training files
        self.train_path = join(self.path, 'training_points')
        self.test_path = join(self.path, 'test_points')

        # Proportion of validation scenes
        if trainval:
            self.validation_clouds = []
        else:
            self.validation_clouds = np.loadtxt(join(self.path, 'scannet_v2_val.txt'), dtype=np.str)

        # 1 to do validation, 2 to train on all data
        self.validation_split = 1 if not trainval else 2
        self.all_splits = []

        # Load test set or train set
        self.load_test = load_test
        self.prepare_pointcloud_ply()

    def init_labels(self):

        # Initiate all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

    def prepare_pointcloud_ply(self):

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        paths = [join(self.path, 'scans'), join(self.path, 'scans_test')]
        new_paths = [self.train_path, self.test_path]
        mesh_paths = [join(self.path, 'training_meshes'), join(self.path, 'test_meshes')]

        # Mapping from annot to NYU labels ID
        if not self.trainval:
            label_files = join(self.path, 'scannetv2-labels.combined.tsv')
            with open(label_files, 'r') as f:
                lines = f.readlines()
                names1 = [line.split('\t')[1] for line in lines[1:]]
                IDs = [int(line.split('\t')[4]) for line in lines[1:]]
                annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}

        for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):

            # Create folder
            if not exists(new_path):
                makedirs(new_path)
            if not exists(mesh_path):
                makedirs(mesh_path)

            # Get scene names
            scenes = np.sort([f for f in listdir(path)])
            N = len(scenes)

            for i, scene in enumerate(scenes):
                if exists(join(new_path, scene + '.ply')):
                    continue
                t1 = time.time()

                vertex_data, faces = read_ply(join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
                vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

                vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
                if new_path == self.train_path:

                    align_mat = None
                    with open(join(path, scene, scene + '.txt'), 'r') as txtfile:
                        lines = txtfile.readlines()
                    for line in lines:
                        line = line.split()
                        if line[0] == 'axisAlignment':
                            align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.int32)
                    R = align_mat[:3, :3]
                    T = align_mat[:3, 3]
                    vertices = vertices.dot(R.T) + T

                    with open(join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                        segmentations = json.load(f)

                    segIndices = np.array(segmentations['segIndices'])

                    with open(join(path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                        aggregation = json.load(f)

                    for segGroup in aggregation['segGroups']:
                        c_name = segGroup['label']
                        if c_name in names1:
                            nyuID = annot_to_nyuID[c_name]
                            if nyuID in self.label_values:
                                for segment in segGroup['segments']:
                                    vertices_labels[segIndices == segment] = nyuID

                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors, vertices_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                              triangular_faces=faces)

                else:
                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'],
                              triangular_faces=faces)

                # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
                points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

                # Subsample points
                sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

                # Collect colors from associated vertex
                sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

                if new_path == self.train_path:

                    # Collect labels from associated vertex
                    sub_labels = vertices_labels[sub_vert_inds.ravel()]

                    # Save points
                    write_ply(join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

                else:

                    # Save points
                    write_ply(join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

                #  Display
                print('{:s} {:.1f} sec  / {:.1f}%'.format(scene,
                                                          time.time() - t1,
                                                          100 * i / N))

        print('Done in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not exists(tree_path):
            makedirs(tree_path)

        # List of training files
        self.train_files = np.sort([join(self.train_path, f) for f in listdir(self.train_path) if f[-4:] == '.ply'])

        # Add test files
        self.test_files = np.sort([join(self.test_path, f) for f in listdir(self.test_path) if f[-4:] == '.ply'])

        if self.debug:
            self.train_files = self.train_files[-101:]
            self.test_files = self.test_files[:10]

        files = np.hstack((self.train_files, self.test_files))
        # Initiate containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_vert_inds = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if cloud_name in self.validation_clouds:
                    self.all_splits += [1]
                    cloud_split = 'validation'
                else:
                    self.all_splits += [0]
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
                continue

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack((data['vert_ind'], data['class'])).T

                # Subsample cloud
                sub_points, sub_colors, sub_int_features = grid_subsampling(points,
                                                                            features=colors,
                                                                            labels=int_features,
                                                                            sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if cloud_split == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features)
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if cloud_split == 'test':
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_vert_inds[cloud_split] += [sub_vert_inds]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        self.test_proj = []
        self.test_labels = []
        i_val = 0
        i_test = 0

        # Advanced display
        N = self.num_validation + self.num_test
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if (not self.load_test) and 'train' in cloud_folder and cloud_name in self.validation_clouds:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'training_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = vertex_data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1

            # Test projection
            if self.load_test and 'test' in cloud_folder:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'test_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = np.zeros(vertices.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1

            print('', end='\r')


        print('\n')

        return

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """
        config.augment_scale_anisotropic = True
        config.augment_scale_min = 0.9
        config.augment_scale_max = 1.1
        config.augment_noise = 0.001
        config.augment_color = 1.0
        config.augment_rotation = 'vertical'

        if split == 'training':
            config.augment_symmetries = [True, False, False]
        else:
            config.augment_symmetries = [False, False, False]

        if split == 'training':
            epoch_n = config.epoch_steps * config.batch_size
        elif split == 'validation':
            epoch_n = config.validation_size * config.batch_size
        elif split == 'test':
            epoch_n = config.validation_size * config.batch_size
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        data_split = split

        # Reset potentials
        def reset_potentials():
            self.potentials[split] = []
            self.min_potentials[split] = []

            for i, tree in enumerate(self.input_trees[data_split]):
                self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        reset_potentials()

        def spatially_regular_gen():
            for i in range(epoch_n):
                cloud_ind = int(np.argmin(self.min_potentials[split]))
                point_ind = np.argmin(self.potentials[split][cloud_ind])
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)
                center_point = points[point_ind, :].reshape(1, -1)
                noise = np.random.normal(scale=0.35, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if config.in_radius > 0:
                    input_inds = self.input_trees[split][cloud_ind].query_radius(pick_point, r=config.in_radius)[0]
                else:
                    buffer = self.buffer+np.random.randint(0,self.buffer//4)
                    if len(points) < self.npoint+buffer:
                        input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=len(points))[1][0]
                    else:
                        input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=self.npoint+buffer)[1][0]

                input_inds = self.shuffle_idx(input_inds)
                input_inds = input_inds[:self.npoint]

                # Number collected
                n = input_inds.shape[0]
                if n == 0:
                    # Reset potentials
                    reset_potentials()
                    return
                    # Safe check for very dense areas

                # Update potentials
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.potentials[split][cloud_ind][input_inds] += delta
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))
                n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]

                if split == 'test':
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                if split in ['test', 'validation']:
                    label_weights = np.zeros(input_points.shape[0])
                else:
                    label_weights = self.label_weights[input_labels]

                if len(input_inds) < self.npoint:
                    input_points, input_colors, input_inds, label_weights, input_labels = \
                        self.data_rep(input_points, input_colors, input_labels, input_inds, label_weights, self.npoint)

                # Add yield data
                if n > 0:
                    yield input_points, np.hstack((input_colors, input_points + pick_point)), input_labels, \
                          [input_points.shape[0]], input_inds, cloud_ind, label_weights

        # Define the generator that should be used for this split
        gen_func = spatially_regular_gen

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
        gen_shapes = ([self.npoint, 3], [self.npoint, 6], [self.npoint], [1], [self.npoint], [], [self.npoint])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds,
                   sample_weights):

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds, config)
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            num_batches = batch_inds[-1] + 1
            s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
            stacked_s = tf.gather(s, batch_inds)
            stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)


            return stacked_points, stacked_colors, point_labels, sample_weights, point_inds, cloud_inds

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        # Evaluation points are from coarse meshes, not from the ply file we created for our own training
        mesh_path = file_path.split('/')
        mesh_path[-2] = mesh_path[-2][:-6] + 'meshes'
        mesh_path = '/'.join(mesh_path)
        vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
        return np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    def tf_augment_input(self, stacked_points, batch_inds, config):

        # Parameter
        num_batches = batch_inds[-1] + 1

        # Rotation
        if config.augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random_uniform((num_batches,), minval=0, maxval=2 * np.pi)

            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))

            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)

            # Apply rotations
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])

        elif config.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))

        else:
            raise ValueError('Unknown rotation augmentation : ' + config.augment_rotation)

        # Scale

        # Choose random scales for each example
        min_s = config.augment_scale_min
        max_s = config.augment_scale_max

        if config.augment_scale_anisotropic:
            s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((num_batches, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if config.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)

        # Apply scales
        stacked_points = stacked_points * stacked_scales

        # Noise
        noise = tf.random_normal(tf.shape(stacked_points), stddev=config.augment_noise)
        stacked_points = stacked_points + noise

        return stacked_points, s, R

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """
        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):

            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1),
                                 lambda: tf.zeros((), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i+1:]),
                                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before,), dtype=tf.int32)
            inds_in = tf.fill((num_in,), batch_i)
            inds_after = tf.zeros((num_after,), dtype=tf.int32)
            n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

            b_inds += n_inds

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None])])

        return batch_inds

    def data_rep(self, xyz, color, labels, idx, weights, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        weights_aug = weights[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, weights_aug, label_aug

    def shuffle_idx(self, x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]


