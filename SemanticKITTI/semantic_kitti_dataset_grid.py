import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

import tensorflow as tf
import numpy as np
import yaml
import pickle
from sklearn.neighbors import KDTree
from os.path import exists, join

data_config = 'semantic-kitti.yaml'
DATA = yaml.safe_load(open(data_config, 'r'))
remap_dict = DATA["learning_map"]
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

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


class SemanticKITTIDataset:
    """
    Class to handle SemanticKITTI dataset for segmentation task.
    """
    def __init__(self, args, test_id=14):
        self.args = args
        self.num_threads = 8
        self.grid_size = args.first_subsampling_dl
        dataset_path = args.data
        subgrid_path = dataset_path + '_' + str(self.grid_size)
        self.seq_list = np.sort(os.listdir(dataset_path))
        self.dataset_path = subgrid_path
        self.label_to_names = {0: 'unlabeled', 1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck',
                               5: 'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
                               9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground', 13: 'building',
                               14: 'fence', 15: 'vegetation', 16: 'trunk', 17: 'terrain', 18: 'pole',
                               19: 'traffic-sign'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        if args.prepare_data:
            self.prepare_pointcloud_ply(dataset_path, subgrid_path)

        self.test_scan_number = str(test_id)
        self.train_list, self.val_list, self.test_list = self.get_file_list(self.dataset_path, self.test_scan_number)
        self.train_list = self.shuffle_list(self.train_list)
        self.val_list = self.shuffle_list(self.val_list)

        num_per_class = np.array([0, 55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                9833174, 129609852, 4506626, 1168181])
        num_per_class = num_per_class.astype(np.float32)
        labelweights = num_per_class/np.sum(num_per_class)
        self.labelweights = np.power(np.amax(labelweights[1:]) / labelweights, 1 / 3.0)
        self.labelweights[0] = 0
        self.possibility = []
        self.min_possibility = []

        self.args.augment_scale_anisotropic = True
        self.args.augment_scale_min = 0.9
        self.args.augment_scale_max = 1.1
        self.args.augment_noise = 0.001
        self.args.augment_color = 1.0
        self.args.augment_rotation = 'vertical'



    def load_pc_kitti(self, pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    def load_label_kitti(self, label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    def shuffle_list(self, data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    def get_file_list(self, dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    def prepare_pointcloud_ply(self, dataset_path, output_path):
        for seq_id in self.seq_list:
            print('sequence' + seq_id + ' start')
            seq_path = join(dataset_path, seq_id)
            seq_path_out = join(output_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            pc_path_out = join(seq_path_out, 'velodyne')
            KDTree_path_out = join(seq_path_out, 'KDTree')
            os.makedirs(seq_path_out) if not exists(seq_path_out) else None
            os.makedirs(pc_path_out) if not exists(pc_path_out) else None
            os.makedirs(KDTree_path_out) if not exists(KDTree_path_out) else None

            if int(seq_id) < 11:
                label_path = join(seq_path, 'labels')
                label_path_out = join(seq_path_out, 'labels')
                os.makedirs(label_path_out) if not exists(label_path_out) else None
                scan_list = np.sort(os.listdir(pc_path))
                for scan_id in scan_list:
                    print(scan_id)
                    points = self.load_pc_kitti(join(pc_path, scan_id))
                    labels = self.load_label_kitti(join(label_path, str(scan_id[:-4]) + '.label'), remap_lut)
                    sub_points, sub_labels = grid_subsampling(points, labels=labels, sampleDl=self.grid_size)
                    search_tree = KDTree(sub_points)
                    KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
                    np.save(join(pc_path_out, scan_id)[:-4], sub_points)
                    np.save(join(label_path_out, scan_id)[:-4], sub_labels)
                    with open(KDTree_save, 'wb') as f:
                        pickle.dump(search_tree, f)
                    if seq_id == '08':
                        proj_path = join(seq_path_out, 'proj')
                        os.makedirs(proj_path) if not exists(proj_path) else None
                        proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                        proj_inds = proj_inds.astype(np.int32)
                        proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                        with open(proj_save, 'wb') as f:
                            pickle.dump([proj_inds], f)
            else:
                proj_path = join(seq_path_out, 'proj')
                os.makedirs(proj_path) if not exists(proj_path) else None
                scan_list = np.sort(os.listdir(pc_path))
                for scan_id in scan_list:
                    print(scan_id)
                    points = self.load_pc_kitti(join(pc_path, scan_id))
                    sub_points = grid_subsampling(points, sampleDl=self.grid_size)
                    search_tree = KDTree(sub_points)
                    proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)
                    KDTree_save = join(KDTree_path_out, str(scan_id[:-4]) + '.pkl')
                    proj_save = join(proj_path, str(scan_id[:-4]) + '_proj.pkl')
                    np.save(join(pc_path_out, scan_id)[:-4], sub_points)
                    with open(KDTree_save, 'wb') as f:
                        pickle.dump(search_tree, f)
                    with open(proj_save, 'wb') as f:
                        pickle.dump([proj_inds], f)


    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = int(len(self.train_list) / self.args.batch_size) * self.args.batch_size
            path_list = self.train_list
            self.args.augment_symmetries = [True, False, False]
        elif split == 'validation':
            num_per_epoch = int(len(self.val_list) / self.args.batch_size) * self.args.batch_size
            self.args.val_steps = int(len(self.val_list) / self.args.batch_size)
            path_list = self.val_list
            self.args.augment_symmetries = [False, False, False]
        elif split == 'test':
            num_per_epoch = int(len(self.test_list) / self.args.batch_size) * self.args.batch_size * 4
            path_list = self.test_list
            for test_file_name in path_list:
                points = np.load(test_file_name)
                self.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                self.min_possibility += [float(np.min(self.possibility[-1]))]
            self.args.augment_symmetries = [False, False, False]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):
                if split != 'test':
                    cloud_ind = i
                    pc_path = path_list[cloud_ind]
                    pc, tree, labels = self.get_data(pc_path)
                    # crop a small point cloud
                    pick_idx = np.random.choice(len(pc), 1)
                    selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
                    label_weights = self.labelweights[selected_labels]
                else:
                    cloud_ind = int(np.argmin(self.min_possibility))
                    pick_idx = np.argmin(self.possibility[cloud_ind])
                    pc_path = path_list[cloud_ind]
                    pc, tree, labels = self.get_data(pc_path)
                    selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

                    # update the possibility of the selected pc
                    dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
                    delta = np.square(1 - dists / np.max(dists))
                    self.possibility[cloud_ind][selected_idx] += delta
                    self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
                    label_weights = np.zeros(selected_pc.shape[0])

                yield (selected_pc.astype(np.float32),
                       selected_labels.astype(np.int32),
                       label_weights.astype(np.float32),
                       selected_idx.astype(np.int32),
                       np.array([cloud_ind], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.int32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([self.args.num_point, 3], [self.args.num_point], [self.args.num_point], [self.args.num_point], [1])

        return gen_func, gen_types, gen_shapes

    def get_data(self, file_path):
        seq_id = file_path.split('/')[-3]
        frame_id = file_path.split('/')[-1][:-4]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # Load labels
        if int(seq_id) >= 11:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
            labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    def crop_pc(self, points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        if self.args.in_radius > 0:
            select_idx = search_tree.query_radius(center_point, r=self.args.in_radius)[0]
        else:
            buffer = self.args.num_buffer + np.random.randint(0, self.args.num_buffer // 4)
            select_idx = search_tree.query(center_point, k=self.args.num_point+buffer)[1][0]

        select_idx = self.shuffle_idx(select_idx)
        select_idx = select_idx[:self.args.num_point]

        if len(select_idx) < self.args.num_point:
            num_in = len(select_idx)
            dup = np.random.choice(num_in, self.args.num_point - num_in)
            idx_dup = list(range(num_in)) + list(dup)
            select_idx = select_idx[idx_dup]

        select_points = points[select_idx]
        select_labels = labels[select_idx]

        return select_points, select_labels, select_idx

    def shuffle_idx(self, x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    def get_tf_mapping(self):

        def tf_map(batch_pc, batch_label, label_weights, batch_pc_idx, batch_cloud_idx):

            batch_pc, scales, rots = self.tf_augment_input(batch_pc, self.args)

            return batch_pc, batch_label, label_weights, batch_pc_idx, batch_cloud_idx

        return tf_map

    def tf_augment_input(self, stacked_points, config):
        # Parameter

        # Rotation
        if config.augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)

            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (3, 3))

            # Apply rotations
            stacked_points = tf.matmul(stacked_points, R)

        elif config.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(1,))

        else:
            raise ValueError('Unknown rotation augmentation : ' + config.augment_rotation)

        # Scale

        # Choose random scales for each example
        min_s = config.augment_scale_min
        max_s = config.augment_scale_max

        if config.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if config.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Apply scales
        stacked_points = stacked_points * s

        # Noise
        noise = tf.random_normal(tf.shape(stacked_points), stddev=config.augment_noise)
        stacked_points = stacked_points + noise
        return stacked_points, s, R

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        self.args.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        map_func = self.get_tf_mapping()
        self.train_data = self.train_data.map(map_func=map_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_data = self.val_data.map(map_func=map_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.test_data = self.test_data.map(map_func=map_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.batch_train_data = self.train_data.batch(self.args.batch_size, drop_remainder=True)
        self.batch_val_data = self.val_data.batch(self.args.batch_size, drop_remainder=True)
        self.batch_test_data = self.test_data.batch(self.args.batch_size, drop_remainder=True)

        self.batch_train_data = self.batch_train_data.prefetch(self.args.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(self.args.batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(self.args.batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)

        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)



