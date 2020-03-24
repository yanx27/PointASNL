import os
import numpy as np
import yaml
import random
from auxiliary import laserscan

splits = ["train", "valid", "test"]
mapped_content = {0: 0.03150183342534689, 1: 0.042607828674502385, 2: 0.00016609538710764618, 3: 0.00039838616015114444,
                  4: 0.0021649398241338114, 5: 0.0018070552978863615, 6: 0.0003375832743104974,
                  7: 0.00012711105887399155, 8: 3.746106399997359e-05, 9: 0.19879647126983288, 10: 0.014717169549888214,
                  11: 0.14392298360372, 12: 0.0039048553037472045, 13: 0.1326861944777486, 14: 0.0723592229456223,
                  15: 0.26681502148037506, 16: 0.006035012012626033, 17: 0.07814222006271769, 18: 0.002855498193863172,
                  19: 0.0006155958086189918}

seed = 100

class SemanticKittiDataset():
    def __init__(self, root, sample_points=8192, block_size=10, num_classes=20, split='train', with_remission=False,
                 config_file='semantic-kitti.yaml', should_map=True, padding=0.01, random_sample=False, random_rate=0.1):
        self.root = root
        assert split in splits
        self.split = split
        self.padding = padding
        self.block_size = block_size
        self.sample_points = sample_points
        self.random_sample = random_sample
        self.with_remission = with_remission
        self.should_map = should_map
        self.config = yaml.safe_load(open(config_file, 'r'))
        self.scan = laserscan.SemLaserScan(nclasses=num_classes, sem_color_dict=self.config['color_map'])
        sequences = self.config['split'][split]

        self.points_name = []
        self.label_name = []
        for sequence in sequences:
            sequence = '{0:02d}'.format(int(sequence))
            points_path = os.path.join(self.root, 'sequences', sequence, 'velodyne')
            label_path = os.path.join(self.root, 'sequences', sequence, 'labels')
            seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(points_path) if pn.endswith('.bin')]
            seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(label_path) if ln.endswith('.label')]
            assert len(seq_points_name) == len(seq_label_name)
            seq_points_name.sort()
            seq_label_name.sort()
            self.points_name.extend(seq_points_name)
            self.label_name.extend(seq_label_name)

        if self.random_sample:
            random.Random(seed).shuffle(self.points_name)
            random.Random(seed).shuffle(self.label_name)
            total_length = len(self.points_name)
            self.points_name = self.points_name[:int(total_length * random_rate)]
            self.label_name = self.label_name[:int(total_length * random_rate)]

        label_weights_dict = mapped_content
        num_keys = len(label_weights_dict.keys())
        self.label_weights_lut = np.zeros((num_keys), dtype=np.float32)
        self.label_weights_lut[list(label_weights_dict.keys())] = list(label_weights_dict.values())
        self.label_weights_lut = np.power(np.amax(self.label_weights_lut[1:]) / self.label_weights_lut, 1 / 3.0)

        if should_map:
            remapdict = self.config["learning_map"]
            # make lookup table for mapping
            maxkey = max(remapdict.keys())
            # +100 hack making lut bigger just in case there are unknown labels
            self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
            self.remap_lut[list(remapdict.keys())] = list(remapdict.values())

    def __getitem__(self, index):
        points_name, label_name = self.points_name[index], self.label_name[index]
        self.scan.open_scan(points_name)
        self.scan.open_label(label_name)
        points = self.scan.points

        label = self.scan.sem_label
        if self.should_map:
            label = self.remap_lut[label]
        label_weights = self.label_weights_lut[label]
        coordmax = np.max(points[:, 0:3], axis=0)
        coordmin = np.min(points[:, 0:3], axis=0)

        for i in range(10):
            curcenter = points[np.random.choice(len(label), 1)[0], 0:3]
            curmin = curcenter - [self.block_size/2, self.block_size/2, 14]
            curmax = curcenter + [self.block_size/2, self.block_size/2, 14]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((points[:, 0:3] >= (curmin - 0.2)) * (points[:, 0:3] <= (curmax + 0.2)),
                               axis=1) == 3
            # print(curchoice)
            cur_point_set = points[curchoice, 0:3]
            cur_point_full = points[curchoice, :]
            cur_semantic_seg = label[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - self.padding)) * (cur_point_set <= (curmax + self.padding)), axis=1) == 3

            isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.sample_points, replace=True)
        point_set = cur_point_full[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = label_weights[semantic_seg]
        sample_weight *= mask
        if self.with_remission:
            point_set = np.concatenate((point_set, np.expand_dims(self.scan.remissions[choice], axis=1)), axis=1)

        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.points_name)


class SemanticKittiDataset_whole():
    def __init__(self, root, sample_points=8192, block_size=10, num_classes=20, split='train', with_remission=False,
                 config_file='semantic-kitti.yaml', should_map=True, padding=0.01, random_sample=False, random_rate=0.1):
        self.root = root
        assert split in splits
        self.split = split
        self.padding = padding
        self.block_size = block_size
        self.sample_points = sample_points
        self.random_sample = random_sample
        self.with_remission = with_remission
        self.should_map = should_map
        self.config = yaml.safe_load(open(config_file, 'r'))
        self.scan = laserscan.SemLaserScan(nclasses=num_classes, sem_color_dict=self.config['color_map'])
        sequences = self.config['split'][split]

        self.points_name = []
        self.label_name = []
        for sequence in sequences:
            sequence = '{0:02d}'.format(int(sequence))
            points_path = os.path.join(self.root, 'sequences', sequence, 'velodyne')
            label_path = os.path.join(self.root, 'sequences', sequence, 'labels')
            seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(points_path) if pn.endswith('.bin')]
            seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(label_path) if ln.endswith('.label')]
            assert len(seq_points_name) == len(seq_label_name)
            seq_points_name.sort()
            seq_label_name.sort()
            self.points_name.extend(seq_points_name)
            self.label_name.extend(seq_label_name)
        if self.random_sample:
            random.Random(seed).shuffle(self.points_name)
            random.Random(seed).shuffle(self.label_name)
            total_length = len(self.points_name)
            self.points_name = self.points_name[:int(total_length * random_rate)]
            self.label_name = self.label_name[:int(total_length * random_rate)]
        label_weights_dict = mapped_content
        num_keys = len(label_weights_dict.keys())
        self.label_weights_lut = np.zeros((num_keys), dtype=np.float32)
        self.label_weights_lut[list(label_weights_dict.keys())] = list(label_weights_dict.values())
        self.label_weights_lut = np.power(np.amax(self.label_weights_lut[1:]) / self.label_weights_lut, 1 / 3.0)

        if should_map:
            remapdict = self.config["learning_map"]
            # make lookup table for mapping
            maxkey = max(remapdict.keys())
            # +100 hack making lut bigger just in case there are unknown labels
            self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
            self.remap_lut[list(remapdict.keys())] = list(remapdict.values())

    def __getitem__(self, index):
        points_name, label_name = self.points_name[index], self.label_name[index]
        self.scan.open_scan(points_name)
        self.scan.open_label(label_name)
        points = self.scan.points

        label = self.scan.sem_label
        if self.should_map:
            label = self.remap_lut[label]
        label_weights = self.label_weights_lut[label]
        coordmax = np.max(points[:, 0:3], axis=0)
        coordmin = np.min(points[:, 0:3], axis=0)

        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / self.block_size).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / self.block_size).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * self.block_size, j * self.block_size, 0]
                curmax = coordmin + [(i + 1) * self.block_size, (j + 1) * self.block_size, coordmax[2] - coordmin[2]]
                curchoice = np.sum(
                    (points[:, 0:3] >= (curmin - 0.2)) * (points[:, 0:3] <= (curmax + 0.2)), axis=1) == 3
                cur_point_set = points[curchoice, 0:3]
                cur_point_full = points[curchoice, :]
                cur_semantic_seg = label[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue
                mask = np.sum((cur_point_set >= (curmin - self.padding)) * (cur_point_set <= (curmax + self.padding)),axis=1) == 3

                choice = np.random.choice(len(cur_semantic_seg), self.sample_points, replace=True)
                point_set = cur_point_full[choice, :]  # Nx3/6
                if self.with_remission:
                    point_set = np.concatenate((point_set, np.expand_dims(self.scan.remissions[choice], axis=1)),axis=1)
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]

                sample_weight = label_weights[semantic_seg]
                sample_weight *= mask  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)

        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.points_name)


class SemanticKittiDatasetSlidingWindow():
    # prepare to give prediction on each points
    def __init__(self, root, sample_points=8192, block_size=10, stride=3.3, num_classes=20, split='test', with_remission=False,
                 config_file='semantic-kitti.yaml', should_map=True):
        self.root = root
        assert split in splits
        self.split = split
        self.stride = stride
        self.block_size = block_size
        self.block_points = sample_points
        self.should_map = should_map
        self.with_remission = with_remission
        self.config = yaml.safe_load(open(config_file, 'r'))
        self.scan = laserscan.SemLaserScan(
            nclasses=num_classes, sem_color_dict=self.config['color_map'])
        sequences = self.config['split'][split]
        color = []
        for values in self.config['learning_map_inv'].values():
            color.append(self.config['color_map'][values])
        self.color_map = np.array(color)

        self.points_name = []
        self.label_name = []
        for sequence in sequences:
            sequence = '{0:02d}'.format(int(sequence))
            points_path = os.path.join(
                self.root, 'sequences', sequence, 'velodyne')
            label_path = os.path.join(self.root, 'sequences', sequence, 'labels')
            seq_points_name = [os.path.join(points_path, pn) for pn in os.listdir(points_path) if pn.endswith('.bin')]
            seq_points_name.sort()
            self.points_name.extend(seq_points_name)
            if split != 'test':
                seq_label_name = [os.path.join(label_path, ln) for ln in os.listdir(label_path) if ln.endswith('.label')]
                seq_label_name.sort()
                self.label_name.extend(seq_label_name)
        if should_map:
            remapdict = self.config["learning_map"]
            # make lookup table for mapping
            maxkey = max(remapdict.keys())
            # +100 hack making lut bigger just in case there are unknown labels
            self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
            self.remap_lut[list(remapdict.keys())] = list(remapdict.values())

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split_data(self, data, idx):
        new_data = []
        for i in range(len(idx)):
            new_data += [np.expand_dims(data[idx[i]], axis=0)]
        return new_data

    def nearest_dist(self, block_center, block_center_list):
        num_blocks = len(block_center_list)
        dist = np.zeros(num_blocks)
        for i in range(num_blocks):
            dist[i] = np.linalg.norm(block_center_list[i] - block_center, ord=2)  # i->j
        return np.argsort(dist)[0]

    def __getitem__(self, index):
        points_name = self.points_name[index]
        self.scan.open_scan(points_name)
        point_set_ini = self.scan.points
        if self.split != 'test':
            label_name = self.label_name[index]
            self.scan.open_label(label_name)
            label = self.scan.sem_label
            if self.should_map:
                label = self.remap_lut[label]

        coordmax = np.max(point_set_ini[:, 0:3], axis=0)
        coordmin = np.min(point_set_ini[:, 0:3], axis=0)
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / self.stride).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / self.stride).astype(np.int32)
        point_sets = []
        point_idxs = []
        block_center = []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * self.stride, j * self.stride, 0]
                curmax = curmin + [self.block_size, self.block_size, coordmax[2] - coordmin[2]]
                curchoice = np.sum(
                    (point_set_ini[:, 0:3] >= (curmin - 0.2)) * (point_set_ini[:, 0:3] <= (curmax + 0.2)), axis=1) == 3
                curchoice_idx = np.where(curchoice)[0]
                cur_point_set = point_set_ini[curchoice, :]
                if self.with_remission:
                    cur_point_set = np.concatenate((cur_point_set, np.expand_dims(self.scan.remissions[curchoice], axis=1)),axis=1)
                point_sets.append(cur_point_set)  # 1xNx3/4
                point_idxs.append(curchoice_idx)  # 1xN
                block_center.append((curmin[0:2] + curmax[0:2]) / 2.0)

        # merge small blocks
        num_blocks = len(point_sets)
        block_idx = 0
        while block_idx < num_blocks:
            if point_sets[block_idx].shape[0] > 4096:
                block_idx += 1
                continue

            small_block_data = point_sets[block_idx].copy()
            small_block_idxs = point_idxs[block_idx].copy()
            small_block_center = block_center[block_idx].copy()
            point_sets.pop(block_idx)
            point_idxs.pop(block_idx)
            block_center.pop(block_idx)
            nearest_block_idx = self.nearest_dist(small_block_center, block_center)
            point_sets[nearest_block_idx] = np.concatenate((point_sets[nearest_block_idx], small_block_data), axis=0)
            point_idxs[nearest_block_idx] = np.concatenate((point_idxs[nearest_block_idx], small_block_idxs), axis=0)
            num_blocks = len(point_sets)

        # divide large blocks
        num_blocks = len(point_sets)
        div_blocks = []
        div_blocks_idxs = []
        div_blocks_center = []
        for block_idx in range(num_blocks):
            cur_num_pts = point_sets[block_idx].shape[0]

            point_idx_block = np.array([x for x in range(cur_num_pts)])
            if point_idx_block.shape[0] % self.block_points != 0:
                makeup_num = self.block_points - point_idx_block.shape[0] % self.block_points
                np.random.shuffle(point_idx_block)
                point_idx_block = np.concatenate((point_idx_block, point_idx_block[0:makeup_num].copy()))

            np.random.shuffle(point_idx_block)

            sub_blocks = list(self.chunks(point_idx_block, self.block_points))

            div_blocks += self.split_data(point_sets[block_idx], sub_blocks)
            div_blocks_idxs += self.split_data(point_idxs[block_idx], sub_blocks)
            div_blocks_center += [block_center[block_idx].copy() for _ in range(len(sub_blocks))]
        div_blocks = np.concatenate(tuple(div_blocks), axis=0)
        div_blocks_idxs = np.concatenate(tuple(div_blocks_idxs), axis=0)
        if self.split != 'test':
            return div_blocks, div_blocks_idxs, point_set_ini, label
        else:
            return div_blocks,  div_blocks_idxs, point_set_ini

    def __len__(self):
        return len(self.points_name)

if __name__ == '__main__':
    dataset = SemanticKittiDatasetSlidingWindow('/home/yxu/github/data/kitti/dataset/',split='valid', with_remission=True)
    print(len(dataset))
    import time
    st = time.time()
    data = dataset[4][1]
    print(data.shape)
    print(data)
    print(time.time()-st)


