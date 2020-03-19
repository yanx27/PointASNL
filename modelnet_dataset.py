import os
import os.path
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset():
    def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=True, normal_channel=False, modelnet10=False, cache_size=15000, shuffle=None, uniform=False):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.uniform = uniform
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = normal_channel
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert(split=='train' or split=='test')

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _get_item(self, index): 
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)

            # Take the first npoints
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)
        return point_set, cls
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, self.num_channel()))
        batch_label = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,cls = self._get_item(self.idxs[i+start_idx])
            batch_data[i] = ps
            batch_label[i] = cls
        self.batch_idx += 1
        return batch_data, batch_label
    
if __name__ == '__main__':
    d = ModelNetDataset(root = '../data/modelnet40_normal_resampled', split='test')
    print(d.shuffle)
    print(len(d))
    import time
    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)

    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch()
    print(ps_batch.shape)
    print(cls_batch.shape)
