import os
import os.path
import numpy as np
import sys
from Common import point_operation
import h5py
import random
import transforms3d
import tensorflow as tf
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
#sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from Common.Const import GPU
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/sampling"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/grouping"%GPU))
from tf_sampling import gather_point, farthest_point_sample
from tf_grouping import query_ball_point, group_point, knn_point

def load_h5_name(h5_filename,mode="uniform",normalized=True):
    f = h5py.File(h5_filename,"r")
    data = f['data'][:]
    if mode == 'scan' or mode == 'partial':
        name = f['label'][:]
    else:
        name = f['name'][:]


    if normalized:
        data = point_operation.normalize_point_cloud_with_normal(data)

    return data,name


def load_h5_cls(h5_filename, normalized=True):
    f = h5py.File(h5_filename)
    name = f['name'][:]
    fps_points =  f['FPS'][:]
    random_points =  f['Random'][:]
    labels =  f['label'][:]

    if normalized:
        fps_points = point_operation.normalize_point_cloud_with_normal(fps_points)
        random_points = point_operation.normalize_point_cloud_with_normal(random_points)

    return fps_points, random_points, labels,name

def load_h5(h5_filename, normalized=True):
    f = h5py.File(h5_filename,"r")
    data = f['data'][:]

    if normalized:
        data = point_operation.normalize_point_cloud_with_normal(data)
    return data
#
# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

def augment_cloud(Ps):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    pc_augm_scale = 1.25
    pc_augm_rot = True
    pc_augm_mirror_prob = 0.2
    if pc_augm_scale > 1:
        s = random.uniform(1 / pc_augm_scale, 1)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_rot:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), M)  # y=upright assumption
    if pc_augm_mirror_prob > 0:  # mirroring x&z, not y
        if random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), M)
    result = []
    for P in Ps:
        P[:, :3] = np.dot(P[:, :3], M.T)
        result.append(P)
    return result


def random_downsample_points(pts, K, sess=None):
    # if num_pts > 8K use farthest sampling
    # else use random sampling

    p1 = random.uniform(0, 1)
    possi = 0.5
    if p1 > possi:
        idx = farthest_point_sample(K, pts[np.newaxis, ...]).eval(session=sess)[0]
        pred_pc = pts[idx, 0:3]
        return pred_pc
    else:
        return pts[np.random.choice(pts.shape[0], K,
            replace=(K<pts.shape[0])), :]



def batch_random_points(pts, K, sess=None):
    # if num_pts > 8K use farthest sampling
    # else use random sampling

    return pts[np.random.choice(pts.shape[0], K, replace=(K<pts.shape[0])), :]

def batch_FPS_points(pts, K, sess=None):
    # if num_pts > 8K use farthest sampling
    # else use random sampling

    fps_points =  gather_point(pts, farthest_point_sample(K, pts)).eval(session=sess)

    return fps_points

class Fetcher(object):
    def __init__(self, opts, split='train', augment=True, shuffle=True):
        self.opts = opts
        self.root = self.opts.data_dir
        self.batch_size = int(self.opts.batch_size)
        self.npoints = int(self.opts.num_point)
        self.normalize = True
        #self.uniform = self.opts.uniform# False
        self.normal_channel = False
        self.augment = augment
        self.shuffle = shuffle #True if split=="train" else False
        self.mode = self.opts.mode
        dataset = self.opts.dataset

        if self.mode == "scan":
            h5_file = os.path.join(self.root, "scannet_train_random_2048.h5")
            print("data_file:", h5_file)
            self.data = load_h5(h5_file,normalized=self.normalize)
        else:
            h5_file = os.path.join(self.root,"%s_%s_%s_%d.h5"%(dataset, split,self.mode,self.npoints))
            # if split == "train":
            #     h5_file = os.path.join(self.root, "%s_%s_raw.h5" % (dataset, split))

            print("data_file:", h5_file)
            self.data = load_h5(h5_file,normalized=self.normalize)

        self.length = self.data.shape[0]
        self.reset()

    def __len__(self):
        return self.length

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3

    def reset(self):
        self.idxs = np.arange(0, self.length)
        if self.shuffle:
            np.random.shuffle(self.idxs)
            self.data = self.data[self.idxs]
            #self.shape_names = self.shape_names[self.idxs]
            #self.labels = self.labels[self.idxs]


        self.num_batches = (self.length+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches



    def next_batch(self, sess=None):
        ''' returned dimension may be smaller than self.batch_size '''
        self.batch_idx += 1

        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.length)
        bsize = end_idx - start_idx
        #print(type(bsize),type(self.npoints))
        batch_data = self.data[start_idx:end_idx,:,:3]# np.zeros([bsize, self.npoints, 3])
        #batch_label = self.labels[start_idx:end_idx]

        if self.mode == "raw":
            temp_data = np.zeros([bsize,self.npoints,3])
            batch_random_data = np.zeros([bsize,self.npoints,3])
            batch_fps_data = batch_FPS_points(batch_data,K=self.npoints,sess=sess)
            for i in range(bsize):
                if np.random.rand() > 0.5:
                    temp_data[i,...] = batch_random_points(batch_data[i],K=self.npoints,sess=sess)
                else:
                    temp_data[i, ...] = batch_fps_data[i,...]
            batch_data = temp_data

        if self.augment:
            batch_data = point_operation.shuffle_points(batch_data)

            batch_data = point_operation.rotate_point_cloud(batch_data)
            batch_data = point_operation.rotate_perturbation_point_cloud(batch_data)

            batch_data[:, :, 0:3] = point_operation.random_scale_point_cloud(batch_data[:, :, 0:3])
            batch_data[:, :, 0:3] = point_operation.shift_point_cloud(batch_data[:, :, 0:3])
            #batch_data[:, :, 0:3] = point_operation.random_mirror(batch_data[:, :, 0:3],mirror_prob=0.2)

            #batch_data = point_operation.random_point_dropout(batch_data)

        return batch_data, None

