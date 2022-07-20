# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
from Common import ops
import numpy as np
import os,sys
#ICLR 18
from Common.tf_utils import mlp_convs,mlps,mlp,mlp_conv

from Common.Const import GPU
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/sampling"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/nn_distance"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/approxmatch"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/grouping"%GPU))

from tf_sampling import gather_point, farthest_point_sample
import tf_sampling
from tf_grouping import query_ball_point, group_point
from Common.loss_utils import chamfer,get_repulsion_loss4

class InvertNet(object):
    def __init__(self, opts,is_training, name="InvertNet"):
        self.opts = opts
        self.reuse = False
        self.is_training = is_training
        self.num_point = self.opts.num_point
        self.num_sample_point = self.opts.num_sample_point

        self.bneck_size = self.opts.bneck_size
        self.verbose = self.opts.verbose
        self.name = name
        self.scope = name
        #self.ratio = 4


    def __call__(self, inputs, bn_decay=0.95, sample=None, use_up_off=False,use_down_off=True, use_sample=False):
        #bn_decay = 0.95
        B, N, _ = inputs.get_shape()
        use_bn= False
        part_bn = False
        n_layer = 6
        K = 16
        filter = 24
        use_noise = False
        dense_block = 2
        with tf.variable_scope(self.name+"/downscale", reuse=self.reuse):
        #with tf.variable_scope("downscale", reuse=self.reuse):
            # features_HD = ops.build_gcn_backbone_block(inputs, is_training=self.is_training, bn_decay=bn_decay, use_bn=part_bn,
            #                                            n_layer=n_layer, K=K, filter=filter, scope="HD_gcn_feat",all=False) # 144
            #

            # features_HD = ops.hierachy_feature_extractor(inputs, is_training=self.is_training, bn_decay=bn_decay,
            #                                             use_bn=use_bn,
            #                                            scope="HD_gcn_feat")

            # features_HD = ops.pointasnl_feature_extractor(inputs, is_training=self.is_training, bn_decay=bn_decay,
            #                                          use_bn=True,
            #                                          scope="HD_gcn_feat")


            #features_HD = ops.feature_extraction_up(inputs, scope='HD_gcn_feat', growth_rate=24, is_training=self.is_training, bn_decay=bn_decay,use_bn=use_bn)
            features_HD = ops.feature_extraction_GCN(inputs, scope='HD_gcn_feat', dense_block=2,
                                                     growth_rate=filter, is_training=self.is_training, bn_decay=bn_decay,use_bn=use_bn)

            as_neighbor = [12, 12]
            nsample = 32
            down_points, offset = ops.PointDownscale3(inputs, features_HD, npoint=self.num_sample_point, nsample=nsample, mlp=[128, 128, 256],
                                                      is_training=self.is_training, bn_decay=bn_decay,
                                                      scope='down_gcn_feat',
                                                      use_bn=use_bn,
                                                      as_neighbor=as_neighbor[0],
                                                      use_noise=False,
                                                      down_off= use_down_off)

        #down_points = tf.reshape(down_points+offset, [-1, self.num_sample_point, 3])

        #with tf.variable_scope("upscale", reuse=self.reuse):
        with tf.variable_scope(self.name+"/upscale", reuse=self.reuse):
            if use_down_off:
                pc = offset + down_points
            else:
                pc = offset

            if use_sample:
                pc = sample
            #
            # features_LD = ops.build_gcn_backbone_block(pc,is_training=self.is_training,bn_decay=bn_decay,use_bn=part_bn,
            #                                            n_layer=n_layer, K=K,filter=filter,scope="LD_gcn_feat",all=False)

            # features_LD = ops.hierachy_feature_extractor(pc, is_training=self.is_training, bn_decay=bn_decay,
            #                                              use_bn=use_bn,
            #                                              scope="LD_gcn_feat",
            #                                              npoints = [N,N//2,N//4], radius=[0.1,0.2,0.4])




            #features_LD = ops.feature_extraction_up(pc, scope='LD_gcn_feat', growth_rate=24, is_training=self.is_training, bn_decay=bn_decay,use_bn=use_bn)
            features_LD = ops.feature_extraction_GCN(pc, scope='LD_gcn_feat',dense_block=1,
                                                     growth_rate=filter, is_training=self.is_training, bn_decay=bn_decay,use_bn=use_bn)


            as_neighbor = [12, 12]

            l1_xyz = ops.PointUpscale(pc, features_LD, npoint=self.num_point, bn=use_bn, up_offset=use_up_off,
                                                        is_training=self.is_training, bn_decay=bn_decay,
                                                        scope='PointUpscale',mode="up3")

            up_points = tf.reshape(l1_xyz, [-1, self.num_point, 3])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return down_points,offset,up_points


