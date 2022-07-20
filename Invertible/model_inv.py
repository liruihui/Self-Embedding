# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import warnings

warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
from Common.visu_utils import plot_pcd_three_views, point_cloud_three_views
from Common.ops import add_scalar_summary, add_hist_summary
from Invertible.dataset_inv import Fetcher
from Common import pc_util
# from Common.loss_utils import chamfer
import logging
import os, sys
from tqdm import tqdm
from time import time
from termcolor import colored
import numpy as np
import os.path as osp
# from Simplify.SimplifyNet import SimplifyNet
from Invertible.InvertNet import InvertNet
from Common import loss_utils

sys.path.append(os.path.join(os.getcwd(), "tf_ops/sampling"))
sys.path.append(os.path.join(os.getcwd(), "tf_ops/nn_distance"))
sys.path.append(os.path.join(os.getcwd(), "tf_ops/approxmatch"))

import tf_nndistance
import tf_approxmatch

MODEL_SAVER_ID = "models.ckpt"


class Model(object):
    def __init__(self, opts, name='invert_net', graph=None):

        self.graph = tf.get_default_graph()
        self.name = name

        with tf.variable_scope(name):
            with tf.device("/cpu:0"):
                self.epoch = tf.get_variable(
                    "epoch", [], initializer=tf.constant_initializer(0), trainable=False
                )
            self.increment_epoch = self.epoch.assign_add(tf.constant(1.0))

        self.no_op = tf.no_op()

        self.opts.num_sample_point = self.opts.num_point //  self.opts.sample_rate

        self.opts = opts
        self.name = name

    def allocate_placeholders(self):
        # self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        with tf.variable_scope(self.name):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.input = tf.placeholder(tf.float32, shape=[self.opts.batch_size, self.opts.num_point, 3])
            self.gt = self.input  # tf.placeholder(tf.float32, shape=[None, self.opts.num_point,3])
            self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])

            self.alpha = tf.train.piecewise_constant(self.epoch, [150.0, 300.0],
                                                     [1.0e-7, 1.0, 1.0], 'alpha_op')
            self.beta = tf.train.piecewise_constant(self.epoch, [150.0, 300.0],
                                                    [1.0e-7, 1.0e-7, 1.0], 'beta_op')
            self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
            feq = 50
            self.weight_cd = tf.train.piecewise_constant(self.epoch, [50.0, 100.0],
                                                         [1.0, 0.2, 0.0], 'weight_cd')

            self.weight_emd = tf.train.piecewise_constant(self.epoch, [feq*1.0, feq*2.0],
                                                          [1.0, 1.0, 1.0], 'weight_emd')

            self.weight_density = tf.train.piecewise_constant(self.epoch, [feq*1.0, feq*2.0],
                                                              [0.0, 0.5, 1.0], 'weight_density')
            self.weight_l1_l2 = tf.train.piecewise_constant(self.epoch, [feq*1.0, feq*2.0],
                                                            [0.0, 1.0, 1.0], 'weight_l1_l2')
            # self.weight_geo = tf.train.piecewise_constant(self.epoch, [50.0, 100.0],
            #                                              [1.0, 0.2, 0.0], 'weight_ae')


    def backup(self):
        if not self.opts.restore:
            source_folder = os.path.join(os.getcwd(), "Invertible")
            common_folder = os.path.join(os.getcwd(), "Common")

            os.system("cp %s/configs_inv.py '%s/configs_inv.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/model_inv.py '%s/model_inv.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/InvertNet.py '%s/InvertNet.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/dataset_inv.py '%s/dataset_inv.py'" % (source_folder, self.opts.log_dir))
            os.system("cp %s/loss_utils.py '%s/loss_utils.py'" % (common_folder, self.opts.log_dir))
            os.system("cp %s/ops.py '%s/ops.py'" % (common_folder, self.opts.log_dir))

    def build_model(self):

        # idx_fps = farthest_point_sample(
        #     self.opts.num_sample_point, self.gt
        # )  # (batch_size, n_pc_point)
        # self.sample_gt = gather_point(self.gt, idx_fps)  # (batch_size, n_pc_point, 3)
        bn_momentum = tf.train.exponential_decay(
            0.5,
            self.epoch,  # global_var indicating the number of steps
            self.opts.decay_steps,  # step size,
            0.5,  # decay rate
            staircase=True
        )
        self.bn_decay = tf.minimum(0.99, 1 - bn_momentum)

        self.use_down_off = True
        self.use_up_off = True

        self.Simp = InvertNet(self.opts, self.is_training, name=self.name)

        # X -> Y
        self.down_point, self.offset, self.recon_point = self.Simp(self.input, self.bn_decay, use_down_off=self.use_down_off, use_up_off=self.use_up_off)
        self.sample_point = self.down_point + self.offset if self.use_down_off else self.offset


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.writer = tf.summary.FileWriter(self.opts.log_dir, self.graph)

        self.create_loss()
        self.setup_optimizer()

        self.summary_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.step = self.sess.run(self.global_step)
        self.backup()

        # self.restore_ae_model(
        #     self.opts.ae_dir, self.opts.ae_name, self.opts.ae_restore_epoch, verbose=True
        # )

    def build_model_eval(self):

        bn_momentum = tf.train.exponential_decay(
            0.5,
            self.epoch,  # global_var indicating the number of steps
            self.opts.decay_steps,  # step size,
            0.5,  # decay rate
            staircase=True
        )
        self.bn_decay = tf.minimum(0.99, 1 - bn_momentum)

        self.use_down_off = True
        self.use_up_off = True

        self.Simp = InvertNet(self.opts, self.is_training, name=self.name)

        # X -> Y
        self.down_point, self.offset, self.recon_point = self.Simp(self.input, self.bn_decay, use_down_off=self.use_down_off, use_up_off=self.use_up_off)
        self.sample_point = self.down_point + self.offset if self.use_down_off else self.offset


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.step = self.sess.run(self.global_step)

        # self.restore_ae_model(
        #     self.opts.ae_dir, self.opts.ae_name, self.opts.ae_restore_epoch, verbose=True
        # )

    def create_loss(self):

        # reconstruction loss
        # cost_p1_p2, _, cost_p2_p1, _ = tf_nndistance.nn_distance(self.recon_point, self.gt)
        # self.loss_ae = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

        self.loss_emd = loss_utils.earth_mover(self.recon_point, self.gt)
        self.loss_ae = loss_utils.chamfer(self.recon_point, self.gt)
        self.loss_hd = loss_utils.get_hausdorff_loss(self.recon_point,self.gt)
        self.loss_ae, self.loss_hd = 100*self.loss_ae, self.loss_hd
        self.loss_l2 = loss_utils.L2_loss(self.recon_point, self.gt)
        self.loss_l1 = 0.1 * loss_utils.L1_loss(self.recon_point, self.gt)
        self.loss_geo, self.loss_shape, self.loss_density, self.loss_direction = loss_utils.get_Geometric_Loss(self.recon_point, self.gt)
        self.loss_geo, self.loss_shape, self.loss_density, self.loss_direction = \
            10 * self.loss_geo, 10 * self.loss_shape, 5*self.loss_density,10 * self.loss_direction


        self.loss_dis = self.weight_density * self.loss_direction + self.loss_density + 1.0 * self.loss_ae  + self.loss_hd

        # self.loss_dis = self.weight_density * self.loss_direction + (self.loss_density + self.loss_shape) + self.loss_emd
        #self.loss_dis = self.weight_density * self.loss_density + self.weight_cd * self.loss_ae  # + self.weight_l1_l2*self.loss_l2

        if self.use_down_off:
            self.loss_simplification = 100 * tf.reduce_mean(tf.reduce_sum(self.offset ** 2, axis=2))
            #self.loss_simplification = 100 * loss_utils.L2_loss(self.offset,0.0,threshold=0.01)
        else:
            #self.loss_simplification = 10 * loss_utils.earth_mover(self.offset, self.down_point)
            self.loss_simplification = 100 * loss_utils.L2_loss(self.offset,self.down_point,threshold=0.01)



        self.repulsion_loss = 0.05 * loss_utils.get_repulsion_loss4(self.recon_point)
        # simplification loss

        self.loss = self.loss_dis + self.loss_simplification + self.repulsion_loss

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        w_reg_alpha = 1.0
        for rl in reg_losses:
            self.loss += w_reg_alpha * rl

    def summary_all(self):

        # summary
        add_scalar_summary('loss', self.loss, collection=self.name)
        add_scalar_summary('loss_l1_l2/loss_l1', self.loss_l1, collection=self.name)
        add_scalar_summary('loss_l1_l2/loss_l2', self.loss_l2, collection=self.name)
        add_scalar_summary('loss_geo/loss_geo', self.loss_geo, collection=self.name)
        add_scalar_summary('loss_geo/loss_shape', self.loss_shape, collection=self.name)
        add_scalar_summary('loss_geo/loss_density', self.loss_density, collection=self.name)
        add_scalar_summary('loss_geo/loss_direction', self.loss_direction, collection=self.name)
        add_scalar_summary('loss_cd_emd/loss_ae', self.loss_ae, collection=self.name)
        add_scalar_summary('loss_cd_emd/loss_emd', self.loss_emd, collection=self.name)
        add_scalar_summary('loss_simplification', self.loss_simplification, collection=self.name)
        add_scalar_summary('repulsion_loss', self.repulsion_loss, collection=self.name)

        #add_hist_summary('offset', tf.reduce_sum(self.offset ** 2, axis=2), collection=self.name)

        add_scalar_summary("weights/learning_rate", self.lr, collection=self.name)
        add_scalar_summary('weights/weight_cd', self.weight_cd, collection=self.name)
        add_scalar_summary('weights/weight_emd', self.weight_emd, collection=self.name)
        add_scalar_summary('weights/weight_density', self.weight_density, collection=self.name)
        add_scalar_summary('weights/weight_l1_l2', self.weight_l1_l2, collection=self.name)

        self.summary_op = tf.summary.merge_all(self.name)

        self.image_merged = tf.placeholder(tf.float32, shape=[None, 2000, 1500, 1])
        self.image_summary = tf.summary.image(self.name, self.image_merged, max_outputs=1)

    def setup_optimizer(self):
        self.lr = self.opts.base_lr
        if self.opts.lr_decay:
            self.lr = tf.train.exponential_decay(
                self.opts.base_lr,
                self.epoch,
                self.opts.decay_steps,
                decay_rate=0.7,
                staircase=True,
                name="learning_rate_decay",
            )
            self.lr = tf.maximum(self.lr, 1e-6)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        train_vars = tf.trainable_variables()
        # sampler_vars = [v for v in train_vars if v.name.startswith(self.name)]
        self.train_step = self.optimizer.minimize(self.loss)



    def train(self):
        self.allocate_placeholders()
        self.build_model()

        # Launch the session

        restore_epoch = 0
        if self.opts.restore:
            restore_epoch = 500
            self.restore_model(self.opts.log_dir, epoch=restore_epoch, verbose=True)
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
        else:
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

        self.train_dataset = Fetcher(self.opts, split="train",augment=True)
        self.opts.mode = "random"
        self.test_dataset_ran = Fetcher(self.opts, split="test", augment=False)
        self.opts.mode = "uniform"
        self.test_dataset_uni = Fetcher(self.opts, split="test", augment=False)
        #self.opts.mode = "partial"
        #self.test_dataset_par = Fetcher(self.opts, split="test", augment=False)
        #self.opts.mode = "scan"
        #self.opts.dataset = "scan"
        #self.test_dataset_scan = Fetcher(self.opts, split="test", augment=False)

        self.log_string("train_dataset: %d"%len(self.train_dataset))
        self.log_string("test_dataset_ran: %d"%len(self.test_dataset_ran))
        self.log_string("test_dataset_uni: %d"%len(self.test_dataset_uni))


        step = self.sess.run(self.global_step)
        start = time()
        best_ae = 100
        best_epoch = 0
        for _ in range(restore_epoch, self.opts.training_epoch):
            (
                loss,
                loss_ae,
                loss_emd,
                loss_simplification,
                duration,
            ) = self.train_one_epoch()
            self.train_dataset.reset()
            epoch = int(self.sess.run(self.increment_epoch))
            # logging.info('**** EPOCH %03d ****\t' % (epoch))
            self.log_string(
                "epoch %04d  loss=%.9f  loss_ae=%.9f  loss_emd=%.9f  loss_simplification=%.9f  time=%.4f" % (
                epoch, loss, loss_ae, loss_emd, loss_simplification, duration / 60.0))

            if (epoch % self.opts.epoch_per_save) == 0:
                # self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
                checkpoint_path = os.path.join(self.opts.log_dir, MODEL_SAVER_ID)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)
                print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

            if (epoch % self.opts.epoch_per_eval) == 0:
                (
                    loss,
                    loss_ae,
                    loss_hd,
                    loss_emd,
                    loss_simplification,
                    duration,
                ) = self.eval_one_epoch2(self.test_dataset_ran, epoch)
                self.log_string(
                    "On Random Test_data: %04d  loss=%.9f  loss_ae=%.9f loss_hd=%.9f  loss_emd=%.9f  loss_simplification=%.9f  time=%.4f" % (
                        epoch, loss, loss_ae, loss_hd, loss_emd, loss_simplification, duration / 60.0))

                (
                    loss,
                    loss_ae,
                    loss_hd,
                    loss_emd,
                    loss_simplification,
                    duration,
                ) = self.eval_one_epoch2(self.test_dataset_uni, epoch)
                self.log_string(
                    "On Uniform Test_data: %04d  loss=%.9f  loss_ae=%.9f  loss_hd=%.9f  loss_emd=%.9f  loss_simplification=%.9f  time=%.4f" % (
                        epoch, loss, loss_ae, loss_hd, loss_emd, loss_simplification, duration / 60.0))


                if best_ae > loss_ae:
                    best_ae = loss_ae
                    best_epoch = epoch
                self.log_string("On Test_data: %04d best_ae=%.9f" % (best_epoch, best_ae))




    def train_one_epoch(self):
        n_examples =  int(len(self.train_dataset))

        epoch_loss = 0.0
        epoch_loss_ae = 0.0
        epoch_loss_emd = 0.0
        epoch_loss_simplification = 0.0
        epoch_loss_code = 0.0

        n_batches = int(n_examples / self.opts.batch_size)-1
        start_time = time()

        # Loop over all batches
        for _ in tqdm(range(n_batches)):
            batch_i, _ = self.train_dataset.next_batch(self.sess)
            feed_dict = {self.input: batch_i, self.is_training: True}

            (
                _,
                loss,
                loss_ae,
                loss_emd,
                loss_simplification,
                recon,
                sample,
                down_point,
                summary
            ) = self.sess.run(
                (
                    self.train_step,
                    self.loss,
                    self.loss_ae,
                    self.loss_emd,
                    self.loss_simplification,
                    self.recon_point,
                    self.sample_point,
                    self.down_point,
                    self.summary_op
                ),
                feed_dict=feed_dict,
            )

            # Compute average loss
            epoch_loss += loss
            epoch_loss_ae += loss_ae
            epoch_loss_emd += loss_emd
            epoch_loss_simplification += loss_simplification
            if True:
                self.writer.add_summary(summary, self.step)
                self.step += 1
                # if self.step % 50 == 0:
                #     image_inputs = point_cloud_three_views(batch_i[0])
                #     image_inv = point_cloud_three_views(sample[0])
                #     image_cyc = point_cloud_three_views(recon[0])
                #     image_sample_gt = point_cloud_three_views(down_point[0])
                #     image_merged = np.concatenate([image_inputs, image_sample_gt, image_inv, image_cyc], axis=1)
                #     image_merged = np.transpose(image_merged, [1, 0])
                #     image_merged = np.expand_dims(image_merged, axis=0)
                #     image_merged = np.expand_dims(image_merged, axis=-1)
                #     image_summary = self.sess.run(self.image_summary, feed_dict={self.image_merged: image_merged})
                #     self.writer.add_summary(image_summary, self.step)

        epoch_loss /= n_batches
        epoch_loss_ae /= n_batches
        epoch_loss_emd /= n_batches
        epoch_loss_simplification /= n_batches
        duration = time() - start_time

        return (
            epoch_loss,
            epoch_loss_ae,
            epoch_loss_emd,
            epoch_loss_simplification,
            duration,
        )

    def eval_one_epoch2(self, dataset, epoch=0):
        n_examples = int(len(dataset))
        epoch_loss = 0.0
        epoch_loss_ae = 0.0
        epoch_loss_hd = 0.0
        epoch_loss_emd = 0.0
        epoch_loss_simplification = 0.0
        epoch_loss_code = 0.0

        n_batches = int(n_examples / self.opts.batch_size)-1
        start_time = time()

        idx = 0
        eval_dir = os.path.join(self.opts.log_dir, "eval")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        # Loop over all batches
        for i in tqdm(range(n_batches)):
            batch_i, _ = dataset.next_batch(self.sess)
            feed_dict = {self.input: batch_i, self.is_training: False}

            (
                loss,
                loss_ae,
                loss_hd,
                loss_emd,
                loss_simplification,
                recon,
                offset,
                down_point,
            ) = self.sess.run(
                (
                    self.loss,
                    self.loss_ae,
                    self.loss_hd,
                    self.loss_emd,
                    self.loss_simplification,
                    self.recon_point,
                    self.offset,
                    self.down_point
                ),
                feed_dict=feed_dict,
            )
            if i==0:
                point = np.concatenate([down_point[0], offset[0]], axis=-1)
                pc_sim, _, _ = pc_util.normalize_point_cloud(down_point[0] + offset[0])
                pc_sim2 = down_point[0] + offset[0]

                np.savetxt(os.path.join(eval_dir, str(epoch) + "_in.xyz"), batch_i[0], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(epoch) + "_out.xyz"), recon[0], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(epoch) + "_sam.xyz"), down_point[0], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(epoch) + "_sim.xyz"), pc_sim, fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(epoch) + "_sim2.xyz"), pc_sim2, fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(epoch) + "_sam_off.xyz"), point, fmt="%.6f")

                pcds = [batch_i[0], down_point[0], pc_sim, recon[0]]
                # print(type(pcds), len(pcds))
                # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                plot_path = os.path.join(eval_dir, str(idx) + ".png")
                visualize_titles = ['Input', 'Sample', 'Encode', 'Recon']
                plot_pcd_three_views(plot_path, pcds, visualize_titles)

            # Compute average loss
            epoch_loss += loss
            epoch_loss_ae += loss_ae
            epoch_loss_hd += loss_hd
            epoch_loss_emd += loss_emd
            epoch_loss_simplification += loss_simplification

        dataset.reset()

        epoch_loss /= n_batches
        epoch_loss_ae /= n_batches
        epoch_loss_hd /= n_batches
        epoch_loss_emd /= n_batches
        epoch_loss_simplification /= n_batches
        duration = time() - start_time

        return (
            epoch_loss,
            epoch_loss_ae,
            epoch_loss_hd,
            epoch_loss_emd,
            epoch_loss_simplification,
            duration,
        )

    def eval_one_epoch(self, eval_data):
        n_examples = eval_data.num_examples
        epoch_loss = 0.0
        epoch_loss_ae = 0.0
        epoch_loss_emd = 0.0
        epoch_loss_simplification = 0.0
        epoch_loss_code = 0.0

        n_batches = int(n_examples / self.opts.batch_size)
        start_time = time()

        idx = 0
        eval_dir = os.path.join(self.opts.log_dir, "eval")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        plot_dir = os.path.join(self.opts.log_dir, "plot")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        # Loop over all batches
        for _ in tqdm(range(n_batches)):

            original_data = None
            batch_i, _, _ = eval_data.next_batch(self.opts.batch_size)

            feed_dict = {self.input: batch_i, self.is_training: False}

            (
                loss,
                loss_ae,
                loss_emd,
                loss_simplification,
                recon,
                offset,
                down_point,
            ) = self.sess.run(
                (
                    self.loss,
                    self.loss_ae,
                    self.loss_emd,
                    self.loss_simplification,
                    self.recon_point,
                    self.offset,
                    self.down_point
                ),
                feed_dict=feed_dict,
            )
            point = np.concatenate([down_point, offset], axis=-1)
            # print(batch_i.shape,down_point.shape,recon.shape)
            for i in range(batch_i.shape[0]):
                # recon0,_, dis0 = pc_util.normalize_point_cloud(batch_i[i])
                pc_sim, _, _ = pc_util.normalize_point_cloud(down_point[i] + offset[i])
                # recon2,_, dis2 = pc_util.normalize_point_cloud(recon[i])
                # print(dis0, dis1,dis2)
                # if i<3:
                #     continue
                # else:
                #     exit(0)

                np.savetxt(os.path.join(eval_dir, str(idx) + "_in.xyz"), batch_i[i], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(idx) + "_out.xyz"), recon[i], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(idx) + "_sam.xyz"), down_point[i], fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(idx) + "_sim.xyz"), pc_sim, fmt="%.6f")
                np.savetxt(os.path.join(eval_dir, str(idx) + "_sam_off.xyz"), point[i], fmt="%.6f")
                #
                # feed_dict = {self.input_x: batch_input_x,
                #              self.input_y: batch_input_y,
                #              self.pc_radius: batch_radius,
                #              self.is_training: False}
                # self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
                # pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
                if idx % 100 == 0:
                    pcds = [batch_i[i], down_point[i], pc_sim, recon[i]]
                    # print(type(pcds), len(pcds))
                    # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                    plot_path = os.path.join(eval_dir, str(idx) + ".png")
                    visualize_titles = ['Input', 'Sample', 'Encode', 'Recon']
                    plot_pcd_three_views(plot_path, pcds, visualize_titles)

                idx += 1

            # Compute average loss
            epoch_loss += loss
            epoch_loss_ae += loss_ae
            epoch_loss_emd += loss_emd
            epoch_loss_simplification += loss_simplification

        epoch_loss /= n_batches
        epoch_loss_ae /= n_batches
        epoch_loss_emd /= n_batches
        epoch_loss_simplification /= n_batches
        duration = time() - start_time

        return (
            epoch_loss,
            epoch_loss_ae,
            epoch_loss_emd,
            epoch_loss_simplification,
            duration,
        )

    def eval2(self):
        # reset_tf_graph()
        self.allocate_placeholders()
        self.build_model()

        # Launch the session
        # restore_epoch = 400
        self.restore_model(self.opts.log_dir, epoch=self.opts.restore_epoch, verbose=True)
        self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_test.txt'), 'w')

        self.test_dataset = Fetcher(self.opts, split="test", augment=False)

        fetchworker = Fetcher(self.opts)
        _, _, pc_data_test_curr = fetchworker.get_dataset()

        step = self.sess.run(self.global_step)
        start = time()
        (
            loss,
            loss_ae,
            loss_emd,
            loss_simplification,
            duration,
        ) = self.eval_one_epoch(pc_data_test_curr)
        self.log_string("On Held_Out: loss=%.9f  loss_ae=%.9f  loss_simplification=%.9f  time=%.4f" % (
        loss, loss_ae, loss_simplification, duration / 60.0))

    def restore_model(self, model_path, epoch, verbose=False):
        """Restore all the variables of a saved model.
        """
        self.saver.restore(
            self.sess, osp.join(model_path, MODEL_SAVER_ID + "-" + str(int(epoch)))
        )

        if self.epoch.eval(session=self.sess) != epoch:
            warnings.warn("Loaded model's epoch doesn't match the requested one.")
        else:
            if verbose:
                print("Model restored in epoch {0}.".format(epoch))

    def restore_ae_model(self, ae_model_path, ae_name, epoch, verbose=False):
        """Restore all the variables of a saved ae model.
        """
        MODEL_SAVER_ID = "models.ckpt"
        global_vars = tf.global_variables()
        ae_params = [v for v in global_vars if v.name.startswith(ae_name)]

        saver_ae = tf.train.Saver(var_list=ae_params)
        saver_ae.restore(
            self.sess, os.path.join(ae_model_path, MODEL_SAVER_ID + "-" + str(int(epoch)))
        )

        if verbose:
            print("AE Model restored from %s, in epoch %d" % (ae_model_path, epoch))

    def log_string(self, msg):
        # global LOG_FOUT
        logging.info(msg)
        self.LOG_FOUT.write(msg + "\n")
        self.LOG_FOUT.flush()



    def test(self,mode_type=""):
        # reset_tf_graph()
        self.allocate_placeholders()
        self.build_model_eval()

        # Launch the session
        # restore_epoch = 400
        self.restore_model(self.opts.log_dir, epoch=self.opts.restore_epoch, verbose=True)
        self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_test.txt'), 'w')
        out_folder = os.path.join("/home/lirh/pointcloud/InvertPoint_tf/experiments/new",self.opts.dataset)
        # if GPU == "52":
        #     out_folder = out_folder.replace("/data","/home")
        print(out_folder)
        subfix = "INV_"+mode_type
        modes = ["partial", "random", "uniform"][0:3]
        modes = ["scan"][0:1]
        for mode in modes:
            print(self.opts.log_dir, subfix, mode)
            self.opts.mode = mode

            HD_out_folder = os.path.join(out_folder, "HD", mode)
            if not os.path.exists(HD_out_folder):
                os.makedirs(HD_out_folder)

            LD_out_folder = os.path.join(out_folder, "LD", mode)
            if not os.path.exists(LD_out_folder):
                os.makedirs(LD_out_folder)

            up_out_folder = os.path.join(out_folder, subfix, mode)
            if not os.path.exists(up_out_folder):
                os.makedirs(up_out_folder)

            test_dataset = Fetcher(self.opts, split="test", augment=False,shuffle=False)
            test_data = test_dataset.data

            print("test_dataset: %d" % len(test_dataset))

            for i in tqdm(range(test_data.shape[0])):
                pc_HD = np.array(test_data[i, :, :3])
                # print(pc_HD.shape,inv_opts.num_sample_point)
                # pc_LD = self.sess.run(pc_LD_tensor, feed_dict={pc: pc_HD[:,:3]})
                # print(i, pc_HD.shape,pc_LD.shape)

                pc_LD,offset,recon = self.sess.run([self.down_point,self.offset,self.recon_point],
                                             feed_dict={self.input:pc_HD[np.newaxis, ...],
                                                        self.is_training: False})
                pc_LD = np.squeeze(pc_LD,axis=0)
                offset = np.squeeze(offset,axis=0)
                recon = np.squeeze(recon,axis=0)
                embed_pc = pc_LD + offset
                pc_off = np.concatenate([pc_LD, offset], axis=-1)

                file_name = str(i).zfill(4) + "_%s_%s.xyz" % (subfix,mode)
                inv_path = os.path.join(up_out_folder, file_name)
                np.savetxt(inv_path, recon, fmt='%.6f')

                file_name = str(i).zfill(4) + "_%s_%s_emded.xyz" % (subfix,mode)
                inv_path = os.path.join(up_out_folder, file_name)
                np.savetxt(inv_path, embed_pc, fmt='%.6f')

                file_name = str(i).zfill(4) + "_%s_%s_emded_off.xyz" % (subfix,mode)
                inv_path = os.path.join(up_out_folder, file_name)
                np.savetxt(inv_path, pc_off, fmt='%.6f')


                # file_name = str(i).zfill(4) + "_HD_%s.xyz" % mode
                # HD_path = os.path.join(HD_out_folder, file_name)
                # np.savetxt(HD_path, pc_HD, fmt='%.6f')
                #
                # file_name = str(i).zfill(4) + "_LD_%s.xyz" % mode
                # LD_path = os.path.join(LD_out_folder, file_name)
                # np.savetxt(LD_path, pc_LD, fmt='%.6f')











