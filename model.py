# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Build model for inference or training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import nets
from ops import icp_grad  # pylint: disable=unused-import
from ops.icp_op import icp
import project
import reader
import tensorflow as tf
import util
import numpy as np

gfile = tf.gfile
slim = tf.contrib.slim

NUM_SCALES = 4


class Model(object):
  """Model code from SfMLearner."""

  def __init__(self,
               data_dir=None,
               is_training=True,
               learning_rate=0.0002,
               beta1=0.9,
               reconstr_weight=0.85,
               ssim_weight=0.15,
               smooth_weight=0.05,
               use_disp_weight=False,
               disp_reg_weight=0.01,
               lr_disp_consistency_weight=1.0,
               egomotion_snap_weight=0,
               fwd_bwd_egomoton_consistency_weight = 0,
               icp_weight=0.0,
               batch_size=4,
               img_height=128,
               img_width=416,
               seq_length=3,
               max_egomotion_step=1,
               #PWC-NET
               pyr_lvls=6,  # 特征金字塔的层数(特征数)
               flow_pred_lvl=2,  # 光流从哪一层输出
               search_range=4,
               use_res_cx=True,  # 是否用参差模块(增加参数量)
               use_dense_cx=True,
               #PWC-NET
               train_steps=40000,
               train_mode='depth_odom',
               sad_loss=False,
               use_charbonnier_loss=False,
               use_geometry_mask=False,
               use_flow_consistency_mask=False,
               use_temporal_dynamic_mask=False,
               use_temporal_occlusion_mask=False,
               legacy_mode=False):
    # self.opt = opt
    self.egomotion_snap_weight = egomotion_snap_weight
    self.fwd_bwd_egomoton_consistency_weight = fwd_bwd_egomoton_consistency_weight
    self.data_dir = data_dir
    self.is_training = is_training
    self.train_steps = train_steps
    self.learning_rate = learning_rate
    self.reconstr_weight = reconstr_weight
    self.smooth_weight = smooth_weight
    self.ssim_weight = ssim_weight
    self.use_disp_weight = use_disp_weight
    self.disp_reg_weight = disp_reg_weight
    self.lr_disp_consistency_weight = lr_disp_consistency_weight
    self.icp_weight = icp_weight
    self.beta1 = beta1
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.max_egomotion_step = max_egomotion_step
    self.num_source = seq_length - 1
    self.train_mode = train_mode
    self.sad_loss = sad_loss
    self.use_charbonnier_loss = use_charbonnier_loss
    self.use_geometry_mask = use_geometry_mask
    self.use_flow_consistency_mask = use_flow_consistency_mask
    self.use_temporal_occlusion_mask = use_temporal_occlusion_mask
    self.use_temporal_dynamic_mask = use_temporal_dynamic_mask
    self.legacy_mode = legacy_mode
    #PWC-NET
    self.pyr_lvls = pyr_lvls
    self.flow_pred_lvl = flow_pred_lvl
    self.search_range = search_range
    self.use_res_cx = use_res_cx
    self.use_dense_cx = use_dense_cx

    logging.info('data_dir: %s', data_dir)
    logging.info('train_steps: %s', train_steps)
    logging.info('learning_rate: %s', learning_rate)
    logging.info('beta1: %s', beta1)
    logging.info('smooth_weight: %s', smooth_weight)
    logging.info('ssim_weight: %s', ssim_weight)
    logging.info('reconstr_weight: %s', reconstr_weight)
    logging.info('disp_reg_weight %s', disp_reg_weight)
    logging.info('lr_disp_consistency_weight %s', lr_disp_consistency_weight)
    logging.info('icp_weight: %s', icp_weight)
    logging.info('batch_size: %s', batch_size)
    logging.info('img_height: %s', img_height)
    logging.info('img_width: %s', img_width)
    logging.info('seq_length: %s', seq_length)
    logging.info('max_egomotion_step: %s', max_egomotion_step)
    logging.info('train_mode: %s', train_mode)
    logging.info('sad_loss: %s', sad_loss)
    logging.info('use_charbonnier_loss: %s', use_charbonnier_loss)
    logging.info('use_geometry_mask: %s', use_geometry_mask)
    logging.info('use_flow_consistency_mask: %s', use_flow_consistency_mask)
    logging.info('use_disp_weight: %s', use_disp_weight)
    logging.info('use_temporal_occlusion_mask: %s', use_temporal_occlusion_mask)
    logging.info('use_temporal_dynamic_mask: %s', use_temporal_dynamic_mask)
    #PWC-NET
    logging.info('pyr_lvls: %s', pyr_lvls)
    logging.info('flow_pred_lvl: %s', flow_pred_lvl)
    logging.info('search_range: %s', search_range)
    logging.info('use_res_cx: %s', use_res_cx)
    logging.info('use_dense_cx: %s', use_dense_cx)


    # logging.info('legacy_mode: %s', legacy_mode)

    if self.is_training:
      self.build_train_graph()
    else:
      if self.train_mode == 'test_depth':
        self.build_depth_test_graph()
      if self.train_mode == 'test odom':
        self.build_egomotion_test_graph()
      if self.train_mode == 'test_flow':
        self.build_pose_flow_test_graph()

    # At this point, the model is ready.  Print some info on model params.
    util.count_parameters()

  def build_train_graph(self):
    self.build_inference_for_training()
    self.build_loss()
    self.build_train_op()
    self.build_summaries()

  def build_inference_for_training(self):
    """Invokes depth and ego-motion networks and computes clouds if needed."""

    with tf.name_scope("data_loading"):
      image_stack_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_height, self.img_width,
                                                     self.seq_length * 6], name='imgs_input')
      intrinsic_mat_input = tf.placeholder(tf.float32, shape=[self.batch_size, NUM_SCALES, 3, 3], name='intrincis')
      intrinsic_mat_inv_input = tf.placeholder(tf.float32, shape=[self.batch_size, NUM_SCALES, 3, 3], name='intrincis_inv')
      self.image_stack = image_stack_input
      self.image_stack_left = self.image_stack[:, :, :, :self.seq_length * 3]
      self.image_stack_right = self.image_stack[:, :, :, self.seq_length * 3:]
      self.intrinsic_mat = intrinsic_mat_input
      self.intrinsic_mat_inv = intrinsic_mat_inv_input
      self.baseline = tf.constant(0.54, shape=[self.batch_size,1])
      self.t_r2l = self.make_t_r2l(self.baseline)
      self.t_l2r = -1.0 * self.t_r2l

    if self.train_mode == "depth_odom" or self.train_mode == "depth":
      with tf.variable_scope('depth_prediction'):
        # Organized by ...[i][scale].  Note that the order is flipped in
        # variables in build_loss() below.
        self.disp_left = {}
        self.depth_left = {}
        self.disp_right = {}
        self.depth_right = {}
        if self.icp_weight > 0:
          self.cloud = {}
        for i in range(self.seq_length):
          image = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]

          # stereo
          # image_left = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
          # image_right = self.image_stack_left[:, :, :, 3 * (i+5):3 * (i + 6)]
          # image = tf.concat([image_left,image_right], axis=3)

          # monodepth upconv deconv
          disp_net_modo = nets.disp_net_monodepth()
          multiscale_disps_left, multiscale_disps_right = disp_net_modo.build_vgg(image,
                                                                                      get_pred=disp_net_modo.get_disp)

          # vid2depth deconv
          # multiscale_disps_left, _ = nets.disp_net(image, is_training=True)

          # geonet upconv
          # multiscale_disps_left, multiscale_disps_right = nets.geo_disp_net(image)

          # # disp to depth 使用sigmoid更加容易收敛
          # scale = [0, 1, 2, 3]
          # multiscale_depths_left = [
          #   tf.reshape(self.intrinsic_mat[:, s, 0, 0], [self.batch_size, 1, 1, 1]) * 0.54 / (d * self.img_width / (2 ** s))
          #   for (s, d) in zip(scale, multiscale_disps_left)]
          # multiscale_depths_right = [
          #   tf.reshape(self.intrinsic_mat[:, s, 0, 0], [self.batch_size, 1, 1, 1]) * 0.54 / (d * self.img_width / (2 ** s))
          #   for (s, d) in zip(scale, multiscale_disps_right)]

          multiscale_depths_left = [1.0 / d for d in multiscale_disps_left]
          multiscale_depths_right = [1.0 / d for d in multiscale_disps_right]
          self.disp_left[i] = multiscale_disps_left
          self.depth_left[i] = multiscale_depths_left
          self.disp_right[i] = multiscale_disps_right
          self.depth_right[i] = multiscale_depths_right

          # if self.icp_weight > 0:
          #   multiscale_clouds_i = [
          #     project.get_cloud(d,
          #                       self.intrinsic_mat_inv[:, s, :, :],
          #                       name='cloud%d_%d' % (s, i))
          #     for (s, d) in enumerate(multiscale_depths_left)
          #   ]
          #   self.cloud[i] = multiscale_clouds_i
          # Reuse the same depth graph for all images.
          tf.get_variable_scope().reuse_variables()
      # logging.info('disp: %s', util.info(self.disp_left))

    if self.train_mode == 'optical_flow' or self.train_mode == 'depth_odom':
      #bwd 和fwd按照相对位姿的方向来确定 前向位姿fwd 后向位姿bwd 光流与利用位姿方向计算出来的光流方向一致
      with tf.variable_scope('direct_flow_prediction'):
        flow_net = nets.Flow_net(pyr_lvls=self.pyr_lvls,
                                 flow_pred_lvl=self.flow_pred_lvl,
                                 search_range=self.search_range,
                                 use_res_cx=self.use_res_cx,
                                 use_dense_cx=self.use_dense_cx)
        self.direct_flow_bwd = {}
        self.direct_flow_fwd = {}
        # 相邻帧 跳一帧 跳两帧
        egomotion_index_base = 0
        for step in range(self.max_egomotion_step):
          egomotion_num = self.num_source - step   #2
          if egomotion_num == 0:
              break
          for i in range(egomotion_num):  #有问题？
            egomotion_index = egomotion_index_base + i

            # flow_fwd t1->t0
            # image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            # image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            # image_fwd = tf.concat([image_t1, image_t0], axis=3)
            # geonet upconv
            # flow_fwd = nets.geo_flow_net(image_fwd)
            # self.direct_flow_fwd[egomotion_index] = flow_fwd
            # tf.get_variable_scope().reuse_variables()

            # flow bwd t0->t1
            # image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            # image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            # image_bwd = tf.concat([image_t0, image_t1], axis=3)
            # flow_bwd = nets.geo_flow_net(image_bwd)
            # self.direct_flow_bwd[egomotion_index] = flow_bwd
            # tf.get_variable_scope().reuse_variables()

            # flow_fwd t1->t0
            image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            image_fwd = tf.concat([tf.expand_dims(image_t1, axis=1), tf.expand_dims(image_t0, axis=1)], axis=1)
            #data_shape prepare for PWC-NET
            image_fwd_adapt, image_fwd_adapt_info = util.adapt_x(image_fwd, self.pyr_lvls)
            if image_fwd_adapt_info is not None:
                direct_flow_fwd_info = [(image_fwd_adapt_info[0], image_fwd_adapt_info[2] ,     image_fwd_adapt_info[3], 2),
                                        (image_fwd_adapt_info[0], image_fwd_adapt_info[2] // 2, image_fwd_adapt_info[3] // 2, 2),
                                        (image_fwd_adapt_info[0], image_fwd_adapt_info[2] // 4, image_fwd_adapt_info[3] // 4, 2),
                                        (image_fwd_adapt_info[0], image_fwd_adapt_info[2] // 8, image_fwd_adapt_info[3] // 8, 2)
                                        ]
            else:
                direct_flow_fwd_info = None
            # flownet input shape (batchsize, 2 , H, W, 3)
            _, _, flow_fwd = flow_net.nn(image_fwd_adapt)
            flow_fwd = util.postproc_pred_flow(flow_fwd, direct_flow_fwd_info)
            self.direct_flow_fwd[egomotion_index] = flow_fwd
            tf.get_variable_scope().reuse_variables()

            # flow bwd t0->t1
            image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            image_bwd = tf.concat([tf.expand_dims(image_t0, axis=1), tf.expand_dims(image_t1, axis=1)], axis=1)
            # data_shape prepare for PWC-NET
            image_bwd_adapt, image_bwd_adapt_info = util.adapt_x(image_bwd, self.pyr_lvls)
            if image_bwd_adapt_info is not None:
                direct_flow_bwd_info = [(image_bwd_adapt_info[0], image_bwd_adapt_info[2] ,     image_bwd_adapt_info[3], 2),
                                        (image_bwd_adapt_info[0], image_bwd_adapt_info[2] // 2, image_bwd_adapt_info[3] // 2, 2),
                                        (image_bwd_adapt_info[0], image_bwd_adapt_info[2] // 4, image_bwd_adapt_info[3] // 4, 2),
                                        (image_bwd_adapt_info[0], image_bwd_adapt_info[2] // 8, image_bwd_adapt_info[3] // 8, 2)
                                        ]
            else:
                direct_flow_bwd_info = None
            # flownet input shape (batchsize, 2 , H, W, 3)
            _, _, flow_bwd = flow_net.nn(image_bwd_adapt)
            flow_bwd = util.postproc_pred_flow(flow_bwd, direct_flow_bwd_info)
            self.direct_flow_bwd[egomotion_index] = flow_bwd
            #for tensorboard show flow
            self.image_bwd_direct_flow_TB = image_bwd
            self.direct_flow_image = flow_bwd

            tf.get_variable_scope().reuse_variables()

          egomotion_index_base += egomotion_num

    if self.train_mode == 'depth_odom':
      with tf.variable_scope('egomotion_prediction'):
        self.egomotion_fwd = {}
        self.egomotion_bwd = {}
        #这里的flow是标准的前向和后向的flow
        self.pose_net_flow_fwd = {}
        self.pose_net_flow_bwd = {}

        #相邻帧 跳一帧 跳两帧
        egomotion_index_base = 0
        for step in range(self.max_egomotion_step):
          egomotion_num = self.num_source - step  #2
          if egomotion_num == 0:
              break
          for i in range(egomotion_num):
            egomotion_index = egomotion_index_base + i
            # egomotion_fwd
            image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            image_fwd = tf.concat([image_t0, image_t1], axis=3)
            egomotion_fwd, pose_flow_fwd, _ = nets.egomotion_net(image_fwd, is_training=True,
                                                                 legacy_mode=self.legacy_mode)
            self.egomotion_fwd[egomotion_index] = egomotion_fwd
            self.pose_net_flow_fwd[egomotion_index] = pose_flow_fwd
            for s in range(NUM_SCALES):
              curr_bs, curr_h, curr_w, _ = self.pose_net_flow_fwd[egomotion_index][s].get_shape().as_list()
              scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), 'float32')
              scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
              self.pose_net_flow_fwd[egomotion_index][s] = self.pose_net_flow_fwd[egomotion_index][s] * scale_factor

            #tensorboard show flow image
            self.image_fwd_pose_flow_TB = tf.concat([tf.expand_dims(image_t0, axis=1), tf.expand_dims(image_t1, axis=1)], axis=1)
            self.pose_flow_image = pose_flow_fwd

            tf.get_variable_scope().reuse_variables()

            # egomotion_bwd
            image_t0 = self.image_stack_left[:, :, :, 3 * i:3 * (i + 1)]
            image_t1 = self.image_stack_left[:, :, :, 3 * (i + 1 + step):3 * (i + 2 + step)]
            image_bwd = tf.concat([image_t1, image_t0], axis=3)
            egomotion_bwd, pose_flow_bwd, _ = nets.egomotion_net(image_bwd, is_training=True,
                                                                 legacy_mode=self.legacy_mode)
            self.egomotion_bwd[egomotion_index] = egomotion_bwd
            self.pose_net_flow_bwd[egomotion_index] = pose_flow_bwd
            for s in range(NUM_SCALES):
              curr_bs, curr_h, curr_w, _ = self.pose_net_flow_bwd[egomotion_index][s].get_shape().as_list()
              scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), 'float32')
              scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
              self.pose_net_flow_bwd[egomotion_index][s] = self.pose_net_flow_bwd[egomotion_index][s] * scale_factor


            tf.get_variable_scope().reuse_variables()

          egomotion_index_base += egomotion_num


  def build_loss(self):
    """Adds ops for computing loss."""
    with tf.name_scope('compute_loss'):
      #前后帧图片
      self.temporal_reconstr_loss_flow = 0
      self.temporal_ssim_loss_flow = 0
      self.temporal_reconstr_loss_pose_flow = 0
      self.temporal_ssim_loss_pose_flow = 0
      self.temporal_reconstr_loss = 0
      self.temporal_ssim_loss = 0
      self.temporal_egomotion_snap_loss = 0
      self.temporal_egomotion_fwd_bwd_consistency_loss = 0

      #左右帧图片
      self.spatial_reconstr_loss = 0
      self.spatial_left_reconstr_loss = [0 for _ in range(NUM_SCALES)]
      self.spatial_right_reconstr_loss = [0 for _ in range(NUM_SCALES)]
      self.spatial_ssim_loss = 0
      self.spatial_left_ssim_loss = [0 for _ in range(NUM_SCALES)]
      self.spatial_right_ssim_loss = [0 for _ in range(NUM_SCALES)]
      self.spatial_lr_consist_warp_loss = 0
      self.spatial_smooth_loss = 0
      self.spatial_disp_reg_loss = 0
      self.spatial_flow_consistency_mask_loss_reg = 0

      self.icp_transform_loss = 0
      self.icp_residual_loss = 0

      # self.images_left is organized by ...[scale][B, h, w, seq_len * 3].
      self.images_left = [{} for _ in range(NUM_SCALES)]
      self.images_right = [{} for _ in range(NUM_SCALES)]


      self.icp_transform = [{} for _ in range(NUM_SCALES)]
      self.icp_residual = [{} for _ in range(NUM_SCALES)]

      self.middle_frame_index = util.get_seq_middle(self.seq_length)

      with tf.name_scope("sad_kernel"):
        self.sad_loss_kernel_scale_constant = [tf.constant([1.0 / (3 * (7 - s) * (9 - s))], shape=[7 - s, 9 - s, 3, 3])
                                               for s in range(0, 2 * NUM_SCALES, 2)]
        self.sad_loss_kernel_scale = [tf.Variable(k, trainable=False, name="sad_kernel") for k in
                                      self.sad_loss_kernel_scale_constant]

      for s in range(NUM_SCALES):
        # Scale image stack.
        height_s = int(self.img_height / (2**s))
        width_s = int(self.img_width / (2**s))
        self.images_left[s] = tf.image.resize_area(self.image_stack_left, [height_s, width_s])
        self.images_right[s] = tf.image.resize_area(self.image_stack_right, [height_s, width_s])

      if self.train_mode == 'depth_odom':
        self.temporal_wrap()

      # if self.train_mode == 'optical_flow' or self.train_mode == 'depth_odom':
      #   self.direct_flow_loss()

      if self.train_mode == 'depth_odom':
        self.temporal_loss()

      if self.train_mode == 'depth_odom' or self.train_mode == 'depth':
        self.spatial_loss()

      self.total_loss = self.reconstr_weight * (self.temporal_reconstr_loss + self.spatial_reconstr_loss + self.temporal_reconstr_loss_pose_flow)
      self.total_loss += self.ssim_weight * (self.temporal_ssim_loss + self.spatial_ssim_loss + self.temporal_ssim_loss_pose_flow)
      self.total_loss += self.smooth_weight * self.spatial_smooth_loss
      self.total_loss += self.disp_reg_weight * self.spatial_disp_reg_loss
      self.total_loss += self.lr_disp_consistency_weight * self.spatial_lr_consist_warp_loss

      self.temporal_total_loss = self.reconstr_weight * (self.temporal_reconstr_loss + self.temporal_reconstr_loss_pose_flow)
      self.temporal_total_loss += self.ssim_weight * (self.temporal_ssim_loss + self.temporal_ssim_loss_pose_flow)

  def build_train_op(self):
    with tf.name_scope('train_op'):
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      # self.val_loss = tf.Variable(-1, name='global_step', trainable=False)
      self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

      # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "egomotion_prediction")
      boundaries = [np.int32((2 / 6) * self.train_steps),
                    np.int32((3 / 6) * self.train_steps),
                    np.int32((4 / 6) * self.train_steps),
                    np.int32((5 / 6) * self.train_steps)]
      values = [self.learning_rate,
                self.learning_rate / 2,
                self.learning_rate / 4,
                self.learning_rate / 8,
                self.learning_rate / 16]
      self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

      optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
      self.train_op = slim.learning.create_train_op(self.total_loss, optim)
      # train_vars = [var for var in tf.trainable_variables()]
      # self.grads_and_vars = optim.compute_gradients(self.total_loss,var_list=train_vars)
      # self.train_op = optim.apply_gradients(self.grads_and_vars)

  def build_summaries(self):
    """Adds scalar and image summaries for TensorBoard."""
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar('temporal_total_loss', self.temporal_total_loss)
    tf.summary.scalar('spatial_reconstr_loss', self.spatial_reconstr_loss)
    tf.summary.scalar('spatial_lr_consist_warp_loss', self.spatial_lr_consist_warp_loss)
    tf.summary.scalar('spatial_disp_reg_loss', self.spatial_disp_reg_loss)

    if self.train_mode == 'depth_odom':
      tf.summary.scalar('temporal_reconstr_loss', self.temporal_reconstr_loss)
      tf.summary.scalar('temporal_reconstr_loss_pose_flow', self.temporal_reconstr_loss_pose_flow)

    if self.ssim_weight > 0:
      tf.summary.scalar('spatial_ssim_loss', self.spatial_ssim_loss)
      if self.train_mode == 'depth_odom':
        tf.summary.scalar('temporal_ssim_loss', self.temporal_ssim_loss)
        tf.summary.scalar('temporal_ssim_loss_pose_flow', self.temporal_ssim_loss_pose_flow)

    if self.smooth_weight > 0:
      tf.summary.scalar('spatial_smooth_loss', self.spatial_smooth_loss)

    # if self.icp_weight > 0:
    #   tf.summary.scalar('icp_transform_loss', self.icp_transform_loss)
    #   tf.summary.scalar('icp_residual_loss', self.icp_residual_loss)

    tf.summary.scalar('learning_rate', self.learning_rate)

    if self.train_mode == 'depth_odom':
      for i in range(self.seq_length - 1):
        tf.summary.histogram('tx%d' % i, self.egomotion_fwd[i][:, :, 0])
        tf.summary.histogram('ty%d' % i, self.egomotion_fwd[i][:, :, 1])
        tf.summary.histogram('tz%d' % i, self.egomotion_fwd[i][:, :, 2])
        tf.summary.histogram('rx%d' % i, self.egomotion_fwd[i][:, :, 3])
        tf.summary.histogram('ry%d' % i, self.egomotion_fwd[i][:, :, 4])
        tf.summary.histogram('rz%d' % i, self.egomotion_fwd[i][:, :, 5])

        # tf.summary.histogram('tx%d_bwd' % i, self.egomotion_bwd[i][:, :, 0])
        # tf.summary.histogram('ty%d_bwd' % i, self.egomotion_bwd[i][:, :, 1])
        # tf.summary.histogram('tz%d_bwd' % i, self.egomotion_bwd[i][:, :, 2])
        # tf.summary.histogram('rx%d_bwd' % i, self.egomotion_bwd[i][:, :, 3])
        # tf.summary.histogram('ry%d_bwd' % i, self.egomotion_bwd[i][:, :, 4])
        # tf.summary.histogram('rz%d_bwd' % i, self.egomotion_bwd[i][:, :, 5])
      for s in range(NUM_SCALES-3):
        for key in self.temporal_occlusion_object_mask[s]:
          tf.summary.image('scale%d_temporal_occlusion_object_mask_%s' % (s, key),
                           self.temporal_occlusion_object_mask[s][key])

          tf.summary.image('scale%d_temporal_dynamic_object_mask_%s' % (s, key),
                           self.temporal_dynamic_object_mask[s][key])
          tf.summary.image('scale%d_temporal_warp_mask_%s' % (s, key),
                           self.temporal_warp_mask[s][key])
          tf.summary.image('scale%d_temporal_warped_image_%s' % (s, key),
                           self.temporal_warped_image[s][key])
          tf.summary.image('scale%d_temporal_warp_image_pose_flow_%s' % (s, key),
                           self.temporal_warp_image_pose_flow[s][key])



    for s in range(NUM_SCALES):
      tf.summary.scalar('scale%d_spatial_left_reconstr_loss' % s, self.spatial_left_reconstr_loss[s])
      tf.summary.scalar('scale%d_spatial_right_reconstr_loss' % s, self.spatial_right_reconstr_loss[s])
      if self.ssim_weight > 0:
        tf.summary.scalar('scale%d_spatial_left_ssim_loss' % s, self.spatial_left_ssim_loss[s])
        tf.summary.scalar('scale%d_spatial_right_ssim_loss' % s, self.spatial_right_ssim_loss[s])

    for s in range(NUM_SCALES-3):
      for i in range(self.seq_length):
        tf.summary.image('scale%d_images_left%d' % (s, i),
                         self.images_left[s][:, :, :, 3 * i:3 * (i + 1)])
        tf.summary.image('scale%d_mages_right%d' % (s, i),
                         self.images_right[s][:, :, :, 3 * i:3 * (i + 1)])
        if i in self.depth_left:
          depth_clip_left = tf.clip_by_value(self.depth_left[i][s], 0.0, 100.0)
          disp_clip_left = tf.clip_by_value(self.disp_left[i][s], 0.0, 0.3)
          tf.summary.histogram('scale%d_depth_left%d' % (s, i), depth_clip_left)
          tf.summary.histogram('scale%d_disp_left%d' % (s, i), disp_clip_left)
          tf.summary.image('scale%d_disparity_left%d' % (s, i), disp_clip_left)
          depth_clip_right = tf.clip_by_value(self.depth_right[i][s], 0.0, 100.0)
          disp_clip_right = tf.clip_by_value(self.disp_right[i][s], 0.0, 0.3)
          tf.summary.histogram('scale%d_depth_right%d' % (s, i), depth_clip_right)
          tf.summary.histogram('scale%d_disp_right%d' % (s, i), disp_clip_right)
          tf.summary.image('scale%d_disparity_right%d' % (s, i), disp_clip_right)

      if self.train_mode == 'depth_odom' or self.train_mode == 'depth':
        for key in self.spatial_warped_image[s]:
          tf.summary.image('scale%d_spatial_warped_image_%s' % (s, key),
                           self.spatial_warped_image[s][key])
          # tf.summary.image('scale%d_spatial_warp_mask_%s' % (s, key),
          #                  self.spatial_warp_mask[s][key])
          # tf.summary.image('scale%d_spatial_warp_error_%s' % (s, key),
          #                  self.warp_spatial_error[s][key])
          # if self.ssim_weight > 0:
          #   tf.summary.image('scale%d_spatial_ssim_error%s' % (s, key),
          #                    self.ssim_spatial_error[s][key])
          # if self.icp_weight > 0:
          #   tf.summary.image('scale%d_icp_residual%s' % (s, key),
          #                    self.icp_residual[s][key])
          #   transform = self.icp_transform[s][key]
          #   tf.summary.histogram('scale%d_icp_tx%s' % (s, key), transform[:, 0])
          #   tf.summary.histogram('scale%d_icp_ty%s' % (s, key), transform[:, 1])
          #   tf.summary.histogram('scale%d_icp_tz%s' % (s, key), transform[:, 2])
          #   tf.summary.histogram('scale%d_icp_rx%s' % (s, key), transform[:, 3])
          #   tf.summary.histogram('scale%d_icp_ry%s' % (s, key), transform[:, 4])
          #   tf.summary.histogram('scale%d_icp_rz%s' % (s, key), transform[:, 5])

        for key in self.spatial_flow_consistency_mask_left[s]:
          tf.summary.image('scale%d_disp_right2left_%s' % (s, key),
                           self.spatial_right2left_disp[s][key])
          tf.summary.image('scale%d_flow_consistency_mask_left_%s' % (s, key),
                           self.spatial_flow_consistency_mask_left[s][key])
        for key in self.spatial_flow_consistency_mask_right[s]:
          tf.summary.image('scale%d_disp_left2right_%s' % (s, key),
                           self.spatial_left2right_disp[s][key])
          tf.summary.image('scale%d_flow_consistency_mask_right_%s' % (s, key),
                           self.spatial_flow_consistency_mask_right[s][key])

    # for var in tf.trainable_variables():
    #   tf.summary.histogram(var.op.name + "/values", var)
    # for grad, var in self.grads_and_vars:
    #   tf.summary.histogram(var.op.name + "/gradients", grad)

  def direct_flow_loss(self):
    self.temporal_warp_image_flow = [{} for _ in range(NUM_SCALES)]

    self.temporal_warp_error_flow = [{} for _ in range(NUM_SCALES)]
    self.temporal_ssim_error_flow = [{} for _ in range(NUM_SCALES)]
    for s in range(NUM_SCALES):
      with tf.name_scope('flow_loss'):
        egomotion_fwd_index_base = 0
        for step in range(self.max_egomotion_step):
          egomotion_num = self.num_source - step
          if egomotion_num == 0:
            break
          for i in range(egomotion_num):
            egomotion_index = egomotion_fwd_index_base + i
            j = i + 1 + step
            key = '%d-%d' % (i, j)

            # curr_bs, curr_h, curr_w, _ = self.direct_flow_fwd[egomotion_index][s].get_shape().as_list()
            # scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), 'float32')
            # scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            # self.direct_flow_fwd[egomotion_index][s] = self.direct_flow_fwd[egomotion_index][s] * scale_factor

            self.temporal_warp_image_flow[s][key] = project.flow_warp(self.images_left[s][:, :, :, 3 * i:3 * (i + 1)],
                                                                      self.direct_flow_fwd[egomotion_index][s])

            self.temporal_warp_error_flow[s][key], self.temporal_ssim_error_flow[s][key] = self.temporal_image_similarity_flow(
              self.temporal_warp_image_flow[s][key], self.images_left[s][:, :, :, 3 * (j):3 * (j + 1)],
              self.sad_loss_kernel_scale[s])

            self.temporal_reconstr_loss_flow += tf.reduce_mean(self.temporal_warp_error_flow[s][key])
            self.temporal_ssim_loss_flow += tf.reduce_mean(self.temporal_ssim_error_flow[s][key])
          egomotion_fwd_index_base += egomotion_num

        egomotion_bwd_index_base = 0
        for step in range(self.max_egomotion_step):
          egomotion_num = self.num_source - step
          if egomotion_num == 0:
            break
          for j in range(egomotion_num):
            egomotion_index = egomotion_bwd_index_base + j
            i = j + 1 + step
            key = '%d-%d' % (i, j)

            # curr_bs, curr_h, curr_w, _ = self.direct_flow_bwd[egomotion_index][s].get_shape().as_list()
            # scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1, 1, 1, 2]), 'float32')
            # scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            # self.direct_flow_bwd[egomotion_index][s] = self.direct_flow_bwd[egomotion_index][s] * scale_factor

            self.temporal_warp_image_flow[s][key] = project.flow_warp(self.images_left[s][:, :, :, 3 * (i):3 * (i + 1)],
                                                                      self.direct_flow_bwd[egomotion_index][s])

            self.temporal_warp_error_flow[s][key], self.temporal_ssim_error_flow[s][
              key] = self.temporal_image_similarity_flow(
              self.temporal_warp_image_flow[s][key], self.images_left[s][:, :, :, 3 * j:3 * (j + 1)],
              self.sad_loss_kernel_scale[s])

            self.temporal_reconstr_loss_flow += tf.reduce_mean(self.temporal_warp_error_flow[s][key])
            self.temporal_ssim_loss_flow += tf.reduce_mean(self.temporal_ssim_error_flow[s][key])
          egomotion_bwd_index_base += egomotion_num

  def temporal_wrap(self):
    self.temporal_warped_image = [{} for _ in range(NUM_SCALES)]
    self.temporal_rigid_flow = [{} for _ in range(NUM_SCALES)]
    self.temporal_warp_mask = [{} for _ in range(NUM_SCALES)]

    for s in range(NUM_SCALES):
      with tf.name_scope('temporal_wrap'):
        with tf.name_scope('temporal_fwd_wrap'):
          egomotion_fwd_index_base = 0
          for step in range(self.max_egomotion_step):
            egomotion_num = self.num_source - step
            if egomotion_num == 0:
              break
            for i in range(egomotion_num):
              egomotion_index = egomotion_fwd_index_base + i
              j = i + 1 + step
              key = '%d-%d' % (i, j)

              # 位姿光度约束
              egomotion = tf.squeeze(self.egomotion_fwd[egomotion_index])
              self.temporal_warped_image[s][key], self.temporal_warp_mask[s][key], self.temporal_rigid_flow[s][key] = (
                project.inverse_warp(self.images_left[s][:, :, :, 3 * i:3 * (i + 1)],
                                     self.depth_left[j][s],
                                     egomotion,
                                     self.intrinsic_mat[:, s, :, :],
                                     self.intrinsic_mat_inv[:, s, :, :]))

            egomotion_fwd_index_base += egomotion_num


        with tf.name_scope('temporal_bwd_wrap'):
          egomotion_bwd_index_base = 0
          for step in range(self.max_egomotion_step):
            egomotion_num = self.num_source - step
            if egomotion_num == 0:
              break
            for j in range(egomotion_num):
              egomotion_index = egomotion_bwd_index_base + j
              i = j + 1 + step
              key = '%d-%d' % (i, j)

              # 位姿光度约束
              egomotion = tf.squeeze(self.egomotion_bwd[egomotion_index])
              self.temporal_warped_image[s][key], self.temporal_warp_mask[s][key], self.temporal_rigid_flow[s][key] = (
                project.inverse_warp(self.images_left[s][:, :, :, 3 * (i):3 * (i + 1)],
                                     self.depth_left[j][s],
                                     egomotion,
                                     self.intrinsic_mat[:, s, :, :],
                                     self.intrinsic_mat_inv[:, s, :, :]))

            egomotion_bwd_index_base += egomotion_num


  def temporal_loss(self):
    self.temporal_warp_image_pose_flow = [{} for _ in range(NUM_SCALES)]
    self.temporal_warp_error_pose_flow = [{} for _ in range(NUM_SCALES)]
    self.temporal_ssim_error_pose_flow = [{} for _ in range(NUM_SCALES)]

    # self.temporal_warped_image = [{} for _ in range(NUM_SCALES)]
    # self.temporal_rigid_flow = [{} for _ in range(NUM_SCALES)]
    # self.temporal_warp_mask = [{} for _ in range(NUM_SCALES)]

    self.temporal_warp_error = [{} for _ in range(NUM_SCALES)]
    self.temporal_ssim_error = [{} for _ in range(NUM_SCALES)]


    self.temporal_dynamic_object_mask = [{} for _ in range(NUM_SCALES)]
    self.temporal_occlusion_object_mask = [{} for _ in range(NUM_SCALES)]

    if self.egomotion_snap_weight > 0:
      with tf.name_scope('temporal_egomotion_snap_loss'):   #加速度？
        self.temporal_egomotion_snap_loss += self.egomotion_snap_loss(self.egomotion_fwd)
        self.temporal_egomotion_snap_loss += self.egomotion_snap_loss(self.egomotion_bwd)

    if self.fwd_bwd_egomoton_consistency_weight > 0:
      with tf.name_scope('temporal_fwd_bwd_egomoton_consistency'):
        tran_rot_ratio = 20
        egomotion_index_base = 0
        for step in range(self.max_egomotion_step):
          egomotion_num = self.num_source - step
          if egomotion_num == 0:
            break
          for i in range(egomotion_num):
            egomotion_index = egomotion_index_base + i
            egomotion_fwd = tf.squeeze(self.egomotion_fwd[egomotion_index])
            egomotion_bwd = tf.squeeze(self.egomotion_bwd[egomotion_index])
            self.temporal_egomotion_fwd_bwd_consistency_loss += \
              tf.reduce_mean(egomotion_fwd[:, :3] + egomotion_bwd[:, :3]) \
              + tran_rot_ratio * tf.reduce_mean(egomotion_fwd[:, 3:] + egomotion_bwd[:, 3:])

          egomotion_index_base += egomotion_num

    #之后光流的Loss有也放在这里，前向与后向，利用geonet当中的利用光流进行wrap的代码
    for s in range(NUM_SCALES):
      with tf.name_scope('temporal_loss'):
        with tf.name_scope('temporal_fwd_loss'):
          egomotion_fwd_index_base = 0
          for step in range(self.max_egomotion_step):
            egomotion_num = self.num_source - step
            if egomotion_num == 0:
              break
            for i in range(egomotion_num):
              egomotion_index = egomotion_fwd_index_base + i
              j = i + 1 + step
              key = '%d-%d' % (i, j)
              key_inverse = '%d-%d' % (j, i)

              # 位姿光度约束
              self.temporal_dynamic_object_mask[s][key] = self.cal_temporal_dynamic_mask(
                self.temporal_rigid_flow[s][key],
                self.pose_net_flow_bwd[egomotion_index][s],
                # self.direct_flow_fwd[egomotion_index][s],
                s)

              self.temporal_occlusion_object_mask[s][key] = self.cal_temporal_occlusion_mask(
                # self.direct_flow_bwd[egomotion_index][s],
                self.temporal_rigid_flow[s][key_inverse],
                self.temporal_rigid_flow[s][key],
                s)

              self.temporal_warp_error[s][key], self.temporal_ssim_error[s][key] = self.temporal_image_similarity(
                self.temporal_warped_image[s][key], self.images_left[s][:, :, :, 3 * (j):3 * (j + 1)],
                target_images_disp= self.disp_left[j][s],
                sad_loss_kernel=self.sad_loss_kernel_scale[s],
                warp_mask=self.temporal_warp_mask[s][key],  # ICP的黑边mask
                dynamic_mask=self.temporal_dynamic_object_mask[s][key],
                occlusion_mask=self.temporal_occlusion_object_mask[s][key])

              self.temporal_reconstr_loss += tf.reduce_mean(self.temporal_warp_error[s][key])
              self.temporal_ssim_loss += tf.reduce_mean(self.temporal_ssim_error[s][key])

              self.temporal_warp_image_pose_flow[s][key] = project.flow_warp(
                self.images_left[s][:, :, :, 3 * (j):3 * (j + 1)],
                self.pose_net_flow_fwd[egomotion_index][s])

              self.temporal_warp_error_pose_flow[s][key], self.temporal_ssim_error_pose_flow[s][
                key] = self.temporal_image_similarity_flow(
                self.temporal_warp_image_pose_flow[s][key], self.images_left[s][:, :, :, 3 * i:3 * (i + 1)],
                self.sad_loss_kernel_scale[s],
                warp_mask=self.temporal_warp_mask[s][key_inverse])

              self.temporal_reconstr_loss_pose_flow += tf.reduce_mean(self.temporal_warp_error_pose_flow[s][key])
              self.temporal_ssim_loss_pose_flow += tf.reduce_mean(self.temporal_ssim_error_pose_flow[s][key])


            egomotion_fwd_index_base += egomotion_num


        with tf.name_scope('temporal_bwd_loss'):
          egomotion_bwd_index_base = 0
          for step in range(self.max_egomotion_step):
            egomotion_num = self.num_source - step
            if egomotion_num == 0:
              break
            for j in range(egomotion_num):
              egomotion_index = egomotion_bwd_index_base + j
              i = j + 1 + step
              key = '%d-%d' % (i, j)
              key_inverse = '%d-%d' % (j, i)

              # 位姿光度约束
              self.temporal_dynamic_object_mask[s][key] = self.cal_temporal_dynamic_mask(
                self.temporal_rigid_flow[s][key],
                self.pose_net_flow_fwd[egomotion_index][s],
                # self.direct_flow_bwd[egomotion_index][s],
                s)

              self.temporal_occlusion_object_mask[s][key] = self.cal_temporal_occlusion_mask(
                # self.direct_flow_fwd[egomotion_index][s],
                self.temporal_rigid_flow[s][key_inverse],
                self.temporal_rigid_flow[s][key],
                s)

              self.temporal_warp_error[s][key], self.temporal_ssim_error[s][key] = self.temporal_image_similarity(
                self.temporal_warped_image[s][key], self.images_left[s][:, :, :, 3 * j:3 * (j + 1)],
                target_images_disp=self.disp_left[j][s],
                sad_loss_kernel=self.sad_loss_kernel_scale[s],
                warp_mask=self.temporal_warp_mask[s][key],
                dynamic_mask=self.temporal_dynamic_object_mask[s][key],
                occlusion_mask=self.temporal_occlusion_object_mask[s][key])

              self.temporal_reconstr_loss += tf.reduce_mean(
                self.temporal_warp_error[s][key])
              self.temporal_ssim_loss += tf.reduce_mean(
                self.temporal_ssim_error[s][key])

              self.temporal_warp_image_pose_flow[s][key] = project.flow_warp(
                self.images_left[s][:, :, :, 3 * j:3 * (j + 1)],
                self.pose_net_flow_bwd[egomotion_index][s])

              self.temporal_warp_error_pose_flow[s][key], self.temporal_ssim_error_pose_flow[s][
                key] = self.temporal_image_similarity_flow(
                self.temporal_warp_image_pose_flow[s][key], self.images_left[s][:, :, :, 3 * (i):3 * (i + 1)],
                self.sad_loss_kernel_scale[s],
                warp_mask=self.temporal_warp_mask[s][key_inverse])

              self.temporal_reconstr_loss_pose_flow += tf.reduce_mean(self.temporal_warp_error_pose_flow[s][key])
              self.temporal_ssim_loss_pose_flow += tf.reduce_mean(self.temporal_ssim_error_pose_flow[s][key])

            egomotion_bwd_index_base += egomotion_num



  def spatial_loss(self):
    self.spatial_warped_image = [{} for _ in range(NUM_SCALES)]
    self.spatial_right2left_disp = [{} for _ in range(NUM_SCALES)]
    self.spatial_left2right_disp = [{} for _ in range(NUM_SCALES)]

    self.spatial_rigid_flow = [{} for _ in range(NUM_SCALES)]
    self.spatial_flow_consistency_mask_left = [{} for _ in range(NUM_SCALES)]
    self.spatial_flow_consistency_mask_right = [{} for _ in range(NUM_SCALES)]
    self.spatial_flow_consistency_mask_left_inverse = [{} for _ in range(NUM_SCALES)]
    self.spatial_flow_consistency_mask_right_inverse = [{} for _ in range(NUM_SCALES)]
    self.spatial_lr_consist_warp_error = [{} for _ in range(NUM_SCALES)]

    self.spatial_warp_mask = [{} for _ in range(NUM_SCALES)]
    self.spatial_warp_error = [{} for _ in range(NUM_SCALES)]
    self.spatial_ssim_error = [{} for _ in range(NUM_SCALES)]

    rwd2lwd = [{} for _ in range(NUM_SCALES)]
    lwd2rwd = [{} for _ in range(NUM_SCALES)]
    lwd_flow_diff = [{} for _ in range(NUM_SCALES)]
    rwd_flow_diff = [{} for _ in range(NUM_SCALES)]
    lwd_consist_bound = [{} for _ in range(NUM_SCALES)]
    rwd_consist_bound = [{} for _ in range(NUM_SCALES)]

    for s in range(NUM_SCALES):
      with tf.name_scope('spatial_loss'):
        with tf.name_scope('Smoothness'):
          if self.smooth_weight > 0:
            for i in range(self.seq_length):
              # In legacy mode, use the depth map from the middle frame only.
              if not self.legacy_mode or i == self.middle_frame_index:
                self.spatial_smooth_loss += 1.0 / (2 ** s) * self.depth_smoothness(
                  self.disp_left[i][s], self.images_left[s][:, :, :, 3 * i:3 * (i + 1)])
                self.spatial_smooth_loss += 1.0 / (2 ** s) * self.depth_smoothness(
                  self.disp_right[i][s], self.images_right[s][:, :, :, 3 * i:3 * (i + 1)])

        with tf.name_scope('spatial_left_right'):
          for j in range(self.seq_length):
            key = 'lwd-%d' % (j)

            # Inverse warp the source_right image to the target_left image frame for
            # photometric consistency loss.
            self.spatial_warped_image[s][key], self.spatial_warp_mask[s][key], self.spatial_rigid_flow[s][key] = (
              project.inverse_warp(self.images_right[s][:, :, :, 3 * j:3 * (j + 1)],
                                   self.depth_left[j][s],
                                   self.t_r2l,
                                   self.intrinsic_mat[:, s, :, :],
                                   self.intrinsic_mat_inv[:, s, :, :]))

            self.spatial_warp_error[s][key], self.spatial_ssim_error[s][key] = self.spatial_image_similarity(
              self.spatial_warped_image[s][key], self.images_left[s][:, :, :, 3 * j:3 * (j + 1)],
              target_images_disp=self.disp_left[j][s],
              sad_loss_kernel=self.sad_loss_kernel_scale[s])

          for j in range(self.seq_length):
            key = 'rwd-%d' % (j)

            # Inverse warp the source_right image to the target_right image frame for
            # photometric consistency loss.
            self.spatial_warped_image[s][key], self.spatial_warp_mask[s][key], self.spatial_rigid_flow[s][key] = (
              project.inverse_warp(self.images_left[s][:, :, :, 3 * j:3 * (j + 1)],
                                   self.depth_right[j][s],
                                   self.t_l2r,
                                   self.intrinsic_mat[:, s, :, :],
                                   self.intrinsic_mat_inv[:, s, :, :]))

            self.spatial_warp_error[s][key], self.spatial_ssim_error[s][key] = self.spatial_image_similarity(
              self.spatial_warped_image[s][key], self.images_right[s][:, :, :, 3 * j:3 * (j + 1)],
              target_images_disp=self.disp_right[j][s],
              sad_loss_kernel=self.sad_loss_kernel_scale[s])

        with tf.name_scope('left_right_occlusion'):
          for j in range(self.seq_length):
            key_lwd = 'lwd-%d' % (j)
            key_rwd = 'rwd-%d' % (j)
            rwd2lwd[s][key_lwd] = project.flow_warp(self.spatial_rigid_flow[s][key_rwd], self.spatial_rigid_flow[s][key_lwd])
            lwd2rwd[s][key_rwd] = project.flow_warp(self.spatial_rigid_flow[s][key_lwd], self.spatial_rigid_flow[s][key_rwd])
            # calculate flow diff
            # # # clip flow to image area
            # cliped_lwd_flow = self.clip_rigid_flow(lwd_flow)
            # cliped_rwd_flow = self.clip_rigid_flow(rwd_flow)
            lwd_flow_diff[s][key_lwd] = tf.abs(rwd2lwd[s][key_lwd] + self.spatial_rigid_flow[s][key_lwd])
            rwd_flow_diff[s][key_rwd] = tf.abs(lwd2rwd[s][key_rwd] + self.spatial_rigid_flow[s][key_rwd])
            # build flow consistency condition
            lwd_consist_bound[s][key_lwd] = 0.04 * self.L2_norm(self.spatial_rigid_flow[s][key_lwd]) * 2 ** s
            rwd_consist_bound[s][key_rwd] = 0.04 * self.L2_norm(self.spatial_rigid_flow[s][key_rwd]) * 2 ** s
            lwd_consist_bound[s][key_lwd] = tf.stop_gradient(tf.maximum(lwd_consist_bound[s][key_lwd], 3.0))
            rwd_consist_bound[s][key_rwd] = tf.stop_gradient(tf.maximum(rwd_consist_bound[s][key_rwd], 3.0))
            # build flow consistency mask
            self.spatial_flow_consistency_mask_left[s][key_lwd] = tf.stop_gradient(
              tf.cast(tf.less(self.L2_norm(lwd_flow_diff[s][key_lwd]) * 2 ** s,
                              lwd_consist_bound[s][key_lwd]), tf.float32))
            self.spatial_flow_consistency_mask_right[s][key_rwd] = tf.stop_gradient(
              tf.cast(tf.less(self.L2_norm(rwd_flow_diff[s][key_rwd]) * 2 ** s,
                              rwd_consist_bound[s][key_rwd]), tf.float32))

            # mask的地方为1，非mask的地方为0
            self.spatial_flow_consistency_mask_left_inverse[s][key_lwd] = tf.stop_gradient(
              tf.cast(tf.greater(self.L2_norm(lwd_flow_diff[s][key_lwd]) * 2 ** s,
                                 lwd_consist_bound[s][key_lwd]), tf.float32))
            self.spatial_flow_consistency_mask_right_inverse[s][key_rwd] = tf.stop_gradient(
              tf.cast(tf.greater(self.L2_norm(rwd_flow_diff[s][key_rwd]) * 2 ** s,
                                 rwd_consist_bound[s][key_rwd]), tf.float32))

            # if self.use_flow_consistency_mask is True:
            #   ref_flow_consistency_mask = self.get_reference_explain_mask(s)
            #   self.spatial_flow_consistency_mask_loss_reg += self.compute_exp_reg_loss(
            #     self.spatial_flow_consistency_mask_left[s][key_lwd], ref_flow_consistency_mask)
            #   self.spatial_flow_consistency_mask_loss_reg += self.compute_exp_reg_loss(
            #     self.spatial_flow_consistency_mask_right[s][key_rwd], ref_flow_consistency_mask)

            # restruct errer
            if self.use_flow_consistency_mask is True:
              self.spatial_warp_error[s][key_lwd] *= self.spatial_flow_consistency_mask_left[s][key_lwd]
              self.spatial_warp_error[s][key_rwd] *= self.spatial_flow_consistency_mask_right[s][key_rwd]
            self.spatial_left_reconstr_loss[s] += tf.reduce_mean(self.spatial_warp_error[s][key_lwd])
            self.spatial_right_reconstr_loss[s] += tf.reduce_mean(self.spatial_warp_error[s][key_rwd])

            self.spatial_reconstr_loss += tf.reduce_mean(self.spatial_warp_error[s][key_lwd])
            self.spatial_reconstr_loss += tf.reduce_mean(self.spatial_warp_error[s][key_rwd])
            # SSIM error
            if self.use_flow_consistency_mask is True:
              self.spatial_ssim_error[s][key_lwd] *= slim.avg_pool2d(self.spatial_flow_consistency_mask_left[s][key_lwd], 3, 1,
                                                             'VALID')
              self.spatial_ssim_error[s][key_rwd] *= slim.avg_pool2d(self.spatial_flow_consistency_mask_right[s][key_rwd], 3, 1,
                                                             'VALID')
            self.spatial_left_ssim_loss[s] += tf.reduce_mean(self.spatial_ssim_error[s][key_lwd])
            self.spatial_right_ssim_loss[s] += tf.reduce_mean(self.spatial_ssim_error[s][key_rwd])

            self.spatial_ssim_loss += tf.reduce_mean(self.spatial_ssim_error[s][key_lwd])
            self.spatial_ssim_loss += tf.reduce_mean(self.spatial_ssim_error[s][key_rwd])

        with tf.name_scope('disp_regularization'):
          # disp regularization Loss from DVSO add a weight on each disp(negtive exp)
          if self.disp_reg_weight > 0:
            for j in range(self.seq_length):
              key_lwd = 'lwd-%d' % (j)
              key_rwd = 'rwd-%d' % (j)
              weights_left = tf.stop_gradient(
                tf.exp(tf.reduce_mean(tf.abs(self.disp_left[j][s]), 3, keepdims=True)))
              disp_reg_left = self.disp_left[j][s] * self.spatial_flow_consistency_mask_left_inverse[s][
                key_lwd]
              weights_right = tf.stop_gradient(
                tf.exp(tf.reduce_mean(tf.abs(self.disp_right[j][s]), 3, keepdims=True)))
              disp_reg_right = self.disp_right[j][s] * \
                               self.spatial_flow_consistency_mask_right_inverse[s][key_rwd]
              self.spatial_disp_reg_loss += tf.reduce_mean(tf.abs(disp_reg_left))
              self.spatial_disp_reg_loss += tf.reduce_mean(tf.abs(disp_reg_right))

        with tf.name_scope('left_right_disp_consistency'):
          for j in range(self.seq_length):
            key_lwd = 'lwd-%d' % (j)
            key_rwd = 'rwd-%d' % (j)

            self.spatial_right2left_disp[s][key_lwd] = project.flow_warp(self.disp_right[j][s], self.spatial_rigid_flow[s][key_lwd])
            self.spatial_left2right_disp[s][key_rwd] = project.flow_warp(self.disp_left[j][s], self.spatial_rigid_flow[s][key_rwd])

            self.spatial_lr_consist_warp_error[s][key_lwd] = tf.abs(self.spatial_right2left_disp[s][key_lwd] - self.disp_left[j][s])
            self.spatial_lr_consist_warp_error[s][key_rwd] = tf.abs(self.spatial_left2right_disp[s][key_rwd] - self.disp_right[j][s])

            if self.use_charbonnier_loss is True:
              self.spatial_lr_consist_warp_error[s][key_lwd] = self.charbonnier_loss(self.spatial_lr_consist_warp_error[s][key_lwd])
              self.spatial_lr_consist_warp_error[s][key_rwd] = self.charbonnier_loss(self.spatial_lr_consist_warp_error[s][key_rwd])
            if self.use_flow_consistency_mask is True:
              self.spatial_lr_consist_warp_error[s][key_lwd] *= self.spatial_flow_consistency_mask_left[s][key_lwd]
              self.spatial_lr_consist_warp_error[s][key_rwd] *= self.spatial_flow_consistency_mask_right[s][key_rwd]

            self.spatial_lr_consist_warp_loss += tf.reduce_mean(self.spatial_lr_consist_warp_error[s][key_lwd])
            self.spatial_lr_consist_warp_loss += tf.reduce_mean(self.spatial_lr_consist_warp_error[s][key_rwd])


  def charbonnier_loss(self, x, mask=None, alpha=0.45, beta=1.0, epsilon=0.01):
    with tf.variable_scope('charbonnier_loss'):
      error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)
      if mask is not None:
        error = tf.multiply(mask, error)
      return error

  def get_reference_explain_mask(self, downscaling):
    tmp = np.array([1])
    ref_exp_mask = np.tile(tmp,
                           (self.batch_size,
                            int(self.img_height / (2 ** downscaling)),
                            int(self.img_width / (2 ** downscaling)),
                            1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask

  def compute_exp_reg_loss(self, pred, ref):
    # l = tf.nn.softmax_cross_entropy_with_logits(
    #   labels=tf.reshape(ref, [-1, 1]),
    #   logits=tf.reshape(pred, [-1, 1]))
    loss = tf.square(pred - ref)
    return tf.reduce_mean(loss)

  def clip_rigid_flow(self, flow):
    coords_x, coords_y = tf.split(flow, [1, 1], axis=3)

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    y_max = tf.cast(tf.shape(flow)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(flow)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    coords_x_safa = tf.clip_by_value(coords_x, zero, x_max)
    coords_y_safe = tf.clip_by_value(coords_y, zero, y_max)
    clip_flow = tf.concat([coords_x_safa, coords_y_safe], axis=3)

    return clip_flow

  def make_t_r2l(self, baseline):
    zeros = tf.zeros_like(baseline)
    t_l2r = tf.concat([-baseline, zeros, zeros, zeros, zeros, zeros], axis=1)
    return t_l2r

  def L2_norm(self, x, axis=3, keepdims=True):
    curr_offset = 1e-10
    l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keepdims=keepdims)
    return l2_norm

  def gradient_x(self, img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

  def gradient_y(self, img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

  def depth_smoothness(self, depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = self.gradient_x(depth)
    depth_dy = self.gradient_y(depth)
    image_dx = self.gradient_x(img)
    image_dy = self.gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    if self.use_charbonnier_loss is True:
      return tf.reduce_mean(self.charbonnier_loss(abs(smoothness_x))) + tf.reduce_mean(
        self.charbonnier_loss(abs(smoothness_y)))
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

  def egomotion_snap_loss(self, egomotion):
    #注意这里输入的egomotion的长度至少为4
    egomotion_acc = []
    for i in range(self.num_source-1):
      acc = egomotion[i+1] - egomotion[i]
      egomotion_acc.append(acc)

    egomotion_jerk = []
    for i in range(self.num_source - 2):
      jerk = egomotion_acc[i + 1] - egomotion_acc[i]
      egomotion_jerk.append(jerk)

    egomotion_snap = egomotion_jerk[1] - egomotion_jerk[0]
    snap_loss = tf.reduce_mean(tf.square(egomotion_snap))
    return snap_loss

  def temporal_image_similarity_flow(self, temporal_warp_image, target_images, sad_loss_kernel = None,
                                     warp_mask=None):
    # Reconstruction.
    temporal_warp_error = tf.abs(
      temporal_warp_image - target_images)
    # use_charbonnier_loss
    if self.use_charbonnier_loss is True:
      temporal_warp_error = self.charbonnier_loss(temporal_warp_error)
    # sad_loss
    if self.sad_loss is True:
      temporal_warp_error = tf.nn.conv2d(temporal_warp_error,  sad_loss_kernel, [1, 1, 1, 1], padding='SAME')
    if self.use_geometry_mask is True and warp_mask is not None:
      temporal_warp_error = temporal_warp_error * warp_mask

    # SSIM.
    temporal_ssim_error = ()
    if self.ssim_weight > 0:
      temporal_ssim_error = self.ssim(temporal_warp_image, target_images)
      # use_charbonnier_loss
      if self.use_charbonnier_loss is True:
        temporal_ssim_error = self.charbonnier_loss(temporal_ssim_error)
      if self.sad_loss is True:
        temporal_ssim_error = tf.nn.conv2d(temporal_ssim_error, sad_loss_kernel, [1, 1, 1, 1], padding='SAME')

      # TODO(rezama): This should be min_pool2d().
      if self.use_geometry_mask is True  and warp_mask is not None:
        temporal_ssim_error = temporal_ssim_error * slim.avg_pool2d(warp_mask, 3, 1, 'VALID')

    return temporal_warp_error, temporal_ssim_error

  def temporal_image_similarity(self, temporal_warp_image, target_images,
                                target_images_disp=None,
                                sad_loss_kernel=None,
                                warp_mask=None,
                                dynamic_mask=None,
                                occlusion_mask=None):
    # Reconstruction.
    temporal_warp_error = tf.abs(
      temporal_warp_image - target_images)
    # use_charbonnier_loss
    if self.use_charbonnier_loss is True:
      temporal_warp_error = self.charbonnier_loss(temporal_warp_error)
    # sad_loss
    if self.sad_loss is True:
      temporal_warp_error = tf.nn.conv2d(temporal_warp_error, sad_loss_kernel, [1, 1, 1, 1], padding='SAME')
    # if self.use_disp_weight is True:
    #   weights_disp = tf.stop_gradient(tf.exp(-tf.reduce_mean(tf.abs(target_images_disp), 3, keepdims=True)))
    #   temporal_warp_error = temporal_warp_error * weights_disp

    if self.use_geometry_mask is True:
      temporal_warp_error = temporal_warp_error * warp_mask
    if self.use_temporal_dynamic_mask:
      temporal_warp_error = temporal_warp_error * dynamic_mask
    if self.use_temporal_occlusion_mask is True:
      temporal_warp_error = temporal_warp_error * occlusion_mask

    # SSIM.
    temporal_ssim_error = ()
    if self.ssim_weight > 0:
      temporal_ssim_error = self.ssim(temporal_warp_image, target_images)
      # use_charbonnier_loss
      if self.use_charbonnier_loss is True:
        temporal_ssim_error = self.charbonnier_loss(temporal_ssim_error)
      if self.sad_loss is True:
        temporal_ssim_error = tf.nn.conv2d(temporal_ssim_error, sad_loss_kernel, [1, 1, 1, 1], padding='SAME')
      # if self.use_disp_weight is True:
      #   weights_disp = tf.stop_gradient(tf.exp(-tf.reduce_mean(tf.abs(target_images_disp), 3, keepdims=True)))
      #   temporal_ssim_error = temporal_ssim_error * slim.avg_pool2d(weights_disp, 3, 1, 'VALID')

      # TODO(rezama): This should be min_pool2d().
      if self.use_geometry_mask is True:
        temporal_ssim_error = temporal_ssim_error * slim.avg_pool2d(warp_mask, 3, 1, 'VALID')
      if self.use_temporal_dynamic_mask:
        temporal_ssim_error = temporal_ssim_error * slim.avg_pool2d(dynamic_mask, 3, 1, 'VALID')
      if self.use_temporal_occlusion_mask is True:
        temporal_ssim_error = temporal_ssim_error * slim.avg_pool2d(occlusion_mask, 3, 1, 'VALID')

    return temporal_warp_error, temporal_ssim_error

  def spatial_image_similarity(self, spatial_warp_image, target_images, target_images_disp = None, sad_loss_kernel = None):
    # Reconstruction.
    spatial_warp_error = tf.abs(
      spatial_warp_image - target_images)
    # use_charbonnier_loss
    if self.use_charbonnier_loss is True:
      spatial_warp_error = self.charbonnier_loss(spatial_warp_error)
    # sad_loss
    if self.sad_loss is True:
      spatial_warp_error = tf.nn.conv2d(spatial_warp_error,  sad_loss_kernel, [1, 1, 1, 1], padding='SAME')

    if self.use_disp_weight is True:
      weights_disp = tf.stop_gradient(tf.exp(-tf.reduce_mean(tf.abs(target_images_disp), 3, keepdims=True)))
      spatial_warp_error = spatial_warp_error * weights_disp


    # SSIM.
    spatial_ssim_error = ()
    if self.ssim_weight > 0:
      spatial_ssim_error = self.ssim(spatial_warp_image, target_images)
      # use_charbonnier_loss
      if self.use_charbonnier_loss is True:
        spatial_ssim_error = self.charbonnier_loss(spatial_ssim_error)
      if self.sad_loss is True:
        spatial_ssim_error = tf.nn.conv2d(spatial_ssim_error, sad_loss_kernel, [1, 1, 1, 1], padding='SAME')

      if self.use_disp_weight is True:
        weights_disp = tf.stop_gradient(tf.exp(-tf.reduce_mean(tf.abs(target_images_disp), 3, keepdims=True)))
        spatial_ssim_error = spatial_ssim_error * slim.avg_pool2d(weights_disp, 3, 1, 'VALID')

    return spatial_warp_error, spatial_ssim_error

  def cal_temporal_occlusion_mask(self, source_flow, target_flow, scale, bata = 3):
    sflow2tflow = project.flow_warp(source_flow, target_flow)
    flow_diff = tf.abs(sflow2tflow + target_flow)
    consist_bound = 0.05 * self.L2_norm(target_flow) * 2 ** scale
    consist_bound = tf.stop_gradient(tf.maximum(consist_bound, bata))
    mask = tf.stop_gradient(tf.cast(tf.less(self.L2_norm(flow_diff) * 2 ** scale, consist_bound), tf.float32))

    return mask

  def cal_temporal_dynamic_mask(self, direct_flow, pose_flow, scale, bata = 5):
    flow_diff = tf.abs(pose_flow - direct_flow)
    consist_bound = 0.05 * self.L2_norm(pose_flow) * 2 ** scale
    consist_bound = tf.stop_gradient(tf.maximum(consist_bound, bata))
    mask = tf.stop_gradient(tf.cast(tf.less(self.L2_norm(flow_diff) * 2 ** scale, consist_bound), tf.float32))

    return mask


  def ssim(self, x: object, y: object) -> object:
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)

  def build_depth_test_graph(self):
    """Builds depth model reading from placeholders."""
    with tf.variable_scope('depth_prediction'):
      input_uint8 = tf.placeholder(
        tf.uint8, [self.batch_size, self.img_height, self.img_width, 3],
        name='raw_input')
      input_float = tf.image.convert_image_dtype(input_uint8, tf.float32)
      # TODO(rezama): Retrain published model with batchnorm params and set
      # is_training to False.
      # est_disp, _ = nets.disp_net(input_float, is_training=True)
      disp_net_modo = nets.disp_net_monodepth()
      est_disp, _ = disp_net_modo.build_vgg(input_float,get_pred=disp_net_modo.get_disp)
      # est_disp, _ = nets.geo_disp_net(input_float)
      est_depth = 1.0 / est_disp[0]

    self.inputs_depth = input_uint8
    self.est_depth = est_depth

  def build_egomotion_test_graph(self):
    """Builds egomotion model reading from placeholders."""
    input_uint8 = tf.placeholder(
        tf.uint8,
        [self.batch_size, self.img_height, self.img_width * self.seq_length, 3],
        name='raw_input')
    input_float = tf.image.convert_image_dtype(input_uint8, tf.float32)
    image_seq = input_float
    image_stack = self.unpack_image_batches(image_seq)
    with tf.variable_scope('egomotion_prediction'):
        # TODO(rezama): Retrain published model with batchnorm params and set
        # is_training to False.
      egomotion, _, _ = nets.egomotion_net(image_stack, is_training=True,
                                        legacy_mode=self.legacy_mode)
    self.inputs_egomotion = input_uint8
    self.est_egomotion = egomotion

  def build_pose_flow_test_graph(self):
    """Builds flow model reading from placeholders."""
    input_uint8 = tf.placeholder(
      tf.uint8,
      [self.batch_size, self.img_height, self.img_width * self.seq_length, 3],
      name='raw_input')
    input_float = tf.image.convert_image_dtype(input_uint8, tf.float32)
    image_seq = input_float
    image_stack = self.unpack_image_batches(image_seq)
    with tf.variable_scope('egomotion_prediction'):
      _, flow, _ = nets.egomotion_net(image_stack, is_training=True,
                                        legacy_mode=self.legacy_mode)
    self.inputs_egomotion_flow = input_uint8
    self.est_flow = flow

  def unpack_image_batches(self, image_seq):
    """[B, h, w * seq_length, 3] -> [B, h, w, 3 * seq_length]."""
    with tf.name_scope('unpack_images'):
      image_list = [
          image_seq[:, :, i * self.img_width:(i + 1) * self.img_width, :]
          for i in range(self.seq_length)
      ]
      image_stack = tf.concat(image_list, axis=3)
      image_stack.set_shape([
          self.batch_size, self.img_height, self.img_width, self.seq_length * 3
      ])
    return image_stack

  def inference(self, inputs, sess, mode):
    """Runs depth or egomotion inference from placeholders."""
    fetches = {}
    if mode == 'depth':
      fetches['depth'] = self.est_depth
      inputs_ph = self.inputs_depth
    if mode == 'egomotion':
      fetches['egomotion'] = self.est_egomotion
      inputs_ph = self.inputs_egomotion
    if mode =='flow':
      fetches['flow'] = self.est_flow
      inputs_ph = self.inputs_egomotion_flow
    results = sess.run(fetches, feed_dict={inputs_ph: inputs})
    return results
