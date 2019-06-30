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

"""Train the model."""

# Example usage:
#
# python train.py \
#   --logtostderr \
#   --data_dir ~/vid2depth/data/kitti_raw_eigen \
#   --seq_length 3 \
#   --reconstr_weight 0.85 \
#   --smooth_weight 0.05 \
#   --ssim_weight 0.15 \
#   --icp_weight 0.1 \
#   --checkpoint_dir ~/vid2depth/checkpoints

# --pretrained_ckpt
# /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat_relu_sigmoid_upconv_sad_left_right/depth_model_kitti_2015_ssim085_015/pretrained_model_relu_sigmoid_deconv/model-7494

# --data_dir /media/wuqi/works/dataset/kitti_raw_stereo_416_128/train --seq_length 1 --batch_size 12 --checkpoint_dir /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat_relu_relu_upconv_sad_left_right/depth_model_kitti_flip --train_mode "depth"


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import model
import reader
import numpy as np
import tensorflow as tf
import util
from logger import OptFlowTBLogger
from optflow import flow_to_img
slim = tf.contrib.slim
gfile = tf.gfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


HOME_DIR = os.path.expanduser('~')
DEFAULT_DATA_DIR = os.path.join(HOME_DIR, 'vid2depth/data/kitti_raw_eigen')
DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')

flags.DEFINE_string('data_dir',                  DEFAULT_DATA_DIR, 'Preprocessed data.')
flags.DEFINE_string('train_mode',                'depth_odom', 'depth_odom or depth')
flags.DEFINE_string('pretrained_ckpt_depthnet',           None, 'Path to checkpoint with '
                    'depth_net pretrained weights. Do not include .data* extension.')
flags.DEFINE_string('pretrained_ckpt_pwcnet',             None, 'Path to checkpoint with '
                    'flow_net pretrained weights.  Do not include .data* extension.')
flags.DEFINE_string('checkpoint_dir',            DEFAULT_CHECKPOINT_DIR, 'Directory to save model checkpoints.')

flags.DEFINE_float('learning_rate',              0.0002, 'Adam learning rate. 0.0002,0.00005')
flags.DEFINE_float('beta1',                      0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight',            0.15, 'Frame reconstruction loss weight.')
flags.DEFINE_float('smooth_weight',              0.1, 'Smoothness loss weight.')
flags.DEFINE_float('ssim_weight',                0.85, 'SSIM loss weight.')
flags.DEFINE_float('disp_reg_weight',            0.02, 'disp_reg_weight. 0.05')
flags.DEFINE_float('lr_disp_consistency_weight', 0.4, 'lr_disp_consistency_weight 0.4')

flags.DEFINE_bool('use_charbonnier_loss',        True, ' if using or not')
flags.DEFINE_bool('use_geometry_mask',           True,  ' if using or not')
flags.DEFINE_bool('use_flow_consistency_mask',   True,  ' if using or not')
flags.DEFINE_bool('use_temporal_dynamic_mask',   False, ' if using or not')
flags.DEFINE_bool('use_temporal_occlusion_mask', False, ' if using or not')
flags.DEFINE_bool('legacy_mode',                 False, 'Whether to limit losses to using only '
                  'the middle frame in sequence as the target frame.')
flags.DEFINE_bool('use_dense_cx',                True,  'use model with dense connections (4705064 params w/o, '
                                                        '9374274 params with (no residual conn.) for PWC-NET')
flags.DEFINE_bool('use_res_cx',                  True,  'use model with residual connections (4705064 params w/o, '
                                                        '6774064 params with (+2069000) (no dense conn.) for PWC-NET')

flags.DEFINE_integer('batch_size',               8, 'The size of a sample batch')
flags.DEFINE_integer('img_height',               256, 'Input frame height.')
flags.DEFINE_integer('img_width',                512, 'Input frame width.')
flags.DEFINE_integer('seq_length',               1, 'Number of frames in sequence.')
flags.DEFINE_integer('max_egomotion_step',       1, 'max_egomotion_step.')
flags.DEFINE_integer('train_steps',              180000, 'Number of training steps. 180000,300000')
flags.DEFINE_integer("epoch",                    50, "Maximum epoch of training iterations")
flags.DEFINE_integer('summary_freq',             400, 'Save summaries every N steps.')

flags.DEFINE_integer('pyr_lvls',                 6,  'which level to upsample to generate the final optical flow prediction')
flags.DEFINE_integer('flow_pred_lvl',            2,  'which level to upsample to generate the final optical flow prediction')
flags.DEFINE_integer('search_range',             4, 'cost volume search range')

FLAGS = flags.FLAGS


'''
--data_dir
/home/lli/kitti__raw/kitti_odometry/flowodometry_split_skip1_00_10_416_128_vid2pose/train
--seq_length
2
--batch_size
4
--checkpoint_dir
/home/lli/tensorflow/vid2pose_log/odpm_00_10_skip1_relu_relu_deconv_1
--pretrained_ckpt_pwcnet
/home/lli/tensorflow/PWC-Net/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000
--pretrained_ckpt_depthnet
/home/lli/tensorflow/vid2pose/checkpoints/depth_model/model-minloss-68400
--train_mode
"depth_odom"
'''

# Maximum number of checkpoints to keep.
MAX_TO_KEEP = 25
NUM_SCALES = 4


def main(_):
  # Fixed seed for repeatability
  seed = 8964
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if not gfile.Exists(FLAGS.checkpoint_dir):
    gfile.MakeDirs(FLAGS.checkpoint_dir)

  # Write all hyperparameters to record_path
  mode = 'a' #if FLAGS.resume else 'w'
  with open(FLAGS.checkpoint_dir + '/训练参数.txt', mode) as f:
      f.write('\n' + '=' * 50 + '\n')
      f.write('\n'.join("%s: %s" % item for item in FLAGS.flag_values_dict().items()))
      f.write('\n' + '=' * 50 + '\n')

  train()


def train():
  """Train model."""

  #Load data
  reader_data = reader.DataReader(FLAGS.data_dir, FLAGS.batch_size,
                    FLAGS.img_height, FLAGS.img_width,
                    FLAGS.seq_length, NUM_SCALES)
  image_stack_train, intrinsic_mat_train, intrinsic_mat_inv_train = reader_data.read_data()
  image_stack_test, intrinsic_mat_test, intrinsic_mat_inv_test = reader_data.read_data_test()

  steps_per_epoch = reader_data.steps_per_epoch
  # train_steps = FLAGS.epoch * steps_per_epoch

  train_steps = FLAGS.train_steps

  #Model
  train_model = model.Model(data_dir=FLAGS.data_dir,
                            is_training=True,
                            train_steps=train_steps,
                            learning_rate=FLAGS.learning_rate,
                            beta1=FLAGS.beta1,
                            reconstr_weight=FLAGS.reconstr_weight,
                            smooth_weight=FLAGS.smooth_weight,
                            ssim_weight=FLAGS.ssim_weight,
                            disp_reg_weight=FLAGS.disp_reg_weight,
                            lr_disp_consistency_weight=FLAGS.lr_disp_consistency_weight,
                            batch_size=FLAGS.batch_size,
                            img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            seq_length=FLAGS.seq_length,
                            max_egomotion_step=FLAGS.max_egomotion_step,
                            train_mode=FLAGS.train_mode,
                            use_charbonnier_loss=FLAGS.use_charbonnier_loss,
                            use_geometry_mask=FLAGS.use_geometry_mask,
                            use_flow_consistency_mask=FLAGS.use_flow_consistency_mask,
                            use_temporal_dynamic_mask=FLAGS.use_temporal_dynamic_mask,
                            use_temporal_occlusion_mask=FLAGS.use_temporal_occlusion_mask,
                            legacy_mode=FLAGS.legacy_mode,
                            pyr_lvls = FLAGS.pyr_lvls,
                            flow_pred_lvl=FLAGS.flow_pred_lvl,
                            search_range=FLAGS.search_range,
                            use_res_cx=FLAGS.use_res_cx,
                            use_dense_cx=FLAGS.use_dense_cx,
  )
  if FLAGS.pretrained_ckpt_depthnet is not None:
    # vars_to_restore = util.get_vars_to_restore(FLAGS.pretrained_ckpt)
    # pretrain_restorer = tf.train.Saver(vars_to_restore)
    # vars_to_restore = slim.get_variables_to_restore(include=["depth_prediction/depth_net/encoder"])
    # renamaed_variables = {
    #     'model/' + var.op.name.split('depth_prediction/')[1]: var
    #     for var in vars_to_restore
    # }
    vars_to_restore_depth_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "depth_prediction/depth_net")
    # for var in vars_to_restore_depth_net:
    #     print(var.op.name)
    pretrain_restorer_depthnet = tf.train.Saver(vars_to_restore_depth_net)
  if FLAGS.pretrained_ckpt_pwcnet is not None:
    vars_to_restore_pwcnet = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "direct_flow_prediction")
    vars_to_restore_pwcnet = {
        var.op.name.split('direct_flow_prediction/')[1]: var
        for var in vars_to_restore_pwcnet
    }
    # print(vars_to_restore_pwcnet)
    pretrain_restorer_pwcnet = tf.train.Saver(vars_to_restore_pwcnet, max_to_keep=MAX_TO_KEEP)
  # vars_to_save = util.get_vars_to_restore()
  vars_to_save = [var for var in tf.model_variables()]
  saver = tf.train.Saver(vars_to_save + [train_model.global_step],
                         max_to_keep=MAX_TO_KEEP)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # used for tensorboard visualize predict flow
  summary_writer3 = OptFlowTBLogger(FLAGS.checkpoint_dir + '/' + 'train_flow_predict')
  summary_writer4 = OptFlowTBLogger(FLAGS.checkpoint_dir + '/' + 'train_direct_flow_predict')
  summary_writer5 = OptFlowTBLogger(FLAGS.checkpoint_dir + '/' + 'test_flow_predict')
  summary_writer6 = OptFlowTBLogger(FLAGS.checkpoint_dir + '/' + 'test_direct_flow_predict')
  with tf.Session(config=config) as sess:
    summary_op = tf.summary.merge_all()
    summary_writer1 = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/' + 'train', sess.graph)
    summary_writer2 = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/' + 'val')

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    if FLAGS.pretrained_ckpt_depthnet is not None:
      logging.info('Restoring pretrained weights from depth-net %s', FLAGS.pretrained_ckpt_depthnet)
      pretrain_restorer_depthnet.restore(sess, FLAGS.pretrained_ckpt_depthnet)
      logging.info('depth-net weight load successfully!!!')
    if FLAGS.pretrained_ckpt_pwcnet is not None:
      logging.info('Restoring pretrained weights from pwc-net %s', FLAGS.pretrained_ckpt_pwcnet)
      pretrain_restorer_pwcnet.restore(sess, FLAGS.pretrained_ckpt_pwcnet)
      logging.info('pwc-net weight load successfully!!!')
    logging.info('Attempting to resume training from %s...', FLAGS.checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)
      logging.info('Last checkpoint load successfully')

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()

    step = 1
    logging.info('train_steps:%d'%train_steps)
    logging.info('train_sampless:%d' % len(reader_data.file_lists['image_file_list']))
    logging.info('train_epoch: %d' % (train_steps / steps_per_epoch))
    val_loss = -1
    total_val_loss = 0
    while step <= train_steps:
      image_forward_pose_flow_show = []
      image_forward_direct_flow_show = []
      pose_flow_image_show = []
      direct_flow_show = []
      image_stack_train_data, intrinsic_mat_train_data, intrinsic_mat_inv_train_data = sess.run(
            [image_stack_train, intrinsic_mat_train, intrinsic_mat_inv_train])
      fetches_train = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
          'incr_global_step': train_model.incr_global_step
      }

      if step % FLAGS.summary_freq == 0:
        fetches_train['loss'] = train_model.total_loss
        fetches_train['summary'] = summary_op
        fetches_train['image_forward'] = train_model.image_fwd_pose_flow_TB
        fetches_train['flow_image'] = train_model.pose_flow_image
        fetches_train['direct_image_forward'] = train_model.image_fwd_direct_flow_TB
        fetches_train['direct_flow_image'] = train_model.direct_flow_image

      results = sess.run(fetches_train, feed_dict={train_model.image_stack: image_stack_train_data,
                                             train_model.intrinsic_mat: intrinsic_mat_train_data,
                                             train_model.intrinsic_mat_inv: intrinsic_mat_inv_train_data})
      global_step = results['global_step']

      if step % FLAGS.summary_freq == 0:

        for i in range(FLAGS.batch_size):
          image_forward_pose_flow_show.append(results["image_forward"][i, :, :, :])
          pose_flow_image_show.append(results["flow_image"][0][i, :, :, :])
          image_forward_direct_flow_show.append(results["direct_image_forward"][i, :, :, :])
          direct_flow_show.append(results["direct_flow_image"][i, :, :, :])
        summary_writer3.log_imgs_w_flows('train/{}_pose_flows', image_forward_pose_flow_show, None, 0, pose_flow_image_show,
                                         None, global_step)
        summary_writer4.log_imgs_w_flows('train/{}_direct_flows', image_forward_direct_flow_show, None, 0, direct_flow_show,
                                         None, global_step)

        summary_writer1.add_summary(results["summary"], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        time_sofar = (time.time() - start_time) / 3600
        training_time_left = (train_steps / step - 1.0) * time_sofar
        last_summary_time += this_cycle
        logging.info(
            'Epoch: [%2d] [%5d/%5d] global_step:[%7d] time: %4.2fs (%4.2fh total) time_left: %4.2fh loss: %.3f',
            train_epoch, train_step, steps_per_epoch, global_step, this_cycle,
            time_sofar, training_time_left, results['loss'])

      if step % FLAGS.summary_freq == 0:
          # to save checkpoints
          total_val_loss = 0
          fetches_val = {}
          val_num_per_epoch = int(100 / FLAGS.batch_size)
          for i in range(val_num_per_epoch):
              fetches_val['val_loss'] = train_model.total_loss
              if i == 0:
                  fetches_val["summary"] = summary_op

                  fetches_val['image_forward'] = train_model.image_fwd_pose_flow_TB
                  fetches_val['flow_image'] = train_model.pose_flow_image
                  fetches_val['direct_image_forward'] = train_model.image_fwd_direct_flow_TB
                  fetches_val['direct_flow_image'] = train_model.direct_flow_image

              image_stack_test_data, intrinsic_mat_test_data, intrinsic_mat_inv_test_data = sess.run(
                  [image_stack_test, intrinsic_mat_test, intrinsic_mat_inv_test])
              results = sess.run(fetches_val,
                                 feed_dict={train_model.image_stack: image_stack_test_data,
                                            train_model.intrinsic_mat: intrinsic_mat_test_data,
                                            train_model.intrinsic_mat_inv: intrinsic_mat_inv_test_data}
                                 )
              if i == 0:
                  image_forward_pose_flow_show = []
                  image_forward_direct_flow_show = []
                  pose_flow_image_show = []
                  direct_flow_show = []
                  for i in range(FLAGS.batch_size):
                      image_forward_pose_flow_show.append(results["image_forward"][i, :, :, :])
                      pose_flow_image_show.append(results["flow_image"][0][i, :, :, :])
                      image_forward_direct_flow_show.append(results["direct_image_forward"][i, :, :, :])
                      direct_flow_show.append(results["direct_flow_image"][i, :, :, :])
                  summary_writer5.log_imgs_w_flows('test/{}_pose_flows', image_forward_pose_flow_show, None, 0,
                                                   pose_flow_image_show,
                                                   None, global_step)
                  summary_writer6.log_imgs_w_flows('test/{}_direct_flows', image_forward_direct_flow_show, None, 0,
                                                   direct_flow_show,
                                                   None, global_step)
                  summary_writer2.add_summary(results["summary"], global_step)
              total_val_loss += results['val_loss'] / val_num_per_epoch

          logging.info("val_loss: %.5f" % (total_val_loss))

          # if global_step>65000:
          if val_loss == -1 or total_val_loss < 1.02 * val_loss:
              if val_loss == -1 or total_val_loss < val_loss:
                  val_loss = total_val_loss
              logging.info('[*] Saving checkpoint to %s...', FLAGS.checkpoint_dir)
              saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model-minloss'),
                         global_step=global_step)
              with open(FLAGS.checkpoint_dir + '/min_val_loss.txt', 'a') as f:
                  f.write("global step: %7d val_loss: %.5f \n" % (global_step, total_val_loss))

      if step % (steps_per_epoch*2) == 0:
        logging.info('[*] Saving checkpoint to %s...', FLAGS.checkpoint_dir)
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'),
                   global_step=global_step)


      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1


if __name__ == '__main__':
  app.run(main)
