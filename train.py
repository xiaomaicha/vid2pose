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
import argparse
import model
import reader
import numpy as np
import tensorflow as tf
import util
import argparse
slim = tf.contrib.slim
gfile = tf.gfile

HOME_DIR = os.path.expanduser('~')
DEFAULT_DATA_DIR = os.path.join(HOME_DIR, 'vid2depth/data/kitti_raw_eigen')
DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')


# parser = argparse.ArgumentParser(description='Parameters')
#
# parser.add_argument('--data_dir',        type=str,   help='Preprocessed data', default= DEFAULT_DATA_DIR)
# parser.add_argument('--train_mode',      type=str,   help='depth_odom or depth', default='depth_odom')
# parser.add_argument('--pretrained_ckpt', type=str,   help='Path to checkpoint with '
#                     'pretrained weights.  Do not include .data* extension.', default=None)
# parser.add_argument('--checkpoint_dir',  type=str,   help='Directory to save model checkpoints',
#                                                      default=DEFAULT_CHECKPOINT_DIR)
#
# parser.add_argument('--sad_loss',              type=bool,   help='True or Flase', default=False)
# parser.add_argument('--use_charbonnier_loss',  type=bool,   help='True or Flase', default=True)
# parser.add_argument('--use_geometry_mask',     type=bool,   help='True or Flase', default=True)
# parser.add_argument('--use_flow_consistency_mask',     type=bool,   help='True or Flase', default=True)
# parser.add_argument('--use_disp_weight',     type=bool,   help='True or Flase', default=False)
# parser.add_argument('--use_temporal_dynamic_mask',     type=bool,   help='True or Flase', default=False)
# parser.add_argument('--use_temporal_occlusion_mask',     type=bool,   help='True or Flase', default=False)
# parser.add_argument('--legacy_mode',           type=bool,   help='Whether to limit losses to using only '
#                   'the middle frame in sequence as the target frame.', default=False)
#
#
# parser.add_argument('--beta1',              type=float,   help='Adam momentum', default=0.9)
# parser.add_argument('--reconstr_weight',    type=float,   help='Frame reconstruction loss weight', default=0.15)
# parser.add_argument('--smooth_weight',      type=float,   help='Smoothness loss weight', default=0.1)
# parser.add_argument('--ssim_weight',        type=float,   help='SSIM loss weight', default=0.85)
# parser.add_argument('--icp_weight',         type=float,   help='ICP loss weight', default=0)
# parser.add_argument('--disp_reg_weight',      type=float,   help='disp_reg_weight. 0.05', default=0.05)
# parser.add_argument('--lr_disp_consistency_weight',     type=float,   help='lr_disp_consistency_weight 0.4', default=0.4)
# parser.add_argument('--egomotion_snap_weight',          type=float,   help='egomotion_snap_weight 0.5', default=0.5)
#
# parser.add_argument('--batch_size',        type=int,   help='batch size', default=8)
# parser.add_argument('--img_height',        type=int,   help='Input frame height', default=128)
# parser.add_argument('--img_width',         type=int,   help='Input frame width', default=416)
# parser.add_argument('--seq_length',        type=int,   help='Number of frames in sequence', default=1)
# parser.add_argument('--max_egomotion_step',type=int,   help='max_egomotion_step', default=1)
# parser.add_argument('--train_steps',       type=int,   help='Number of training steps. 120000,300000', default=180000)
# parser.add_argument('--epoch',             type=int,   help='Maximum epoch of training iterations', default=50)
# parser.add_argument('--summary_freq',      type=int,   help='Save summaries every N steps', default=400)
# FLAGS = parser.parse_args()



flags.DEFINE_string('data_dir',                  DEFAULT_DATA_DIR, 'Preprocessed data.')
flags.DEFINE_string('train_mode',                'depth_odom', 'depth_odom or depth')
flags.DEFINE_string('pretrained_ckpt',           None, 'Path to checkpoint with '
                    'pretrained weights.  Do not include .data* extension.')
flags.DEFINE_string('checkpoint_dir',            DEFAULT_CHECKPOINT_DIR,'Directory to save model checkpoints.')

flags.DEFINE_float('learning_rate',              0.0002, 'Adam learning rate. 0.0002,0.00005')
flags.DEFINE_float('beta1',                      0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight',            0.15, 'Frame reconstruction loss weight.')
flags.DEFINE_float('smooth_weight',              0.1, 'Smoothness loss weight.')
flags.DEFINE_float('ssim_weight',                0.85, 'SSIM loss weight.')
flags.DEFINE_float('icp_weight',                 0.0, 'ICP loss weight.')
flags.DEFINE_float('disp_reg_weight',            0.02, 'disp_reg_weight. 0.02')
flags.DEFINE_float('lr_disp_consistency_weight', 0.4, 'lr_disp_consistency_weight 0.4')
flags.DEFINE_float('egomotion_snap_weight',      0, 'lr_disp_consistency_weight 1.0')

flags.DEFINE_bool('sad_loss',                    False, ' if using sad_loss_filter in L1 output')
flags.DEFINE_bool('use_charbonnier_loss',        True, ' if using or not')
flags.DEFINE_bool('use_geometry_mask',           True, ' if using or not')
flags.DEFINE_bool('use_flow_consistency_mask',   True, ' if using or not')
flags.DEFINE_bool('use_disp_weight',             True, ' if using or not')
flags.DEFINE_bool('use_temporal_dynamic_mask',   False, ' if using or not')
flags.DEFINE_bool('use_temporal_occlusion_mask', False, ' if using or not')
flags.DEFINE_bool('legacy_mode',                 False, 'Whether to limit losses to using only '
                  'the middle frame in sequence as the target frame.')

flags.DEFINE_integer('batch_size',               8, 'The size of a sample batch')
flags.DEFINE_integer('img_height',               256, 'Input frame height.')
flags.DEFINE_integer('img_width',                512, 'Input frame width.')
flags.DEFINE_integer('seq_length',               1, 'Number of frames in sequence.')
flags.DEFINE_integer('max_egomotion_step',       1, 'max_egomotion_step.')
flags.DEFINE_integer('train_steps',              180000, 'Number of training steps. 180000,300000')
flags.DEFINE_integer("epoch",                    50, "Maximum epoch of training iterations")
flags.DEFINE_integer('summary_freq',             400, 'Save summaries every N steps.')

FLAGS = flags.FLAGS


# Maximum number of checkpoints to keep.
MAX_TO_KEEP = 25
NUM_SCALES = 4


def main(_):
  # Fixed seed for repeatability
  seed = 8964
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if FLAGS.legacy_mode and FLAGS.seq_length < 3:
    raise ValueError('Legacy mode supports sequence length > 2 only.')

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
                            use_disp_weight=FLAGS.use_disp_weight,
                            disp_reg_weight=FLAGS.disp_reg_weight,
                            lr_disp_consistency_weight=FLAGS.lr_disp_consistency_weight,
                            egomotion_snap_weight = FLAGS.egomotion_snap_weight,
                            icp_weight=FLAGS.icp_weight,
                            batch_size=FLAGS.batch_size,
                            img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            seq_length=FLAGS.seq_length,
                            max_egomotion_step=FLAGS.max_egomotion_step,
                            train_mode=FLAGS.train_mode,
                            sad_loss=FLAGS.sad_loss,
                            use_charbonnier_loss=FLAGS.use_charbonnier_loss,
                            use_geometry_mask=FLAGS.use_geometry_mask,
                            use_flow_consistency_mask=FLAGS.use_flow_consistency_mask,
                            use_temporal_dynamic_mask=FLAGS.use_temporal_dynamic_mask,
                            use_temporal_occlusion_mask=FLAGS.use_temporal_occlusion_mask,
                            legacy_mode=FLAGS.legacy_mode)

  if FLAGS.pretrained_ckpt is not None:
    # vars_to_restore = util.get_vars_to_restore(FLAGS.pretrained_ckpt)
    # pretrain_restorer = tf.train.Saver(vars_to_restore)
    # vars_to_restore = slim.get_variables_to_restore(include=["depth_prediction/depth_net/encoder"])
    # renamaed_variables = {
    #     'model/' + var.op.name.split('depth_prediction/')[1]: var
    #     for var in vars_to_restore
    # }
    vars_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "depth_prediction/depth_net")
    for var in vars_to_restore:
        print(var.op.name)
    pretrain_restorer = tf.train.Saver(vars_to_restore)
  # vars_to_save = util.get_vars_to_restore()
  vars_to_save = [var for var in tf.model_variables()]
  saver = tf.train.Saver(vars_to_save + [train_model.global_step],
                         max_to_keep=MAX_TO_KEEP)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    summary_op = tf.summary.merge_all()
    summary_writer1 = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/' + 'train', sess.graph)
    summary_writer2 = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/' + 'val')

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    if FLAGS.pretrained_ckpt is not None:
      logging.info('Restoring pretrained weights from %s', FLAGS.pretrained_ckpt)
      pretrain_restorer.restore(sess, FLAGS.pretrained_ckpt)
    logging.info('Attempting to resume training from %s...', FLAGS.checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()

    step = 1
    logging.info('train_steps:%d'%train_steps)
    logging.info('train_epoch: %d' % (train_steps / steps_per_epoch))
    logging.info('train samples: %d' % (len(reader_data.file_lists['image_file_list'])))
    logging.info('val samples: %d' % (len(reader_data.file_lists_test['image_file_list'])))
    val_loss = -1
    total_val_loss = 0
    while step <= train_steps:
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

      results = sess.run(fetches_train,feed_dict={train_model.image_stack:image_stack_train_data,
                                          train_model.intrinsic_mat:intrinsic_mat_train_data,
                                          train_model.intrinsic_mat_inv:intrinsic_mat_inv_train_data})
      global_step = results['global_step']

      if step % FLAGS.summary_freq == 0:
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
          # image_stack_test_data, intrinsic_mat_test_data, intrinsic_mat_inv_test_data = sess.run(
          #     [image_stack_test, intrinsic_mat_test, intrinsic_mat_inv_test])
          # fetches = {
          #     "val_loss": train_model.total_loss,
          #     "summary": summary_op
          # }
          # results = sess.run(fetches,
          #                    feed_dict={train_model.image_stack:image_stack_test_data,
          #                                 train_model.intrinsic_mat:intrinsic_mat_test_data,
          #                                 train_model.intrinsic_mat_inv:intrinsic_mat_inv_test_data}
          #                    )
          # # sv.summary_writer.add_summary(results["summary"], gs)
          # summary_writer2.add_summary(results["summary"], global_step)

          #to save checkpoints
          total_val_loss = 0
          fetches_test = {}
          val_num_per_epoch = int(200/FLAGS.batch_size)
          for i in range(val_num_per_epoch):
              fetches_test['val_loss'] = train_model.total_loss
              if i == 0:
                  fetches_test["summary"] = summary_op
              image_stack_test_data, intrinsic_mat_test_data, intrinsic_mat_inv_test_data = sess.run(
                  [image_stack_test, intrinsic_mat_test, intrinsic_mat_inv_test])
              results = sess.run(fetches_test,
                                 feed_dict={train_model.image_stack: image_stack_test_data,
                                            train_model.intrinsic_mat: intrinsic_mat_test_data,
                                            train_model.intrinsic_mat_inv: intrinsic_mat_inv_test_data}
                                 )
              if i == 0:
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

          fetches_test = {}

      if step % (steps_per_epoch*2) == 0:
        logging.info('[*] Saving checkpoint to %s...', FLAGS.checkpoint_dir)
        with open(FLAGS.checkpoint_dir + '/val_loss.txt', 'a') as f:
            f.write("global step: %7d val_loss: %.5f \n" % (global_step, total_val_loss))

        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'),
                   global_step=global_step)


      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1


if __name__ == '__main__':
  app.run(main)
