import tensorflow as tf

ckpt = '/home/lli/tensorflow/PWC-Net/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
# ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
# ckpt_var_names = sorted(ckpt_var_names, key=lambda x: x.op.name)
for name in ckpt_var_names:
    print(name)
#
# ckpt = '/media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat_relu_sigmoid_upconv_sad_left_right/depth_model_kitti_2015/model-179856'
# ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
# # ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
# # ckpt_var_names = sorted(ckpt_var_names, key=lambda x: x.op.name)
# for name in ckpt_var_names:
#     print(name)

#
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import math
# import os
# import random
# import time
# from absl import app
# from absl import flags
# from absl import logging
# import model
# import reader
# import numpy as np
# import tensorflow as tf
# import util
# slim = tf.contrib.slim
# gfile = tf.gfile
#
# HOME_DIR = os.path.expanduser('~')
# DEFAULT_DATA_DIR = os.path.join(HOME_DIR, 'vid2depth/data/kitti_raw_eigen')
# DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')
#
# flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Preprocessed data.')
# flags.DEFINE_string('train_mode','depth_odom', 'depth_odom or depth')
# flags.DEFINE_float('learning_rate', 0.0002, 'Adam learning rate. 0.00005')
# flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
# flags.DEFINE_float('reconstr_weight', 0.15, 'Frame reconstruction loss weight.')
# flags.DEFINE_float('smooth_weight', 0.1, 'Smoothness loss weight.')
# flags.DEFINE_float('ssim_weight', 0.85, 'SSIM loss weight.')
# flags.DEFINE_float('icp_weight', 0.0, 'ICP loss weight.')
# flags.DEFINE_float('disp_reg_weight', 0.05, 'disp_reg_weight. 0.05')
# flags.DEFINE_float('lr_disp_consistency_weight', 0.4, 'lr_disp_consistency_weight 0.4')
# flags.DEFINE_float('egomotion_snap_weight', 0, 'lr_disp_consistency_weight 1.0')
# flags.DEFINE_bool('sad_loss', False, ' if using sad_loss_filter in L1 output')
# flags.DEFINE_bool('use_charbonnier_loss', True, ' if using or not')
# flags.DEFINE_bool('use_geometry_mask', True, ' if using or not')
# flags.DEFINE_bool('use_flow_consistency_mask', True, ' if using or not')
# flags.DEFINE_integer('batch_size', 8, 'The size of a sample batch')
# flags.DEFINE_integer('img_height', 128, 'Input frame height.')
# flags.DEFINE_integer('img_width', 416, 'Input frame width.')
# # Note: Training time grows linearly with sequence length.  Use 2 or 3.
# flags.DEFINE_integer('seq_length', 1, 'Number of frames in sequence.')
# flags.DEFINE_integer('max_egomotion_step', 1, 'max_egomotion_step.')
# flags.DEFINE_string('pretrained_ckpt', None, 'Path to checkpoint with '
#                     'pretrained weights.  Do not include .data* extension.')
# flags.DEFINE_string('checkpoint_dir', DEFAULT_CHECKPOINT_DIR,
#                     'Directory to save model checkpoints.')
# flags.DEFINE_integer('train_steps', 180000, 'Number of training steps. 120000,300000')
# flags.DEFINE_integer("epoch", 55, "Maximum epoch of training iterations")
# flags.DEFINE_integer('summary_freq', 400, 'Save summaries every N steps.')
# flags.DEFINE_bool('legacy_mode', False, 'Whether to limit losses to using only '
#                   'the middle frame in sequence as the target frame.')
# FLAGS = flags.FLAGS

# import argparse
# import os
#
# HOME_DIR = os.path.expanduser('~')
# DEFAULT_DATA_DIR = os.path.join(HOME_DIR, 'vid2depth/data/kitti_raw_eigen')
# DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')
#
# parser = argparse.ArgumentParser(description='Parameters')
#
# parser.add_argument('--data_dir',        type=str,   help='Preprocessed data', default= DEFAULT_DATA_DIR)
# parser.add_argument('--train_mode',      type=str,   help='depth_odom or depth', default='depth_odom')
# parser.add_argument('--pretrained_ckpt', type=str,   help='Path to checkpoint with '
#                     'pretrained weights.  Do not include .data* extension.', default=None)
# parser.add_argument('--checkpoint_dir',  type=str,   help='Directory to save model checkpoints', default=DEFAULT_CHECKPOINT_DIR)
#
# parser.add_argument('--sad_loss',              type=bool,   help='True or Flase', default=False)
# parser.add_argument('--use_charbonnier_loss',  type=bool,   help='True or Flase', default=True)
# parser.add_argument('--use_geometry_mask',     type=bool,   help='True or Flase', default=True)
# parser.add_argument('--use_flow_consistency_mask',     type=bool,   help='True or Flase', default=True)
# parser.add_argument('--legacy_mode',           type=bool,   help='Whether to limit losses to using only '
#                   'the middle frame in sequence as the target frame.', default=False)
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
#
#
# FLAGS = parser.parse_args()

# mode = 'a' #if FLAGS.resume else 'w'
# with open('/media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat_relu_sigmoid_upconv_sad_left_right/depth_model_kitti_2015_ssim085_015' + '训练参数.txt', mode) as f:
#     f.write('\n' + '=' * 50 + '\n')
#     f.write('\n'.join("%s: %s" % item for item in FLAGS.flag_values_dict().items()))
#     f.write('\n' + '=' * 50 + '\n')