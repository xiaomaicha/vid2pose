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

"""Depth and Ego-Motion networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util


# TODO(rezama): Move flag to main, pass as argument to functions.
flags.DEFINE_bool('use_bn', False, 'Add batch norm layers.')
FLAGS = flags.FLAGS

# Weight regularization.
WEIGHT_REG = 0.0001

# Disparity (inverse depth) values range from 0.01 to 10.
DISP_SCALING = 5
MIN_DISP = 0.002

EGOMOTION_VEC_SIZE = 6

SIMPLE = 'simple'
RESNET = 'resnet'
ARCHITECTURES = [SIMPLE, RESNET]

SCALE_TRANSLATION = 0.001
SCALE_ROTATION = 0.01

DISP_SCALING_RESNET50 = 5
DISP_SCALING_VGG = 5
FLOW_SCALING = 0.1


def egomotion_net(image_stack, is_training=True, legacy_mode=False):
  """Predict ego-motion vectors from a stack of frames.

  Args:
    image_stack: Input tensor with shape [B, h, w, seq_length * 3].  Regardless
        of the value of legacy_mode, the input image sequence passed to the
        function should be in normal order, e.g. [1, 2, 3].
    is_training: Whether the model is being trained or not.
    legacy_mode: Setting legacy_mode to True enables compatibility with
        SfMLearner checkpoints.  When legacy_mode is on, egomotion_net()
        rearranges the input tensor to place the target (middle) frame first in
        sequence.  This is the arrangement of inputs that legacy models have
        received during training.  In legacy mode, the client program
        (model.Model.build_loss()) interprets the outputs of this network
        differently as well.  For example:

        When legacy_mode == True,
        Network inputs will be [2, 1, 3]
        Network outputs will be [1 -> 2, 3 -> 2]

        When legacy_mode == False,
        Network inputs will be [1, 2, 3]
        Network outputs will be [1 -> 2, 2 -> 3]

  Returns:
    Egomotion vectors with shape [B, seq_length - 1, 6].
  """
  seq_length = image_stack.get_shape()[3].value // 3  # 3 == RGB.
  if legacy_mode:
    # Put the target frame at the beginning of stack.
    with tf.name_scope('rearrange_stack'):
      mid_index = util.get_seq_middle(seq_length)
      left_subset = image_stack[:, :, :, :mid_index * 3]
      target_frame = image_stack[:, :, :, mid_index * 3:(mid_index + 1) * 3]
      right_subset = image_stack[:, :, :, (mid_index + 1) * 3:]
      image_stack = tf.concat([target_frame, left_subset, right_subset], axis=3)
  batch_norm_params = {'is_training': is_training}
  num_egomotion_vecs = seq_length - 1

  h = image_stack.get_shape()[1].value
  w = image_stack.get_shape()[2].value

  #adopt and improve from lsvo
  with tf.variable_scope('pose_flow_net') as sc:
      with tf.variable_scope('encodee'):
          end_points_collection = sc.original_name_scope + '_end_points'
          normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
          normalizer_params = batch_norm_params if FLAGS.use_bn else None
          with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                              normalizer_fn=normalizer_fn,
                              weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                              normalizer_params=normalizer_params,
                              activation_fn=tf.nn.relu,
                              outputs_collections=end_points_collection):
              cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
              cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
              cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
              cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
              cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')

              # undeepvo
              cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
              cnv7 = slim.conv2d(cnv6, 512, [3, 3], stride=2, scope='cnv7')
      with tf.variable_scope('pose_net'):
          flatten = slim.flatten(cnv7, outputs_collections=end_points_collection, scope='flatten')

          fc1 = slim.fully_connected(flatten, 512, normalizer_fn=None, scope='fc1')
          fc2 = slim.fully_connected(fc1, 512, normalizer_fn=None, scope='fc2')
          egomotion_tran = slim.fully_connected(fc2, 3, activation_fn=None, normalizer_fn=None, scope='fc3') * 0.1

          fc4 = slim.fully_connected(flatten, 512, normalizer_fn=None, scope='fc4')
          fc5 = slim.fully_connected(fc4, 512, normalizer_fn=None, scope='fc5')
          egomotion_rot = slim.fully_connected(fc5, 3, activation_fn=None, normalizer_fn=None, scope='fc6') * 0.1

          egomotion_avg = tf.concat([egomotion_tran, egomotion_rot], axis=1)
          egomotion_final = tf.reshape(egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])

      with tf.variable_scope('flow_net'):
          # #参照sfmlearner的mask网路
          # upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')
          #
          # upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
          # flow4 = get_flow(upcnv4, scope='flow4')
          #
          # upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
          # flow3 = get_flow(upcnv3, scope='flow3')
          #
          # upcnv2 = slim.conv2d_transpose(upcnv3, 32, [5, 5], stride=2, scope='upcnv2')
          # flow2 = get_flow(upcnv2, scope='flow2')
          #
          # upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
          # flow1 = get_flow(upcnv1, scope='flow1')

          #参照跳连接上采样网络
          up7 = slim.conv2d_transpose(cnv7, 256, [3, 3], stride=2, scope='upcnv7')
          # There might be dimension mismatch due to uneven down/up-sampling.
          up7 = _resize_like(up7, cnv6)
          i7_in = tf.concat([up7, cnv6], axis=3)
          icnv7 = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')

          up6 = slim.conv2d_transpose(icnv7, 256, [3, 3], stride=2, scope='upcnv6')
          up6 = _resize_like(up6, cnv5)
          i6_in = tf.concat([up6,cnv5], axis=3)
          icnv6 = slim.conv2d(i6_in, 256, [3, 3], stride=1, scope='icnv6')

          up5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
          up5 = _resize_like(up5, cnv4)
          i5_in = tf.concat([up5, cnv4], axis=3)
          icnv5 = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')

          up4 = slim.conv2d_transpose(icnv5, 64, [3, 3], stride=2, scope='upcnv4')
          up4 = _resize_like(up4, cnv3)
          i4_in = tf.concat([up4, cnv3], axis=3)
          icnv4 = slim.conv2d(i4_in, 64, [3, 3], stride=1, scope='icnv4')
          flow4 = get_flow(icnv4, scope='flow4')
          flow4_up = tf.image.resize_bilinear(flow4, [np.int(h / 4), np.int(w / 4)])

          up3 = slim.conv2d_transpose(icnv4, 32, [3, 3], stride=2, scope='upcnv3')
          i3_in = tf.concat([up3, cnv2, flow4_up], axis=3)
          icnv3 = slim.conv2d(i3_in, 32, [3, 3], stride=1, scope='icnv3')
          flow3 = get_flow(icnv3, scope='flow3')
          flow3_up = tf.image.resize_bilinear(flow3, [np.int(h / 2), np.int(w / 2)])

          up2 = slim.conv2d_transpose(icnv3, 16, [3, 3], stride=2, scope='upcnv2')
          i2_in = tf.concat([up2, cnv1, flow3_up], axis=3)
          icnv2 = slim.conv2d(i2_in, 16, [3, 3], stride=1, scope='icnv2')
          flow2 = get_flow(icnv2, scope='flow2')
          flow2_up = tf.image.resize_bilinear(flow2, [h, w])

          up1 = slim.conv2d_transpose(icnv2, 8, [3, 3], stride=2, scope='upcnv1')
          i1_in = tf.concat([up1, flow2_up], axis=3)
          icnv1 = slim.conv2d(i1_in, 8, [3, 3], stride=1, scope='icnv1')
          flow1 = get_flow(icnv1, scope='flow1')

  # with tf.variable_scope('pose_exp_net') as sc:
  #   end_points_collection = sc.original_name_scope + '_end_points'
  #   normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
  #   normalizer_params = batch_norm_params if FLAGS.use_bn else None
  #   with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
  #                       normalizer_fn=normalizer_fn,
  #                       weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
  #                       normalizer_params=normalizer_params,
  #                       activation_fn=tf.nn.relu,
  #                       outputs_collections=end_points_collection):
  #     cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
  #     cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
  #     cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
  #     cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
  #     cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
  #
  #     #undeepvo
  #     cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
  #     cnv7 = slim.conv2d(cnv6, 512, [3, 3], stride=2, scope='cnv7')
  #     flatten = slim.flatten(cnv7, outputs_collections=end_points_collection, scope='flatten')
  #     with tf.variable_scope('pose_tran'):
  #         fc1 = slim.fully_connected(flatten, 512, normalizer_fn=None, scope='fc1')
  #         fc2 = slim.fully_connected(fc1, 512, normalizer_fn=None, scope='fc2')
  #         egomotion_tran = slim.fully_connected(fc2, 3, activation_fn=None, normalizer_fn=None, scope='fc3') * 0.1
  #     with tf.variable_scope('pose_rot'):
  #         fc4 = slim.fully_connected(flatten, 512, normalizer_fn=None, scope='fc4')
  #         fc5 = slim.fully_connected(fc4, 512, normalizer_fn=None, scope='fc5')
  #         egomotion_rot = slim.fully_connected(fc5, 3, activation_fn=None, normalizer_fn=None, scope='fc6') * 0.1
  #     egomotion_avg = tf.concat([egomotion_tran, egomotion_rot], axis=1)
  #     egomotion_final = tf.reshape(egomotion_avg,  [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
  #
  #     #vid2depth
  #     '''
  #     # Ego-motion specific layers
  #     with tf.variable_scope('pose'):
  #       cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
  #       cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
  #       pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
  #       egomotion_pred = slim.conv2d(cnv7,
  #                                    pred_channels,
  #                                    [1, 1],
  #                                    scope='pred',
  #                                    stride=1,
  #                                    normalizer_fn=None,
  #                                    activation_fn=None)
  #       # egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
  #       # # Tinghui found that scaling by a small constant facilitates training.
  #       # egomotion_final = 0.01 * tf.reshape(
  #       #     egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
  #
  #       egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
  #       egomotion_res = tf.reshape(
  #           egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
  #       # Tinghui found that scaling by a small constant facilitates training.
  #       egomotion_final = tf.concat([egomotion_res[:, 0:3] * SCALE_TRANSLATION,
  #                                     egomotion_res[:, 3:6] * SCALE_ROTATION],
  #                                    axis=1)
  #     '''

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return egomotion_final, [flow1, flow2, flow3, flow4], end_points