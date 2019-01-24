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


def disp_net(target_image, is_training=True):
  """Predict inverse of depth from a single image."""
  batch_norm_params = {'is_training': is_training}
  h = target_image.get_shape()[1].value
  w = target_image.get_shape()[2].value
  inputs = target_image
  with tf.variable_scope('depth_net') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
    normalizer_params = batch_norm_params if FLAGS.use_bn else None
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                        activation_fn=tf.nn.elu,
                        outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inputs, 32, [7, 7], stride=1, scope='cnv1')
      cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=2, scope='cnv1b')
      cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=1, scope='cnv2')
      cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=2, scope='cnv2b')

      cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=1, scope='cnv3')
      cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv3b')
      cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=1, scope='cnv4')
      cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv4b')
      cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=1, scope='cnv5')
      cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=2, scope='cnv5b')
      cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=1, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=2, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=1, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=2, scope='cnv7b')

      up7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
      # There might be dimension mismatch due to uneven down/up-sampling.
      up7 = _resize_like(up7, cnv6b)
      i7_in = tf.concat([up7, cnv6b], axis=3)
      icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

      up6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
      up6 = _resize_like(up6, cnv5b)
      i6_in = tf.concat([up6, cnv5b], axis=3)
      icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

      up5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
      up5 = _resize_like(up5, cnv4b)
      i5_in = tf.concat([up5, cnv4b], axis=3)
      icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

      up4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
      i4_in = tf.concat([up4, cnv3b], axis=3)
      icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
      disp4 = (slim.conv2d(icnv4, 1, [3, 3], stride=1, activation_fn=tf.nn.sigmoid,
                           normalizer_fn=None, scope='disp4')
               * DISP_SCALING + MIN_DISP)
      disp4_up = tf.image.resize_bilinear(disp4, [np.int(h / 4), np.int(w / 4)])

      up3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
      i3_in = tf.concat([up3, cnv2b, disp4_up], axis=3)
      icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
      disp3 = (slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.nn.sigmoid,
                           normalizer_fn=None, scope='disp3')
               * DISP_SCALING + MIN_DISP)
      disp3_up = tf.image.resize_bilinear(disp3, [np.int(h / 2), np.int(w / 2)])

      up2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
      i2_in = tf.concat([up2, cnv1b, disp3_up], axis=3)
      icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
      disp2 = (slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.nn.sigmoid,
                           normalizer_fn=None, scope='disp2')
               * DISP_SCALING + MIN_DISP)
      disp2_up = tf.image.resize_bilinear(disp2, [h, w])

      up1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
      i1_in = tf.concat([up1, disp2_up], axis=3)
      icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
      disp1 = (slim.conv2d(icnv1, 1, [3, 3], stride=1, activation_fn=tf.nn.sigmoid,
                           normalizer_fn=None, scope='disp1')
               * DISP_SCALING + MIN_DISP)

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return [disp1, disp2, disp3, disp4], end_points


def _resize_like(inputs, ref):
  i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
  r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
  if i_h == r_h and i_w == r_w:
    return inputs
  else:
    return tf.image.resize_nearest_neighbor(inputs, [r_h.value, r_w.value])


###geonet
def geo_disp_net(dispnet_inputs):
    is_training = True
    # return build_resnet50_2(dispnet_inputs, get_disp_vgg, is_training, 'depth_net')
    return build_vgg(dispnet_inputs, get_disp_vgg, is_training, 'depth_net')

def geo_flow_net(flownet_inputs):
    is_training = True
    # return build_resnet50(flownet_inputs, get_flow, is_training, 'flow_net')
    return build_vgg(flownet_inputs, get_flow, is_training, 'flow_net')


def build_vgg(inputs, get_pred, is_training, var_scope):
    batch_norm_params = {'is_training': is_training}
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    with tf.variable_scope(var_scope) as sc:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,#slim.batch_norm,
                            normalizer_params=None,#batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                            activation_fn=tf.nn.relu):
            # ENCODING
            with tf.variable_scope('encoder'):
                # conv1 = slim.conv2d(inputs, 32, 7, 2)
                # conv1b = slim.conv2d(conv1, 32, 7, 1)
                # conv2 = slim.conv2d(conv1b, 64, 5, 2)
                # conv2b = slim.conv2d(conv2, 64, 5, 1)
                # conv3 = slim.conv2d(conv2b, 128, 3, 2)
                # conv3b = slim.conv2d(conv3, 128, 3, 1)
                # conv4 = slim.conv2d(conv3b, 256, 3, 2)
                # conv4b = slim.conv2d(conv4, 256, 3, 1)
                # conv5 = slim.conv2d(conv4b, 512, 3, 2)
                # conv5b = slim.conv2d(conv5, 512, 3, 1)
                # conv6 = slim.conv2d(conv5b, 512, 3, 2)
                # conv6b = slim.conv2d(conv6, 512, 3, 1)
                # conv7 = slim.conv2d(conv6b, 512, 3, 2)
                # conv7b = slim.conv2d(conv7, 512, 3, 1)

                conv1 = conv(inputs, 32, 7, 2)
                conv1b = conv(conv1, 32, 7, 1)
                conv2 = conv(conv1b, 64, 5, 2)
                conv2b = conv(conv2, 64, 5, 1)
                conv3 = conv(conv2b, 128, 3, 2)
                conv3b = conv(conv3, 128, 3, 1)
                conv4 = conv(conv3b, 256, 3, 2)
                conv4b = conv(conv4, 256, 3, 1)
                conv5 = conv(conv4b, 512, 3, 2)
                conv5b = conv(conv5, 512, 3, 1)
                conv6 = conv(conv5b, 512, 3, 2)
                conv6b = conv(conv6, 512, 3, 1)
                conv7 = conv(conv6b, 512, 3, 2)
                conv7b = conv(conv7, 512, 3, 1)

                # conv1 = conv(inputs, 32, 7, 1)
                # conv1b = conv(conv1, 32, 7, 2)
                # conv2 = conv(conv1b, 64, 5, 1)
                # conv2b = conv(conv2, 64, 5, 2)
                # conv3 = conv(conv2b, 128, 3, 1)
                # conv3b = conv(conv3, 128, 3, 2)
                # conv4 = conv(conv3b, 256, 3, 1)
                # conv4b = conv(conv4, 256, 3, 2)
                # conv5 = conv(conv4b, 512, 3, 1)
                # conv5b = conv(conv5, 512, 3, 2)
                # conv6 = conv(conv5b, 512, 3, 1)
                # conv6b = conv(conv6, 512, 3, 2)
                # conv7 = conv(conv6b, 512, 3, 1)
                # conv7b = conv(conv7, 512, 3, 2)

            # DECODING
            with tf.variable_scope('decoder'):
                upconv7 = upconv(conv7b, 512, 3, 2)
                # There might be dimension mismatch due to uneven down/up-sampling
                upconv7 = resize_like(upconv7, conv6b)
                i7_in = tf.concat([upconv7, conv6b], axis=3)
                iconv7 = conv(i7_in, 512, 3, 1)

                upconv6 = upconv(iconv7, 512, 3, 2)
                upconv6 = resize_like(upconv6, conv5b)
                i6_in = tf.concat([upconv6, conv5b], axis=3)
                iconv6 = conv(i6_in, 512, 3, 1)

                upconv5 = upconv(iconv6, 256, 3, 2)
                upconv5 = resize_like(upconv5, conv4b)
                i5_in = tf.concat([upconv5, conv4b], axis=3)
                iconv5 = conv(i5_in, 256, 3, 1)

                upconv4 = upconv(iconv5, 128, 3, 2)
                i4_in = tf.concat([upconv4, conv3b], axis=3)
                iconv4 = conv(i4_in, 128, 3, 1)
                pred4 = get_pred(iconv4)
                pred4_up = tf.image.resize_nearest_neighbor(pred4, [np.int(H / 4), np.int(W / 4)])

                upconv3 = upconv(iconv4, 64, 3, 2)
                i3_in = tf.concat([upconv3, conv2b, pred4_up], axis=3)
                iconv3 = conv(i3_in, 64, 3, 1)
                pred3 = get_pred(iconv3)
                pred3_up = tf.image.resize_nearest_neighbor(pred3, [np.int(H / 2), np.int(W / 2)])

                upconv2 = upconv(iconv3, 32, 3, 2)
                i2_in = tf.concat([upconv2, conv1b, pred3_up], axis=3)
                iconv2 = conv(i2_in, 32, 3, 1)
                pred2 = get_pred(iconv2)
                pred2_up = tf.image.resize_nearest_neighbor(pred2, [H, W])

                upconv1 = upconv(iconv2, 16, 3, 2)
                i1_in = tf.concat([upconv1, pred2_up], axis=3)
                iconv1 = conv(i1_in, 16, 3, 1)
                pred1 = get_pred(iconv1)

                # upconv7 = upconv(conv7b, 512, 3, 2)
                # # There might be dimension mismatch due to uneven down/up-sampling
                # upconv7 = resize_like(upconv7, conv6b)
                # i7_in = tf.concat([upconv7, conv6b], axis=3)
                # iconv7 = slim.conv2d(i7_in, 512, 3, 1)
                #
                # upconv6 = upconv(iconv7, 512, 3, 2)
                # upconv6 = resize_like(upconv6, conv5b)
                # i6_in = tf.concat([upconv6, conv5b], axis=3)
                # iconv6 = slim.conv2d(i6_in, 512, 3, 1)
                #
                # upconv5 = upconv(iconv6, 256, 3, 2)
                # upconv5 = resize_like(upconv5, conv4b)
                # i5_in = tf.concat([upconv5, conv4b], axis=3)
                # iconv5 = slim.conv2d(i5_in, 256, 3, 1)
                #
                # upconv4 = upconv(iconv5, 128, 3, 2)
                # i4_in = tf.concat([upconv4, conv3b], axis=3)
                # iconv4 = slim.conv2d(i4_in, 128, 3, 1)
                # pred4 = get_pred(iconv4)
                # pred4_up = tf.image.resize_nearest_neighbor(pred4, [np.int(H / 4), np.int(W / 4)])
                #
                # upconv3 = upconv(iconv4, 64, 3, 2)
                # i3_in = tf.concat([upconv3, conv2b, pred4_up], axis=3)
                # iconv3 = slim.conv2d(i3_in, 64, 3, 1)
                # pred3 = get_pred(iconv3)
                # pred3_up = tf.image.resize_nearest_neighbor(pred3, [np.int(H / 2), np.int(W / 2)])
                #
                # upconv2 = upconv(iconv3, 32, 3, 2)
                # i2_in = tf.concat([upconv2, conv1b, pred3_up], axis=3)
                # iconv2 = slim.conv2d(i2_in, 32, 3, 1)
                # pred2 = get_pred(iconv2)
                # pred2_up = tf.image.resize_nearest_neighbor(pred2, [H, W])
                #
                # upconv1 = upconv(iconv2, 16, 3, 2)
                # i1_in = tf.concat([upconv1, pred2_up], axis=3)
                # iconv1 = slim.conv2d(i1_in, 16, 3, 1)
                # pred1 = get_pred(iconv1)

                # disp_est = [pred1, pred2, pred3, pred4]
                # disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in disp_est]
                # disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in disp_est]

            return [pred1, pred2, pred3, pred4] #




def conv(x, num_out_layers, kernel_size, stride):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    # p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(pad(x, p), num_out_layers, kernel_size, stride, 'VALID')


def get_disp_vgg(x):
    disp = 0.3 * slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) + 0.002
    return disp


def get_flow(x, scope = None):
    # Output flow value is normalized by image height/width
    flow = FLOW_SCALING * slim.conv2d(x, 2, 3, 1, activation_fn=None, normalizer_fn=None, scope=scope)
    return flow


def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def upsample_nn(x, ratio):
    h = x.get_shape()[1].value
    w = x.get_shape()[2].value
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])


def upconv(x, num_out_layers, kernel_size, scale):
    # upsample = upsample_nn(x, scale)
    cnv = conv(upsample_nn(x, scale), num_out_layers, kernel_size, 1)
    return cnv


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])

# def resconv(x, num_layers, stride):
#     # Actually here exists a bug: tf.shape(x)[3] != num_layers is always true,
#     # but we preserve it here for consistency with Godard's implementation.
#     do_proj = tf.shape(x)[3] != num_layers or stride == 2
#     shortcut = []
#     conv1 = conv(x,         num_layers, 1, 1)
#     conv2 = conv(conv1,     num_layers, 3, stride)
#     conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
#     if do_proj:
#         shortcut = conv(x, 4 * num_layers, 1, stride, None)
#     else:
#         shortcut = x
#     return tf.nn.relu(conv3 + shortcut)
#
# def resblock(x, num_layers, num_blocks):
#     out = x
#     for i in range(num_blocks - 1):
#         out = resconv(out, num_layers, 1)
#     out = resconv(out, num_layers, 2)
#     return out



##monodepth disp net

class disp_net_monodepth(object):
    def __init__(self):
        pass

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def resize_like(self, inputs, ref):
        iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
        rH, rW = ref.get_shape()[1], ref.get_shape()[2]
        if iH == rH and iW == rW:
            return inputs
        return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

    def get_disp(self, x, scope = None):
        disp = slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None, scope = scope) + 0.00001
        return disp

    # def get_disp(self, x, scope = None):
    #     disp = 1.2 * slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope = scope) + 0.00001
    #     return disp

    # def get_depth(self, x, scope = None):
    #     depth = 100.0 * slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope = scope)
    #     # depth = Lambda(lambda x: 100.0 * x)(depth)
    #     return depth

    def pad(self, tensor, num=1):
        """
        Pads the given tensor along the height and width dimensions with `num` 0s on each side
        """
        return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")

    def antipad(self, tensor, num=1):
        """
        Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
        padding from the output rather than adding it to the input.
        """
        batch, h, w, c = tensor.shape.as_list()
        return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])

    def conv(self, x, num_out_layers, kernel_size, stride):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        # p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(self.pad(x,p), num_out_layers, kernel_size, stride, 'VALID')

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 2)
        return self.conv(conv1, num_out_layers, kernel_size, 1)

    # def maxpool(self, x, kernel_size):
    #     p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    #     p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    #     return slim.max_pool2d(p_x, kernel_size)
    #
    # def resconv(self, x, num_layers, stride):
    #     do_proj = tf.shape(x)[3] != num_layers or stride == 2
    #     shortcut = []
    #     conv1 = self.conv(x, num_layers, 1, 1)
    #     conv2 = self.conv(conv1, num_layers, 3, stride)
    #     conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
    #     if do_proj:
    #         shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
    #     else:
    #         shortcut = x
    #     return tf.nn.elu(conv3 + shortcut)
    #
    # def resblock(self, x, num_layers, num_blocks):
    #     out = x
    #     for i in range(num_blocks - 1):
    #         out = self.resconv(out, num_layers, 1)
    #     out = self.resconv(out, num_layers, 2)
    #     return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        # upsample = self.upsample_nn(x, scale)
        conv = self.conv(self.upsample_nn(x, scale), num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        # p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])  self.pad(x, 1)
        conv = slim.conv2d_transpose(x, num_out_layers, kernel_size, scale, 'SAME', activation_fn=tf.nn.relu)
        return conv #conv[:, 3:-1, 3:-1, :]

    def original_disp_net(self, input):
        conv = self.conv
        upconv = self.deconv
        batch_norm_params = {'is_training': True}

        with tf.variable_scope('depth_net'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None,  # slim.batch_norm,
                                normalizer_params=None,  # batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                                activation_fn=tf.nn.relu):
                with tf.variable_scope('encoder'):
                    conv1_a = conv(input, 64,   7, 2)
                    conv1 = conv(conv1_a, 64,   7, 1)
                    conv2_a = conv(conv1, 128,  5, 2)
                    conv2 = conv(conv2_a, 128,  5, 1)
                    conv3_a = conv(conv2, 256,  3, 2)
                    conv3 = conv(conv3_a, 256,  3, 1)
                    conv4_a = conv(conv3, 512,  3, 2)
                    conv4 = conv(conv4_a, 512,  3, 1)
                    conv5_a = conv(conv4, 512,  3, 2)
                    conv5 = conv(conv5_a, 512,  3, 1)
                    conv6_a = conv(conv5, 1024, 3, 2)
                    conv6 = conv(conv6_a, 1024, 3, 1)


                with tf.variable_scope('decoder'):

                    upconv6 = upconv(conv6, 512, 3, 2)  # H/32
                    upconv6 = self.resize_like(upconv6, conv5)
                    concat6 = tf.concat([upconv6, conv5], 3)
                    iconv6 = conv(concat6, 512, 3, 1)

                    upconv5 = upconv(iconv6, 512, 3, 2)  # H/16
                    upconv5 = self.resize_like(upconv5, conv4)
                    concat5 = tf.concat([upconv5, conv4], 3)
                    iconv5 = conv(concat5, 512, 3, 1)

                    upconv4 = upconv(iconv5, 256, 3, 2)  # H/8
                    concat4 = tf.concat([upconv4, conv3], 3)
                    iconv4 = conv(concat4, 256, 3, 1)
                    disp4 = self.get_disp(iconv4)
                    udisp4 = self.upsample_nn(disp4, 2)

                    upconv3 = upconv(iconv4, 128, 3, 2)  # H/4
                    concat3 = tf.concat([upconv3, conv2, udisp4], 3)
                    iconv3 = conv(concat3, 128, 3, 1)
                    disp3 = self.get_disp(iconv3)
                    udisp3 = self.upsample_nn(disp3, 2)

                    upconv2 = upconv(iconv3, 64, 3, 2)  # H/2
                    concat2 = tf.concat([upconv2, conv1, udisp3], 3)
                    iconv2 = conv(concat2, 64, 3, 1)
                    disp2 = self.get_disp(iconv2)
                    udisp2 = self.upsample_nn(disp2, 2)

                    upconv1 = upconv(iconv2, 32, 3, 2)  # H
                    concat1 = tf.concat([upconv1, udisp2], 3)
                    iconv1 = conv(concat1, 32, 3, 1)
                    disp1 = self.get_disp(iconv1)

        self.disp_est = [disp1, disp2, disp3, disp4]
        self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
        self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        return self.disp_left_est, self.disp_right_est


    def build_vgg(self, input, get_pred,  *args, **kwargs):
        # set convenience functions
        conv = self.conv
        upconv = self.deconv
        batch_norm_params = {'is_training': True}

        with tf.variable_scope('depth_net'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                normalizer_fn=None, #slim.batch_norm,
                                normalizer_params=None,#batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                                activation_fn=tf.nn.relu):
                with tf.variable_scope('encoder'):
                    conv1 = self.conv_block(input, 32, 7)  # H/2
                    conv2 = self.conv_block(conv1, 64, 5)  # H/4
                    conv3 = self.conv_block(conv2, 128, 3)  # H/8
                    conv4 = self.conv_block(conv3, 256, 3)  # H/16
                    conv5 = self.conv_block(conv4, 512, 3)  # H/32
                    conv6 = self.conv_block(conv5, 512, 3)  # H/64
                    conv7 = self.conv_block(conv6, 512, 3)  # H/128


                with tf.variable_scope('decoder'):
                    upconv7 = upconv(conv7, 512, 3, 2)  # H/64
                    upconv7 = self.resize_like(upconv7, conv6)
                    concat7 = tf.concat([upconv7, conv6], 3)
                    iconv7 = conv(concat7, 512, 3, 1)

                    upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
                    upconv6 = self.resize_like(upconv6, conv5)
                    concat6 = tf.concat([upconv6, conv5], 3)
                    iconv6 = conv(concat6, 512, 3, 1)

                    upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
                    upconv5 = self.resize_like(upconv5, conv4)
                    concat5 = tf.concat([upconv5, conv4], 3)
                    iconv5 = conv(concat5, 256, 3, 1)

                    upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
                    concat4 = tf.concat([upconv4, conv3], 3)
                    iconv4 = conv(concat4, 128, 3, 1)
                    disp4 = self.get_disp(iconv4)
                    udisp4 = self.upsample_nn(disp4, 2)

                    upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
                    concat3 = tf.concat([upconv3, conv2, udisp4], 3)
                    iconv3 = conv(concat3, 64, 3, 1)
                    disp3 = self.get_disp(iconv3)
                    udisp3 = self.upsample_nn(disp3, 2)

                    upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
                    concat2 = tf.concat([upconv2, conv1, udisp3], 3)
                    iconv2 = conv(concat2, 32, 3, 1)
                    disp2 = self.get_disp(iconv2)
                    udisp2 = self.upsample_nn(disp2, 2)

                    upconv1 = upconv(iconv2, 16, 3, 2)  # H
                    concat1 = tf.concat([upconv1, udisp2], 3)
                    iconv1 = conv(concat1, 16, 3, 1)
                    disp1 = self.get_disp(iconv1)


        self.disp_est = [disp1, disp2, disp3, disp4]
        self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
        self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        return self.disp_left_est, self.disp_right_est