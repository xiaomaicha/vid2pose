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

"""optical flow networks."""

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