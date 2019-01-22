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

    # def get_disp(self, x, scope = None):
    #     disp = slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.relu, normalizer_fn=None, scope = scope) + 0.00001
    #     return disp

    def get_disp(self, x, scope = None):
        disp = 1.2 * slim.conv2d(x, 2, 3, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None, scope = scope) + 0.00001
        return disp

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
        # p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(self.pad(x, 1), num_out_layers, kernel_size, scale, 'SAME', activation_fn=tf.nn.relu)
        return conv[:, 3:-1, 3:-1, :]

    def build_vgg_128(self, input, get_pred,  *args, **kwargs):
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

                    # conv1_a = conv(input, 32, 7, 1)
                    # conv1 = conv(conv1_a, 32, 7, 2)
                    # conv2_a = conv(conv1, 64, 5, 1)
                    # conv2 = conv(conv2_a, 64, 5, 2)
                    # conv3_a = conv(conv2, 128, 3, 1)
                    # conv3 = conv(conv3_a, 128, 3, 2)
                    # conv4_a = conv(conv3, 256, 3, 1)
                    # conv4 = conv(conv4_a, 256, 3, 2)
                    # conv5_a = conv(conv4, 512, 3, 1)
                    # conv5 = conv(conv5_a, 512, 3, 2)
                    # conv6_a = conv(conv5, 512, 3, 1)
                    # conv6 = conv(conv6_a, 512, 3, 2)
                    # conv7_a = conv(conv6, 512, 3, 1)
                    # conv7 = conv(conv7_a, 512, 3, 2)

                # with tf.variable_scope('skips'):
                #     skip1 = conv1
                #     skip2 = conv2
                #     skip3 = conv3
                #     skip4 = conv4
                #     skip5 = conv5
                #     skip6 = conv6

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