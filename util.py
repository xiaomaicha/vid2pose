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

"""Contains common flags and functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale
import os
from absl import logging
import numpy as np
import tensorflow as tf


def get_seq_middle(seq_length):
  """Returns relative index for the middle frame in sequence."""
  half_offset = int((seq_length - 1) / 2)
  return seq_length - 1 - half_offset


def info(obj):
  """Return info on shape and dtype of a numpy array or TensorFlow tensor."""
  if obj is None:
    return 'None.'
  elif isinstance(obj, list):
    if obj:
      return 'List of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty list.'
  elif isinstance(obj, tuple):
    if obj:
      return 'Tuple of %d... %s' % (len(obj), info(obj[0]))
    else:
      return 'Empty tuple.'
  else:
    if is_a_numpy_array(obj):
      return 'Array with shape: %s, dtype: %s' % (obj.shape, obj.dtype)
    else:
      return str(obj)


def is_a_numpy_array(obj):
  """Returns true if obj is a numpy array."""
  return type(obj).__module__ == np.__name__


def count_parameters(also_print=True):
  """Cound the number of parameters in the model.

  Args:
    also_print: Boolean.  If True also print the numbers.

  Returns:
    The total number of parameters.
  """
  total = 0
  if also_print:
    logging.info('Model Parameters:')
  for v in get_vars_to_restore():
    shape = v.get_shape()
    if also_print:
      logging.info('%s %s: %s', v.op.name, shape,
                   format_number(shape.num_elements()))
    total += shape.num_elements()
  if also_print:
    logging.info('Total: %s', format_number(total))
  return total


def get_vars_to_restore(ckpt=None):
  """Returns list of variables that should be saved/restored.

  Args:
    ckpt: Path to existing checkpoint.  If present, returns only the subset of
        variables that exist in given checkpoint.

  Returns:
    List of all variables that need to be saved/restored.
  """
  model_vars = tf.trainable_variables()
  # Add batchnorm variables.
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name]
  model_vars.extend(bn_vars)
  model_vars = sorted(model_vars, key=lambda x: x.op.name)
  if ckpt is not None:
    ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
    ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
    for v in model_vars:
      if v.op.name not in ckpt_var_names:
        logging.warn('Missing var %s in checkpoint: %s', v.op.name,
                     os.path.basename(ckpt))
    model_vars = [v for v in model_vars if v.op.name in ckpt_var_names]
  return model_vars


def format_number(n):
  """Formats number with thousands commas."""
  locale.setlocale(locale.LC_ALL) #, 'en_US'
  return locale.format('%d', n, grouping=True)


def read_text_lines(filepath):
  with open(filepath, 'r') as f:
    lines = f.readlines()
  lines = [l.rstrip() for l in lines]
  return lines

def read_text_lines_stereo(filepath):
  with open(filepath, 'r') as f:
    lines = f.readlines()
  lines1 = [l.split(' ')[0].rstrip() for l in lines]
  lines2 = [l.split(' ')[1].rstrip() for l in lines]
  return lines1, lines2

def adapt_x(x, pyr_lvls):
  """Preprocess the input samples to adapt them to the network's requirements
  Here, x, is the actual data, not the x TF tensor.
  Args:
      x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
  Returns:
      Samples ready to be given to the network (w. same shape as x)
      Also, return adaptation info in (N,2,H,W,3) format
  """
  # Ensure we're dealing with RGB image pairs
  # assert (isinstance(x, np.ndarray) or isinstance(x, list))
  # if isinstance(x, np.ndarray):
  assert (len(x.get_shape().as_list()) == 5)
  assert (x.get_shape().as_list()[1] == 2 and x.get_shape().as_list()[4] == 3)
  # else:
  #   assert (len(x[0].shape) == 4)
  #   assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

  # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
  # if self.opts['use_mixed_precision'] is True:
  #   x_adapt = np.array(x, dtype=np.float16) if isinstance(x, list) else x.astype(np.float16)
  # else:
  #   x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
  # x_adapt /= 255.
  x_adapt = x
  # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
  _, pad_h = divmod(x.get_shape().as_list()[2], 2 ** pyr_lvls)
  if pad_h != 0:
    pad_h = 2 ** pyr_lvls - pad_h
  _, pad_w = divmod(x.get_shape().as_list()[3], 2 ** pyr_lvls)
  if pad_w != 0:
    pad_w = 2 ** pyr_lvls - pad_w
  x_adapt_info = None
  if pad_h != 0 or pad_w != 0:
    padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
    x_adapt_info = x.get_shape()  # Save original shape
    x_adapt = tf.pad(x, padding, mode='constant', constant_values=0.)

  return x_adapt, x_adapt_info


def postproc_pred_flow(y_hat, adapt_info=None):
  """Postprocess the results coming from the network during the test mode.
  Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
  Args:
      y_hat: predictions, see set_output_tnsrs() for details
      adapt_info: adaptation information in list[(N,H,W,2),...] format
  Returns:
      Postprocessed labels
  """
  # assert (isinstance(y_hat, list) and len(y_hat) == 2)

  # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
  pred_flows = []
  if adapt_info is not None:
    for i in range(len(adapt_info)):
      pred_flow = y_hat[i][:, 0:adapt_info[i][1], 0:adapt_info[i][2], :]
      pred_flows.append(pred_flow)

  # Individuate flows of the flow pyramid (at this point, they are still batched)
  # pyramids = y_hat[1]
  # pred_flows_pyramid = []
  # for idx in range(len(pred_flows)):
  #   pyramid = []
  #   for lvl in range(self.opts['pyr_lvls'] - self.opts['flow_pred_lvl'] + 1):
  #     pyramid.append(pyramids[lvl][idx])
  #   pred_flows_pyramid.append(pyramid)

  return pred_flows
  # pred_flows_pyramid
