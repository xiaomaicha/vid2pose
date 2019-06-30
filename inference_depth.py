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

"""Generates depth estimates for an entire KITTI video."""

# Example usage:
#
# python inference.py \
# --logtostderr
# --kitti_dir /media/wuqi/works/dataset/kitti_raw
# --kitti_video 2011_09_26/2011_09_26_drive_0009_sync
# --output_dir /media/wuqi/ubuntu/code/slam/vid2pose_log/inference_monodepth
# --model_ckpt /media/wuqi/ubuntu/code/slam/monodepth/pretrained_model/model_kitti
#
# python inference.py
# --logtostderr
# --kitti_dir /media/wuqi/ubuntu/dataset/stereo/KITTI2015/data_scene_flow
# --kitti_video test_files_stereo
# --output_dir /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat/depth
# --model_ckpt /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat/model-31108
# --img_height 128
# --img_width 416
# --seq_length 2
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging
import matplotlib.pyplot as plt
import model
import numpy as np
import scipy.misc
import tensorflow as tf
import util
slim = tf.contrib.slim
gfile = tf.gfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HOME_DIR = os.path.expanduser('~')
DEFAULT_OUTPUT_DIR = os.path.join(HOME_DIR, 'vid2depth/inference')
DEFAULT_KITTI_DIR = os.path.join(HOME_DIR, 'kitti-raw-uncompressed')

flags.DEFINE_string('output_dir', DEFAULT_OUTPUT_DIR,
                    'Directory to store estimated depth maps.')
flags.DEFINE_string('kitti_dir', DEFAULT_KITTI_DIR, 'KITTI dataset directory.')
flags.DEFINE_string('train_mode', "test_depth", 'KITTI dataset directory.')
flags.DEFINE_string('model_ckpt', None, 'Model checkpoint to load.')
flags.DEFINE_string('kitti_video', None, 'KITTI video directory name.')
flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch.')
flags.DEFINE_integer('img_height', 256, 'Image height.')
flags.DEFINE_integer('img_width', 512, 'Image width.')
flags.DEFINE_integer('seq_length', 2, 'Sequence length for each example.')
FLAGS = flags.FLAGS

flags.mark_flag_as_required('kitti_video')
flags.mark_flag_as_required('model_ckpt')

CMAP = 'plasma'


def _run_inference():
  """Runs all images through depth model and saves depth maps."""
  ckpt_basename = os.path.basename(FLAGS.model_ckpt)
  # ckpt_modelname = os.path.basename(os.path.dirname(FLAGS.model_ckpt))
  # output_dir = os.path.join(FLAGS.output_dir,
  #                           FLAGS.kitti_video.replace('/', '_') + '_' +
  #                           ckpt_modelname + '_' + ckpt_basename)
  output_dir = os.path.join(FLAGS.output_dir,
                            FLAGS.kitti_video.replace('/', '_'))
  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)
  inference_model = model.Model(is_training=False,
                                seq_length=FLAGS.seq_length,
                                batch_size=FLAGS.batch_size,
                                img_height=FLAGS.img_height,
                                img_width=FLAGS.img_width,
                                train_mode=FLAGS.train_mode)
  # var_to_restore = util.get_vars_to_restore(FLAGS.model_ckpt)
  # var_to_restore = [v for v in var_to_restore if "depth_prediction" in v.op.name]

  var_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "depth_prediction")
  bn_vars = [v for v in tf.global_variables()
             if 'moving_mean' in v.op.name or 'moving_variance' in v.op.name]
  var_to_restore.extend(bn_vars)
  var_to_restore = sorted(var_to_restore, key=lambda x: x.op.name)
  restorer = tf.train.Saver(var_to_restore)
  for var in var_to_restore:
    print(var.name)

  # vars_to_restore = slim.get_variables_to_restore(include=["depth_prediction"])
  # for var in vars_to_restore:
  #   print(var.name)
  # renamaed_variables = {
  #    'model/' + var.op.name.split('depth_prediction/depth_net/')[1]: var
  #   for var in vars_to_restore
  # }
  # restorer = tf.train.Saver(renamaed_variables)


  sv = tf.train.Supervisor(logdir='/tmp/', saver=None)
  with sv.managed_session() as sess:
    # saver.restore(sess, FLAGS.model_ckpt)
    restorer.restore(sess, FLAGS.model_ckpt)
    if FLAGS.kitti_video == 'test_files_stereo':
      im_files = util.read_text_lines(
         'dataset/kitti/test_files_stereo.txt')
      im_files = [os.path.join(FLAGS.kitti_dir, f) for f in im_files]
    if FLAGS.kitti_video == 'test_files_eigen':
      im_files = util.read_text_lines(
         'dataset/kitti/test_files_eigen.txt')
      im_files = [os.path.join(FLAGS.kitti_dir, f) for f in im_files]
    # else:
    #   video_path = os.path.join(FLAGS.kitti_dir, FLAGS.kitti_video)
    #   im_files = gfile.Glob(os.path.join(video_path, 'image_02/data', '*.png'))
    #   im_files = [f for f in im_files if 'disp' not in f]
    #   im_files = sorted(im_files)

    depth = np.zeros((len(im_files), FLAGS.img_height, FLAGS.img_width), dtype=np.float32)
    for i in range(0, len(im_files), FLAGS.batch_size):
      if i % 100 == 0:
        logging.info('Generating from %s: %d/%d', ckpt_basename, i,
                     len(im_files))
      inputs = np.zeros(
          (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, 3),
          dtype=np.uint8)
      for b in range(FLAGS.batch_size):
        idx = i + b
        if idx >= len(im_files):
          break
        im = scipy.misc.imread(im_files[idx])
        inputs[b] = scipy.misc.imresize(im, (FLAGS.img_height, FLAGS.img_width))
      results = inference_model.inference(inputs, sess, mode='depth')
      for b in range(FLAGS.batch_size):
        idx = i + b
        if idx >= len(im_files):
          break
        if FLAGS.kitti_video == 'test_files_stereo':
          depth_path = os.path.join(output_dir, '%03d.png' % idx)
        else:
          depth_path = os.path.join(output_dir, '%04d.png' % idx)
        depth_map = results['depth'][b]
        depth_map = np.squeeze(depth_map)
        depth[idx] = depth_map
        # plt.imsave(depth_path, depth_map)
        colored_map = _normalize_depth_for_display(depth_map)
        input_float = inputs[b].astype(np.float32) / 255.0
        vertical_stack = np.concatenate((input_float, colored_map), axis=0)
        scipy.misc.imsave(depth_path, vertical_stack)
        # scipy.misc.imsave(depth_path, colored_map)

    np.save(FLAGS.output_dir + '/depth.npy', depth)


def _gray2rgb(im, cmap=CMAP):
  cmap = plt.get_cmap(cmap)
  rgba_img = cmap(im.astype(np.float32))
  rgb_img = np.delete(rgba_img, 3, 2)
  return rgb_img


def _normalize_depth_for_display(depth,
                                 pc=95,
                                 crop_percent=0,
                                 normalizer=None,
                                 cmap=CMAP):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.
  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  disp = _gray2rgb(disp, cmap=cmap)
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp


def main(_):
  _run_inference()


if __name__ == '__main__':
  app.run(main)
