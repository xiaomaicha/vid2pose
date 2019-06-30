from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
# from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM, euler2mat
from matplotlib import pyplot as plt
from absl import app
from absl import flags
from absl import logging
import model
import util
import flowlib as fl


'''
--img_height
128
--img_width
416
--dataset_dir
/home/lli/kitti__raw/Kitti_stereo_2015/data_scene_flow/KITTI15/training
--output_dir
/home/lli/tensorflow/vid2pose_log/odpm_00_10_skip1_relu_relu_deconv_1/predict_flow
--ckpt_file
/home/lli/tensorflow/vid2pose_log/odpm_00_10_skip1_relu_relu_deconv_1/model-minloss-179200
--func
eval_flow
'''



flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")
flags.DEFINE_integer("seq_length", 2, "Sequence length for each example")
# flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", '/home/lli/kitti__raw/Kitti_stereo_2015/data_scene_flow/KITTI15/training', "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("func", 'generate_flow', "Select function suggest:(1. generate_flow; 2. eval_flow)")
FLAGS = flags.FLAGS


class kittipredict_flow():
    def __init__(self):
        pass

    def load_image_sequence(self,
                            dataset_dir,
                            frames,
                            tgt_idx,
                            seq_length,
                            img_height,
                            img_width):
        for o in range(seq_length):
            curr_frame_id = frames[tgt_idx].split(' ')[o]
            img_file = os.path.join(
                dataset_dir, 'image_2/%s.png' % (curr_frame_id))
            curr_img = scipy.misc.imread(img_file)
            curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
            if o == 0:
                image_seq = curr_img
            else:
                image_seq = np.hstack((image_seq, curr_img))
        # frame_0_id = frames[tgt_idx].split(' ')[0]
        # frame_1_id = frames[tgt_idx].split(' ')[1]



        return image_seq

    # def is_valid_sample(self, frames, tgt_idx, seq_length):
    #     N = len(frames)
    #     tgt_0, tgt_1 = frames[tgt_idx].split(' ')
    #     # max_src_offset = int((seq_length - 1)/2)
    #     min_src_idx = tgt_idx
    #     max_src_idx = min_src_idx + seq_length - 1
    #     if min_src_idx < 0 or max_src_idx >= N:
    #         return False
    #     # TODO: unnecessary to check if the drives match
    #     min_src_drive, _ = frames[min_src_idx].split(' ')
    #     max_src_drive, _ = frames[max_src_idx].split(' ')
    #     if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
    #         return True
    #     return False

    def getpreflow(self):
        # sfm = SfMLearner(FLAGS)
        # sfm.setup_inference(FLAGS.img_height,
        #                     FLAGS.img_width,
        #                     'pose',
        #                     FLAGS.seq_length)
        inference_model = model.Model(is_training=False,
                                      train_mode="test_flow",
                                      seq_length=FLAGS.seq_length,
                                      batch_size=FLAGS.batch_size,
                                      img_height=FLAGS.img_height,
                                      img_width=FLAGS.img_width)
        var_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "egomotion_prediction")
        saver = tf.train.Saver(var_to_restore)

        # [var for var in tf.trainable_variables()]

        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        # seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
        img_dir = os.path.join(FLAGS.dataset_dir, 'image_2')
        a = glob(img_dir + '/*.png')
        N = len(glob(img_dir + '/*.png'))
        test_frames = ['%.6d_10 %.6d_11' % (n, n) for n in range(N//2)]


        output_file = FLAGS.output_dir
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        binary_dir = os.path.join(output_file, 'binary')
        color_dir = os.path.join(output_file, 'color')
        png_dir = os.path.join(output_file, 'png')
        if (not os.path.exists(binary_dir)):
            os.makedirs(binary_dir)
        if (not os.path.exists(color_dir)):
            os.makedirs(color_dir)
        if (not os.path.exists(png_dir)):
            os.makedirs(png_dir)
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            print('model load successfully!!')
            for tgt_idx in range(N//2):
                if tgt_idx % 50 == 0:
                    print('Progress: %d/%d' % (tgt_idx, N//2))
                # TODO: currently assuming batch_size = 1
                image_seq = self.load_image_sequence(FLAGS.dataset_dir,
                                                     test_frames,
                                                     tgt_idx,
                                                     FLAGS.seq_length,
                                                     FLAGS.img_height,
                                                     FLAGS.img_width)
                # pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
                pred = inference_model.inference(image_seq[None, :, :, :], sess, mode='flow')
                pred_flow = pred['flow'][0]
                pred_flow = np.squeeze(pred_flow,axis=0)

                flow_fn = '%.6d.png' % tgt_idx
                color_fn = os.path.join(color_dir, flow_fn)
                color_flow = fl.flow_to_image(pred_flow)
                color_flow = cv2.cvtColor(color_flow, cv2.COLOR_RGB2BGR)
                color_flow = cv2.imwrite(color_fn, color_flow)

                png_fn = os.path.join(png_dir, flow_fn)
                mask_blob = np.ones((FLAGS.img_height, FLAGS.img_width), dtype=np.uint16)
                fl.write_kitti_png_file(png_fn, pred_flow, mask_blob)

                binary_fn = flow_fn.replace('.png', '.flo')
                binary_fn = os.path.join(binary_dir, binary_fn)
                fl.write_flow(pred_flow, binary_fn)

class kittiEvalFlow():
    def __init__(self):
        pass

    def eval(self):
        img_num = 200
        noc_epe = np.zeros(img_num, dtype=np.float)
        noc_acc = np.zeros(img_num, dtype=np.float)
        occ_epe = np.zeros(img_num, dtype=np.float)
        occ_acc = np.zeros(img_num, dtype=np.float)

        eval_log = os.path.join(FLAGS.output_dir, 'flow_result.txt')
        with open(eval_log, 'w') as el:
            for idx in range(img_num):
                # read groundtruth flow
                gt_noc_fn = FLAGS.dataset_dir + '/flow_noc/%.6d_10.png' % idx
                gt_occ_fn = FLAGS.dataset_dir + '/flow_occ/%.6d_10.png' % idx
                gt_noc_flow = fl.read_flow(gt_noc_fn)
                gt_occ_flow = fl.read_flow(gt_occ_fn)

                # read predicted flow (in png format)
                pred_flow_fn = FLAGS.output_dir + '/png/%.6d.png' % idx
                pred_flow = fl.read_flow(pred_flow_fn)

                # resize pred_flow to the same size as gt_flow
                dst_h = gt_noc_flow.shape[0]
                dst_w = gt_noc_flow.shape[1]
                pred_flow = fl.resize_flow(pred_flow, dst_w, dst_h)

                # evaluation
                (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(gt_noc_flow, pred_flow, None)
                (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(gt_occ_flow, pred_flow, None)
                noc_epe[idx] = single_noc_epe
                noc_acc[idx] = single_noc_acc
                occ_epe[idx] = single_occ_epe
                occ_acc[idx] = single_occ_acc
                output_line = 'Flow %.6d Noc EPE = %.4f' + ' Noc ACC = %.4f' + ' Occ EPE = %.4f' + ' Occ ACC = %.4f\n';
                el.write(output_line % (idx, noc_epe[idx], noc_acc[idx], occ_epe[idx], occ_acc[idx]))

        noc_mean_epe = np.mean(noc_epe)
        noc_mean_acc = np.mean(noc_acc)
        occ_mean_epe = np.mean(occ_epe)
        occ_mean_acc = np.mean(occ_acc)

        print('Mean Noc EPE = %.4f ' % noc_mean_epe)
        print('Mean Noc ACC = %.4f ' % noc_mean_acc)
        print('Mean Occ EPE = %.4f ' % occ_mean_epe)
        print('Mean Occ ACC = %.4f ' % occ_mean_acc)



if FLAGS.func == "generate_flow":
    generator = kittipredict_flow()
    generator.getpreflow()

elif FLAGS.func == "eval_flow":
    odom_eval = kittiEvalFlow()
    odom_eval.eval()

