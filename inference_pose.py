from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
# from SfMLearner import SfMLearner
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM,euler2mat
from matplotlib import pyplot as plt
from absl import app
from absl import flags
from absl import logging
import model
import util

# --test_seq 9
# --dataset_dir "/media/wuqi/ubuntu/dataset/kitti/data_odometry_color/dataset/"
# --output_dir
# /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat/output/
# --ckpt_file
# /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat/model-31108
# --img_height 128
# --img_width 416

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 256, "Image height")
flags.DEFINE_integer("img_width", 512, "Image width")
flags.DEFINE_integer("seq_length", 2, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("func", 'generate_odom', "Select function (g generate_odom; eval_odom)")
FLAGS = flags.FLAGS


class kittiEvalOdom():
    # ----------------------------------------------------------------------
    # poses: N,4,4
    # pose: 4,4
    # ----------------------------------------------------------------------
    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        self.gt_dir = "/media/wuqi/ubuntu/dataset/kitti/data_odometry_color/dataset/poses"

    def loadPoses(self, file_name):
        # ----------------------------------------------------------------------
        # Each line in the file should follow one of the following structures
        # (1) idx pose(3x4 matrix in terms of 12 numbers)
        # (2) pose(3x4 matrix in terms of 12 numbers)
        # ----------------------------------------------------------------------
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ")]
            withIdx = int(len(line_split) == 13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectoryDistances(self, poses):
        # ----------------------------------------------------------------------
        # poses: dictionary: [frame_idx: pose]
        # ----------------------------------------------------------------------
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    def rotationError(self, pose_error):
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        return np.arccos(max(min(d, 1.0), -1.0))

    def translationError(self, pose_error):
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def calcSequenceErrors(self, poses_gt, poses_result):
        err = []
        dist = self.trajectoryDistances(poses_gt)
        self.step_size = 10

        for first_frame in range(9, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.lastFrameFromSegmentLength(dist, first_frame, len_)

                # ----------------------------------------------------------------------
                # Continue if sequence not long enough
                # ----------------------------------------------------------------------
                if last_frame == -1 or not (last_frame in poses_result.keys()) or not (
                        first_frame in poses_result.keys()):
                    continue

                # ----------------------------------------------------------------------
                # compute rotational and translational errors
                # ----------------------------------------------------------------------
                pose_delta_gt = np.dot(np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame])
                pose_delta_result = np.dot(np.linalg.inv(poses_result[first_frame]), poses_result[last_frame])
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotationError(pose_error)
                t_err = self.translationError(pose_error)

                # ----------------------------------------------------------------------
                # compute speed
                # ----------------------------------------------------------------------
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def saveSequenceErrors(self, err, file_name):
        fp = open(file_name, 'w')
        for i in err:
            line_to_write = " ".join([str(j) for j in i])
            fp.writelines(line_to_write + "\n")
        fp.close()

    def computeOverallErr(self, seq_err):
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        for item in seq_err:
            r_err += item[1]
            t_err += item[2]
        ave_t_err = t_err / seq_len
        ave_r_err = r_err / seq_len
        return ave_t_err, ave_r_err

    def plotPath(self, seq, poses_gt, poses_result):
        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num = -1

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict["Ours"] = poses_result

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')

        for key in plot_keys:
            pos_xz = []
            # for pose in poses_dict[key]:
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.scatter(0, 0, s=50)

        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{:02}".format(seq)
        plt.savefig(self.plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)

    # plt.show()

    # def plotError(self, avg_segment_errs):
    #     # ----------------------------------------------------------------------
    #     # avg_segment_errs: dict [100: err, 200: err...]
    #     # ----------------------------------------------------------------------
    #     plot_y = []
    #     plot_x = []
    #     for len_ in self.lengths:
    #         plot_x.append(len_)
    #         plot_y.append(avg_segment_errs[len_][0])
    #     fig = plt.figure()
    #     plt.plot(plot_x, plot_y)
    #     # plt.show()

    def plotError(self, avg_errs):
        avg_seg_errs = avg_errs[0]
        avg_speed_errs = avg_errs[1]

        plot_keys = ["Ground Truth", "Ours"]
        fontsize_ = 20
        plot_num = -1

        plot_x = []
        plot_t_errs = []
        plot_r_errs = []
        for len_ in self.lengths:
            plot_x.append(len_)
            plot_t_errs.append(avg_seg_errs[len_][0] * 100)
            plot_r_errs.append((avg_seg_errs[len_][1] / np.pi * 180 * 100))

        fig = plt.figure()
        plt.plot(plot_x,plot_t_errs,label='tran err', c='r', marker='d')
        # plt.scatter(plot_x, plot_t_errs)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.ylim(0,20)
        plt.xlabel('Path Length [m]', fontsize=fontsize_)
        plt.ylabel('Translation Error [%%]', fontsize=fontsize_)
        png_title = "avg_tl_error"
        plt.savefig(self.plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)

        fig = plt.figure()
        plt.plot(plot_x,plot_r_errs,label='rot_err', c='b', marker='D')
        plt.scatter(plot_x, plot_r_errs)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.ylim(0, 10)
        plt.xlabel('Path Length [m]', fontsize=fontsize_)
        plt.ylabel('Rotation Error [deg/m]', fontsize=fontsize_)
        png_title = "avg_rl_error"
        plt.savefig(self.plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)

        plot_x = []
        plot_t_errs = []
        plot_r_errs = []
        for speed_ in range(2,24,2):
            if avg_speed_errs[speed_] != []:
                plot_x.append(speed_)
                plot_t_errs.append(avg_speed_errs[speed_][0] * 100)
                plot_r_errs.append((avg_speed_errs[speed_][1] / np.pi * 180 * 100))

        fig = plt.figure()
        plt.plot(plot_x, plot_t_errs, label='tran err', c='r', marker='d')
        # plt.scatter(plot_x, plot_t_errs)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.ylim(0, 20)
        plt.xlabel('Speed [km/h]', fontsize=fontsize_)
        plt.ylabel('Translation Error [%%]', fontsize=fontsize_)
        png_title = "avg_ts_error"
        plt.savefig(self.plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)

        fig = plt.figure()
        plt.plot(plot_x, plot_r_errs, label='rot_err', c='b', marker='D')
        plt.scatter(plot_x, plot_r_errs)
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.ylim(0, 10)
        plt.xlabel('Speed [km/h]', fontsize=fontsize_)
        plt.ylabel('Rotation Error [deg/m]', fontsize=fontsize_)
        png_title = "avg_rs_error"
        plt.savefig(self.plot_error_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0)



    def computeavgErr(self, seq_errs):

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        speed_errs = {}
        avg_speed_errs = {}
        for speed_ in range(2, 24, 2):
            speed_errs[speed_] = []
        # ----------------------------------------------------------------------
        # Get errors
        # ----------------------------------------------------------------------
        for err in seq_errs:
            speed_ = int(math.floor(err[4])/2)*2
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])
            speed_errs[speed_].append([t_err, r_err])
        # ----------------------------------------------------------------------
        # Compute average
        # ----------------------------------------------------------------------
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []

        for speed_ in range(2, 24, 2):
            if speed_errs[speed_] != []:
                avg_t_err = np.mean(np.asarray(speed_errs[speed_])[:, 0])
                avg_r_err = np.mean(np.asarray(speed_errs[speed_])[:, 1])
                avg_speed_errs[speed_] = [avg_t_err, avg_r_err]
            else:
                avg_speed_errs[speed_] = []
        return [avg_segment_errs, avg_speed_errs]

    # def computeSegmentErr(self, seq_errs):
    #     # ----------------------------------------------------------------------
    #     # This function calculates average errors for different segment.
    #     # ----------------------------------------------------------------------
    #
    #     segment_errs = {}
    #     avg_segment_errs = {}
    #     for len_ in self.lengths:
    #         segment_errs[len_] = []
    #     # ----------------------------------------------------------------------
    #     # Get errors
    #     # ----------------------------------------------------------------------
    #     for err in seq_errs:
    #         len_ = err[3]
    #         t_err = err[2]
    #         r_err = err[1]
    #         segment_errs[len_].append([t_err, r_err])
    #     # ----------------------------------------------------------------------
    #     # Compute average
    #     # ----------------------------------------------------------------------
    #     for len_ in self.lengths:
    #         if segment_errs[len_] != []:
    #             avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
    #             avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
    #             avg_segment_errs[len_] = [avg_t_err, avg_r_err]
    #         else:
    #             avg_segment_errs[len_] = []
    #     return avg_segment_errs

    def eval(self, result_dir):
        self.error_dir = result_dir + "/errors"
        self.plot_path_dir = result_dir + "/plot_path"
        self.plot_error_dir = result_dir + "/plot_error"

        if not os.path.exists(self.error_dir):
            os.makedirs(self.error_dir)
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)
        if not os.path.exists(self.plot_error_dir):
            os.makedirs(self.plot_error_dir)

        total_err = []

        ave_t_errs = []
        ave_r_errs = []

        for i in self.eval_seqs:
            self.cur_seq = '{:02}'.format(i)
            file_name = '{:02}.txt'.format(i)

            poses_result = self.loadPoses(result_dir + "/" + file_name)
            poses_gt = self.loadPoses(self.gt_dir + "/" + file_name)
            self.result_file_name = result_dir + file_name

            # ----------------------------------------------------------------------
            # compute sequence errors
            # ----------------------------------------------------------------------
            seq_err = self.calcSequenceErrors(poses_gt, poses_result)
            self.saveSequenceErrors(seq_err, self.error_dir + "/" + file_name)

            #add total err
            total_err.extend(seq_err)

            # # ----------------------------------------------------------------------
            # # Compute segment errors
            # # ----------------------------------------------------------------------
            # avg_segment_errs = self.computeSegmentErr(seq_err)

            # ----------------------------------------------------------------------
            # compute overall error
            # ----------------------------------------------------------------------
            ave_t_err, ave_r_err = self.computeOverallErr(seq_err)

            print("Sequence: " + str(i))
            print("Average translational RMSE (%): ", ave_t_err * 100)
            print( "Average rotational error (deg/100m): ", ave_r_err / np.pi * 180 * 100)

            ave_t_errs.append(ave_t_err)
            ave_r_errs.append(ave_r_err)

            # ----------------------------------------------------------------------
            # Ploting (To-do)
            # (1) plot trajectory
            # (2) plot per segment error
            # ----------------------------------------------------------------------
            self.plotPath(i, poses_gt, poses_result)
        # self.plotError(avg_segment_errs)

        total_avg_err = self.computeavgErr(total_err)
        self.plotError(total_avg_err)




        print("-------------------- For Copying ------------------------------")

        for i in range(len(ave_t_errs)):
            # print("Sequence: " + str(i))

            print("{0:.2f}".format(ave_t_errs[i] * 100))

            print("{0:.2f}".format(ave_r_errs[i] / np.pi * 180 * 100))
        print("-------------------- For copying ------------------------------")


class kittipreodom():
    def __init__(self):
        pass

    def load_image_sequence(self,dataset_dir,
                            frames,
                            tgt_idx,
                            seq_length,
                            img_height,
                            img_width):
        half_offset = int((seq_length - 1) / 2)
        for o in range(seq_length):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            img_file = os.path.join(
                dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
            curr_img = scipy.misc.imread(img_file)
            curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
            if o == 0:
                image_seq = curr_img
            else:
                image_seq = np.hstack((image_seq, curr_img))
        return image_seq

    def is_valid_sample(self,frames, tgt_idx, seq_length):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        # max_src_offset = int((seq_length - 1)/2)
        min_src_idx = tgt_idx
        max_src_idx = min_src_idx + seq_length - 1
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        # TODO: unnecessary to check if the drives match
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def SE3_cam2world(self,pred_poses):
        cur_T = np.eye(4)
        tmp_SE3_world = []
        tmp_SE3_world.append(cur_T)
        filler = np.array([0, 0, 0, 1]).reshape((1, 4))
        for pose in pred_poses:
            pose = np.concatenate((pose, filler), axis=0)
            cur_T = np.dot(cur_T, pose)
            tmp_SE3_world.append(cur_T)
        return tmp_SE3_world

    def saveResultPoses(self,pred_poses_list_word):
        result_dir = FLAGS.output_dir
        output_file = result_dir + '/%.2d.txt' % FLAGS.test_seq
        with open(output_file, 'w') as f:
            for cnt, SE3 in enumerate(pred_poses_list_word):
                tx = str(SE3[0, 3])
                ty = str(SE3[1, 3])
                tz = str(SE3[2, 3])
                R00 = str(SE3[0, 0])
                R01 = str(SE3[0, 1])
                R02 = str(SE3[0, 2])
                R10 = str(SE3[1, 0])
                R11 = str(SE3[1, 1])
                R12 = str(SE3[1, 2])
                R20 = str(SE3[2, 0])
                R21 = str(SE3[2, 1])
                R22 = str(SE3[2, 2])
                line_to_write = " ".join([R00, R01, R02, tx, R10, R11, R12, ty, R20, R21, R22, tz])
                f.writelines(line_to_write + "\n")

    def getpreposes(self):
        # sfm = SfMLearner(FLAGS)
        # sfm.setup_inference(FLAGS.img_height,
        #                     FLAGS.img_width,
        #                     'pose',
        #                     FLAGS.seq_length)
        inference_model = model.Model(is_training=False,
                                      seq_length=FLAGS.seq_length,
                                      batch_size=FLAGS.batch_size,
                                      img_height=FLAGS.img_height,
                                      img_width=FLAGS.img_width)
        var_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "egomotion_prediction")
        saver = tf.train.Saver(var_to_restore)

        #[var for var in tf.trainable_variables()]

        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        seq_dir = os.path.join(FLAGS.dataset_dir, 'sequences', '%.2d' % FLAGS.test_seq)
        img_dir = os.path.join(seq_dir, 'image_2')
        N = len(glob(img_dir + '/*.png'))
        test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]
        with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
            times = f.readlines()

        pred_poses_list = []
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.ckpt_file)
            for tgt_idx in range(N):
                if not self.is_valid_sample(test_frames, tgt_idx, FLAGS.seq_length):
                    continue
                if tgt_idx % 100 == 0:
                    print('Progress: %d/%d' % (tgt_idx, N))
                # TODO: currently assuming batch_size = 1
                image_seq = self.load_image_sequence(FLAGS.dataset_dir,
                                                test_frames,
                                                tgt_idx,
                                                FLAGS.seq_length,
                                                FLAGS.img_height,
                                                FLAGS.img_width)
                # pred = sfm.inference(image_seq[None, :, :, :], sess, mode='pose')
                pred = inference_model.inference(image_seq[None, :, :, :], sess, mode='egomotion')
                pred_poses = pred['egomotion'][0, 0]

                pose_tran = np.array(pred_poses[:3]).reshape((3, 1))
                pose_euler = pred_poses[3:]
                pose_mat_rot = euler2mat(pose_euler[2], pose_euler[1], pose_euler[0])
                pose_mat = np.concatenate((pose_mat_rot, pose_tran), axis=1)

                pred_poses_list.append(pose_mat)

        pred_poses_list_word = self.SE3_cam2world(pred_poses_list)
        self.saveResultPoses(pred_poses_list_word)


if FLAGS.func == "generate_odom":
    generator = kittipreodom()
    generator.getpreposes()

elif FLAGS.func == "eval_odom":
    odom_eval = kittiEvalOdom()
    odom_eval.eval_seqs = [5,6,7,8,9,10]  # Seq 03 is missing since the dataset is not available in KITTI homepage.
    odom_eval.eval(FLAGS.output_dir)