from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc
from sfmmono_data.tools.lieFunctions import rotMat_to_euler
import time
# import sys
# sys.path.append('../../')
# from utils.misc import *

class kitti_odom_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7,8] # #12 left and right color is different ,11,13,14,15,16,17,18,19,20,21
        self.test_seqs = [ 9, 10]
        self. train_startFrames = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.train_endFrames = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070]
        self.test_startFrames = [0, 0]
        self.test_endFrames = [1590, 1200]

        # self.train_seqs = [0, 1, 2,  5, 8, 9]  # , 1, 2, 3, 4, 5, 6, 7, 8  ,11,12,13,14,15,16,17,18,19,20,21
        # self.test_seqs = [3, 4, 6, 7, 10]
        # self.train_startFrames = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.train_endFrames = [4540, 1100, 4660,  2760, 4070, 1590]
        # self.test_startFrames = [0, 0]
        # self.test_endFrames = [800, 270, 1100, 1100, 1200]

        self.collect_test_frames()
        self.collect_train_frames()
        # self.create_pose_data()

    def collect_test_frames(self):
        self.test_frames = []
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            print('seq_dir:', seq_dir)
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        print('num_test:',self.num_test)
        
    def collect_train_frames(self):
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_2')
            print('seq_dir:',seq_dir)
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)
        print('num_train:', self.num_train)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        # half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx
        max_src_idx = tgt_idx + self.seq_length -1
        # if tgt_idx == 1586:
        #     print(tgt_idx)
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        # half_offset = int((seq_length - 1)/2)
        image_seq = []
        image_seq_1 = []
        for o in range(seq_length):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img,curr_img_1 = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            curr_img_1 = scipy.misc.imresize(curr_img_1, (self.img_height, self.img_width))
            image_seq.append(curr_img) #list seq_len * image
            image_seq_1.append(curr_img_1)
        return image_seq, image_seq_1, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx, load_pose=False):
        image_seq, image_seq_1, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        # pose_seq = self.load_gt_pose_sequence(frames, tgt_idx, self.seq_length)
        # relative_pose_seq = self.load_relative_pose_sequence(pose_seq, tgt_idx, self.seq_length)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['image_seq_1'] = image_seq_1
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        # example['relative_pose_seq'] = relative_pose_seq
        if load_pose:
            pass
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def get_test_example_with_idx(self, tgt_idx):
        example = self.load_example(self.test_frames, tgt_idx)
        return example


    def load_image(self, drive, frame_id):
        img_file_2 = os.path.join(self.dataset_dir, 'sequences', '%s/image_2/%s.png' % (drive, frame_id))
        img_file_3 = os.path.join(self.dataset_dir, 'sequences', '%s/image_3/%s.png' % (drive, frame_id))
        img_2 = scipy.misc.imread(img_file_2)
        img_3 = scipy.misc.imread(img_file_3)
        return img_2, img_3

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics


    def read_calib_file(self, filepath, cid=2):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[cid], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    def scale_intrinsics(self,mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

    def load_gt_pose_sequence(self, frames, tgt_idx, seq_length):
        # half_offset = int((seq_length - 1) / 2)
        gt_pose_seq = []
        for o in range(seq_length):
            curr_idx = tgt_idx + o

            curr_gt_pose = self.train_pose_gt_seq[curr_idx]
            gt_pose_seq.append(curr_gt_pose)

        return gt_pose_seq

    def load_relative_pose_sequence(self, pose_seq, tgt_idx, seq_length):
        pose_mat = [np.reshape(value, (3,4)) for value in pose_seq]
        hfiller = np.array([0, 0, 0, 1]).reshape((1, 4))
        pose_mat = [np.concatenate((mat, hfiller), axis=0) for mat in pose_mat]
        pose_mat_inv = [np.linalg.inv(mat) for mat in pose_mat]

        last_pose = pose_mat_inv[0]
        # half_offset = int((seq_length - 1) / 2)
        relative_gt_pose_seq = []
        for i in range(seq_length):
            this_pose = pose_mat_inv[i]
            relative_pose_mat = np.dot(last_pose, np.linalg.inv(this_pose))
            relative_pose_T = relative_pose_mat[:3,-1]
            relative_pose_R = relative_pose_mat[:3,:3]
            relative_pose_euler = rotMat_to_euler(relative_pose_R,seq='xyz')
            relative_pose = np.concatenate((relative_pose_T,relative_pose_euler))
            last_pose = this_pose
            if i == 0:
                continue
            relative_gt_pose_seq.append(relative_pose)

        return relative_gt_pose_seq


    def create_pose_data(self):
        train_info = ['00', '01', '02',
                      '05',  '08', '09']
        test_info  = ['03', '04', '06', '07', '10']

        self.train_pose_gt_seq = []
        self.test_pose_gt_seq = []

        #train
        for video in train_info:
            pose_dir = os.path.join(self.dataset_dir,'poses')
            fn = '{}/{}.txt'.format(pose_dir, video)
            # self.dataset_dir, 'sequences', '%.2d' % seq
            print('Transforming {}...'.format(fn))
            with open(fn) as f:
                lines = [line.split('\n')[0] for line in f.readlines()]
                poses_original = [[float(value) for value in l.split(' ')] for l in lines]
                self.train_pose_gt_seq.extend(poses_original)
                # poses = np.array(poses)
                # base_fn = os.path.splitext(fn)[0]
                # np.save(base_fn + '.npy', poses)
                # print('Video {}: shape={}'.format(video, poses.shape))
        print('train_pose_gt_len',len(self.train_pose_gt_seq))
        # print('elapsed time = {}'.format(time.time() - start_t))

        #test
        for video in test_info:
            pose_dir = os.path.join(self.dataset_dir,'poses')
            fn = '{}/{}.txt'.format(pose_dir, video)
            print('Transforming {}...'.format(fn))
            with open(fn) as f:
                lines = [line.split('\n')[0] for line in f.readlines()]
                poses_original = [[float(value) for value in l.split(' ')] for l in lines]
                self.test_pose_gt_seq.extend(poses_original)
        print('test_pose_gt_len',len(self.test_pose_gt_seq))
