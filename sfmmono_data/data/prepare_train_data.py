from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import random

# --dataset_dir=/media/wuqi/ubuntu/dataset/kitti/data_odometry_color/dataset/
# --dataset_name="kitti_odom"
# --dump_root=/media/wuqi/works/dataset/kitti_odom/flowodometry_split_skip4_416_128_vid2pose/
# --seq_length=5
# --img_width=416
# --img_height=128
# --num_threads=6


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True,
                    choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root", type=str, required=True,
                    help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True,
                    help="Length of each training sequence")
parser.add_argument("--mode", type=str, default="train",
                    help="data train or test")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4,
                    help="number of threads to use")
args = parser.parse_args()


def concat_image_seq_train(seq,  args):

    seq_2 = [np.hstack((im1, im2)) for (im1, im2) in zip(seq[:args.seq_length], seq[1:])]

    for i, im in enumerate(seq_2):
        if i == 0:
            res = im
        else:
            res = np.vstack((res, im))
    return res

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

# def concat_image_seq_test(seq,  args):
#
#     seq_2 = [np.hstack((im1, im2)) for (im1, im2) in zip(seq[:args.seq_length], seq[1:])]
#
#     for i, im in enumerate(seq_2):
#         if i == 0:
#             res = im
#         else:
#             res = np.vstack((res, im))
#     return res

def dump_example(n, args):
    if n % 2000 == 0:
        print('/Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        print('false')
        return
    # image_seq = concat_image_seq(example['image_seq'],  args)
    # image_seq_1 = concat_image_seq(example['image_seq_1'], args)
    image_seq = concat_image_seq(example['image_seq'])
    image_seq_1 = concat_image_seq(example['image_seq_1'])
    image_seq_lr = np.vstack((image_seq,image_seq_1))
    intrinsics = example['intrinsics']
    # rel_gt_pose = example['relative_pose_seq']

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    # print(image_seq)

    scipy.misc.imsave(dump_img_file, image_seq_lr.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

    # dump_gt_pose_file = dump_dir + '/%s_gt_rel_pose.txt' % example['file_name']
    # with open(dump_gt_pose_file,'w') as f:
    #     for rel_pose in rel_gt_pose:
    #         f.write('%.6e %.6e %.6e %.6e %.6e %.6e\n' %(rel_pose[0], rel_pose[1], rel_pose[2],
    #                                        rel_pose[3], rel_pose[4], rel_pose[5]))


def dump_example_test(n, args):
    if n % 2000 == 0:
        print('/Progress %d/%d....' % (n, data_loader.num_test))
    example = data_loader.get_test_example_with_idx(n)
    if example == False:
        print('false')
        return
    image_seq = concat_image_seq(example['image_seq'])
    image_seq_1 = concat_image_seq(example['image_seq_1'])
    image_seq_lr = np.vstack((image_seq, image_seq_1))
    intrinsics = example['intrinsics']
    # rel_gt_pose = example['relative_pose_seq']

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']


    scipy.misc.imsave(dump_img_file, image_seq_lr.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

    # dump_gt_pose_file = dump_dir + '/%s_gt_rel_pose.txt' % example['file_name']
    # with open(dump_gt_pose_file, 'w') as f:
    #     for rel_pose in rel_gt_pose:
    #         f.write('%.6e %.6e %.6e %.6e %.6e %.6e\n' % (rel_pose[0], rel_pose[1], rel_pose[2],
    #                                                      rel_pose[3], rel_pose[4], rel_pose[5]))


def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    print(args.mode)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from sfmmono_data.data.kitti.kitti_odom_loader import kitti_odom_loader
        print(args.dataset_dir)
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from sfmmono_data.data.kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_stereo':
        from sfmmono_data.data.kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'cityscapes':
        from sfmmono_data.data.cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.mode == "train":
        args.dump_root = args.dump_root + '/train/'

        if not os.path.exists(args.dump_root):
            os.makedirs(args.dump_root)

        Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(0,data_loader.num_train,2))

        # Split into train/val
        np.random.seed(8964)
        subfolders = os.listdir(args.dump_root)
        # print(args.dump_root)
        with open(args.dump_root + 'train.txt', 'w') as tf:
            with open(args.dump_root + 'val.txt', 'w') as vf:
                frame_ids_shuffle = []
                for s in subfolders:
                    if not os.path.isdir(args.dump_root + '/%s' % s):
                        continue
                    imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                    frame_ids = ['%s '%s + str(os.path.basename(fi).split('.')[
                                     0]) for fi in imfiles]
                    # frame_ids = [os.path.basename(fi).split('.')[
                    #                  0] for fi in imfiles]
                    random.shuffle(frame_ids)
                    frame_ids_shuffle.extend(frame_ids)
                    # for frame in frame_ids:
                    #     # if np.random.random() < 0.06:
                    #     #     vf.write('%s %s\n' % (s, frame))
                    #     # else:
                    #     tf.write('%s %s\n' % (s, frame))

                random.shuffle(frame_ids_shuffle)
                for frame in frame_ids_shuffle:
                    # if np.random.random() < 0.06:
                    #     vf.write('%s %s\n' % (s, frame))
                    # else:
                    tf.write('%s\n' % (frame))
    else:
        args.dump_root = args.dump_root + '/test/'

        if not os.path.exists(args.dump_root):
            os.makedirs(args.dump_root)

        # Parallel(n_jobs=args.num_threads)(delayed(dump_example_test)(n, args) for n in range(0,data_loader.num_test,4))

        np.random.seed(8964)
        subfolders = os.listdir(args.dump_root)
        # print(args.dump_root)
        with open(args.dump_root + 'test.txt', 'w') as tf:
            frame_ids_shuffle = []
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = ['%s ' % s + str(os.path.basename(fi).split('.')[
                                                 0]) for fi in imfiles]
                random.shuffle(frame_ids)
                frame_ids_shuffle.extend(frame_ids)
            random.shuffle(frame_ids_shuffle)
            for frame in frame_ids_shuffle:
                tf.write('%s\n' %(frame))




main()
