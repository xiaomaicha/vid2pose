from __future__ import division
import numpy as np
import os
import cv2
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="path to kitti scene flow multiview dataset")
parser.add_argument("--calib_dir",   type=str, help="path to data_scene_flow_calib")
parser.add_argument("--dump_root",   type=str, help="where to dump the data")
parser.add_argument("--cam_id",      type=str, default='02', help="camera id")
parser.add_argument("--seq_length",  type=int, default=3, help="sequence length of pose snippets")
parser.add_argument("--img_height",  type=int, default=256, help="image height")
parser.add_argument("--img_width",   type=int, default=512, help="image width")
args = parser.parse_args()

# --dataset_dir=/media/wuqi/ubuntu/dataset/stereo/KITTI2015/data_scene_flow_multiview/
# --calib_dir=/media/wuqi/ubuntu/dataset/stereo/KITTI2015/data_scene_flow_calib/
# --dump_root=/media/wuqi/works/dataset/kitti_raw_stereo_416_128/test/
# --cam_id=02
# --seq_length=3

def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                    data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                    pass
    return data

def load_intrinsics_raw(dataset_dir, fr_id, cid):
    calib_file = os.path.join(dataset_dir, '%.6d.txt' % fr_id)

    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
    intrinsics = P_rect[:3, :3]
    return intrinsics

def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy
    return out

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def main():
    frame_list = list(range(200))
    random.shuffle(frame_list)
    # read calib files
    calib_path = os.path.join(args.calib_dir, 'training', 'calib_cam_to_cam')
    intri_list = []
    for i in frame_list:
        intri = load_intrinsics_raw(calib_path, i, args.cam_id)
        intri_list.append(intri)

    # generate test examples
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)
    with open(os.path.join(args.dump_root, 'test.txt'), 'w') as tf:
        for i in frame_list:
            half_offset = int((args.seq_length-1)/2)
            img_seq_left = []
            img_seq_right = []
            for o in range(args.seq_length):
                curr_img_path_left = args.dataset_dir + 'training/image_%s/%.6d_%.2d.png' % (args.cam_id[-1], i, 10+o)
                curr_img_left = cv2.imread(curr_img_path_left)
                if o==0:
                    zoom_y = args.img_height/(curr_img_left.shape[0])
                    zoom_x = args.img_width/(curr_img_left.shape[1])
                curr_img_left = cv2.resize(curr_img_left, (args.img_width, args.img_height))
                img_seq_left.append(curr_img_left)

                curr_img_path_right = args.dataset_dir + 'training/image_%s/%.6d_%.2d.png' % ('3', i, 10 + o)
                curr_img_right = cv2.imread(curr_img_path_right)
                curr_img_right = cv2.resize(curr_img_right, (args.img_width, args.img_height))
                img_seq_right.append(curr_img_right)

            intrinsics = scale_intrinsics(intri_list[i], zoom_x, zoom_y)
            img_seq_left = concat_image_seq(img_seq_left)
            img_seq_right = concat_image_seq(img_seq_right)
            img_seq = np.vstack((img_seq_left, img_seq_right))
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            dump_dir = os.path.join(args.dump_root, 'kitti_2015_%s' % args.cam_id)
            # dump_dir = os.path.join(args.dump_root, 'test')
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            dump_img_file = dump_dir + '/%.6d.jpg' % i
            cv2.imwrite(dump_img_file, img_seq.astype(np.uint8))
            dump_cam_file = dump_dir + '/%.6d_cam.txt' % i
            with open(dump_cam_file, 'w') as f:
                f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))
            tf.write('kitti_2015_%s %.6d\n' % (args.cam_id, i))

main()
