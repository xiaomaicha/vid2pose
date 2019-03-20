import numpy as np
import cv2
import argparse
from utils.evaluation_utils import *

# utils/evaluate_kitti.py
# --split kitti --predicted_disp_path /media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat/inv_depth/test_files_stereo_sl_5_skip4_416_128_depthvofeat_model-31108/inv_depth.npy
# --gt_path /media/wuqi/ubuntu/dataset/stereo/KITTI2015/data_scene_flow

parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=False)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1e-3)
parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80)
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')

args = parser.parse_args()

if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)

    if args.split == 'kitti':
        num_samples = 200
        
        gt_disparities = load_gt_disp_kitti(args.gt_path)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)

    elif args.split == 'eigen':
        num_samples = 697
        # test_files = read_text_lines(args.gt_path + 'test_files_eigen.txt')
        # gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args.gt_path)

        num_test = 697
        # gt_depths = []
        gt_depths = np.load(args.gt_path, encoding="latin1")
        pred_depths = []
        for t_id in range(num_samples):
            height, width = gt_depths[t_id].shape
            # camera_id = cams[t_id]  # 2 is left, 3 is right
            # depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            # gt_depths.append(depth.astype(np.float32))

            # disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
            # disp_pred = disp_pred * disp_pred.shape[1]
            #
            # # need to convert from disparity to depth
            # focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
            # depth_pred = (baseline * focal_length) / disp_pred
            # depth_pred[np.isinf(depth_pred)] = 0



            depth_pred = cv2.resize(pred_disparities[t_id], (width, height), interpolation=cv2.INTER_LINEAR)
            # 1.0 / (inv_depth_pred + 1e-4)
            depth_pred[np.isinf(depth_pred)] = 0


            pred_depths.append(depth_pred)

        # np.save(args.gt_path + 'gt_depth.npy',gt_depths)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        if i % 100 == 0:
            print(i, '\n')

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        if args.split == 'eigen':
            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)


            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                # crop used by Garg ECCV16
                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                if args.garg_crop:
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                # crop we found by trial and error to reproduce Eigen NIPS14 results
                elif args.eigen_crop:
                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

        if args.split == 'kitti':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0.05
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
