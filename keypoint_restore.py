import tensorflow as tf

# ckpt = '/media/wuqi/ubuntu/code/slam/monodepth/flow_checkpoints/left_lr/model-40000'
# ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
# # ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
# # ckpt_var_names = sorted(ckpt_var_names, key=lambda x: x.op.name)
# for name in ckpt_var_names:
#     print(name)

ckpt = '/media/wuqi/ubuntu/code/slam/vid2pose_log/sl_5_skip4_416_128_depthvofeat_relu_sigmoid_upconv_sad_left_right/depth_model_kitti_2015/model-179856'
ckpt_var_names = tf.contrib.framework.list_variables(ckpt)
# ckpt_var_names = [name for (name, unused_shape) in ckpt_var_names]
# ckpt_var_names = sorted(ckpt_var_names, key=lambda x: x.op.name)
for name in ckpt_var_names:
    print(name)