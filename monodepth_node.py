# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', default='images/room.png')
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default='models/model_eigen')
parser.add_argument('--input_height',     type=int,   help='input height', default=240)
parser.add_argument('--input_width',      type=int,   help='input width', default=320)

args = parser.parse_args()

params = monodepth_parameters(
    encoder=args.encoder,
    height=args.input_height,
    width=args.input_width,
    batch_size=2,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)

left = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
model = MonodepthModel(params, "test", left, True)

# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

# ROS Stuff
cv_bridge = CvBridge()
last_time = 0.0
depth_image_pub = None

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def initialize_network(params):
    """Test function."""

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

def run(data):
    global last_time, depth_image_pub
    if last_time + 0.75 > rospy.Time.now().to_sec():
        return
    last_time = rospy.Time.now().to_sec()

    print("loading image")
    start = time.clock()


    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "rgb8")
    except CvBridgeError as e:
        print(e)
        quit()

    input_image = np.asarray(cv_image)
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # RUN
    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    try:
        depth_image_pub.publish(cv_bridge.cv2_to_imgmsg(disp_pp, "32FC1"))
    except CvBridgeError as e:
        print (e)
        quit()

    plt.imshow(disp_pp, cmap='plasma')
    plt.pause(0.005)z




    # output_directory = os.path.dirname(args.image_path)
    # output_name = os.path.splitext(os.path.basename(args.image_path))[0]
    #
    # np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    # plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

    print('done!, took {} seconds' .format(time.clock() - start))

def main(_):
    global depth_image_pub
    initialize_network(params)
    rospy.init_node('monodepth_node')
    image_sub = rospy.Subscriber("camera/image_raw", Image, run)
    depth_image_pub = rospy.Publisher("depth", Image)
    rospy.spin()


if __name__ == '__main__':
    tf.app.run()
