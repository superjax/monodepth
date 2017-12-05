import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
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

class DNN_mono:
    def __init__(self):
        self.height = rospy.get_param("height", 480)
        self.width = rospy.get_param("width", 640)
        encoder = rospy.get_param("encoder", 'vgg')

        self.params = monodepth_parameters(
            encoder=encoder,
            height=self.height,
            width=self.width,
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

        # Initialize the neural network
        self.left_image = tf.placeholder(tf.float32, [2, self.height, self.width, 3])
        self.network = MonodepthModel(self.params, "test", self.left_image, None)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.coordinator = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coordinator)

        # create the image message handling
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.image_callback)

    def post_process_disparity(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0, :, :]
        r_disp = np.fliplr(disp[1, :, :])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        if cols != self.width or rows != self.height or channels != 3:
            raise StandardError, "Incorrect Image Size"

        np_image = np.asarray(cv_image)

        input_image = np_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = self.sess.run(self.network.disp_left_est[0], feed_dict={self.left_image: input_images})
        disp_pp = self.post_process_disparity(disp.squeeze()).astype(np.float32)

        disparity_cv = cv2.fromarray(disp_pp)

        cv2.imshow("Image", disparity_cv)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('depth_from_mono')
    dnn = DNN_mono()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()






