#!/usr/bin/python

#
# Copyright 2018-2019 Autoware Foundation
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
#
#  v1.0 Abraham Monrroy
#

#ros
import sys
import numpy as np
import rospy
import cv2
import time

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError

#deeplab
import os
import tarfile
import argparse

from PIL import Image as PILImage

import roslib
roslib.load_manifest('vision_deeplab_segment')

import tensorflow as tf

class AutowareDeeplab():
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
    out_topic = '/image_segmented'
    __APP_NAME__ = "vision_deeplab_segment"

    PASCAL_LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    CITYSCAPES_LABEL_NAMES = np.asarray([
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
        'bus', 'train', 'motorcycle', 'bicycle'
    ])

    def __init__(self, tarball_path, image_topic, label_format):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        if label_format == "cityscapes":
            LABEL_NAMES =self. CITYSCAPES_LABEL_NAMES
        else:
            LABEL_NAMES = self.PASCAL_LABEL_NAMES

        self.FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
        self.colormap = self.create_label_colormap()
        self.FULL_COLOR_MAP = self.label_to_color_image(self.FULL_LABEL_MAP)
        self.original_width = 0
        self.original_height = 0

        graph_def = None

        rospy.loginfo('[%s] Loading specified pretrained model...', self.__APP_NAME__)
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(image_topic, ROSImage, self.callback, queue_size=1)
        self.segmentation_pub = rospy.Publisher(self.out_topic, ROSImage, queue_size=1)

        rospy.loginfo('[%s] Subscribing to %s', self.__APP_NAME__, image_topic)
        rospy.loginfo('[%s] Publishing segmented image to %s', self.__APP_NAME__, self.out_topic)
        rospy.loginfo('[%s] Ready. Waiting for data...', self.__APP_NAME__)

    def run(self, pil_image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = pil_image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = pil_image.convert('RGB').resize(target_size, PILImage.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        self.original_width = width
        self.original_height = height
        return resized_image, seg_map


    def create_label_colormap(self):
        """Creates a label colormap compatible with PASCAL VOC and Cityiscape segmentation benchmarks.

        Returns:
          A Colormap for visualizing segmentation results.
        """
        colormap = np.asarray([
            [128, 64, 128],
            [232, 35, 244],
            [70, 70, 70],
            [156, 102, 102],
            [153, 153, 190],
            [153, 153, 153],
            [30, 170, 250],
            [0, 220, 220],
            [35, 142, 107],
            [152, 251, 152],
            [180, 130, 70],
            [60, 20, 220],
            [0, 0, 255],
            [142, 0, 0],
            [70, 0, 0],
            [100, 60, 0],
            [100, 80, 0],
            [0, 0, 230],
            [32, 11, 119],
            [255, 128, 0],
            [255, 153, 255]
        ])

        return colormap


    def label_to_color_image(self, label):
        """Adds color defined by the dataset colormap to the label.

        Args:
          label: A 2D array with integer type, storing the segmentation label.

        Returns:
          result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

        Raises:
          ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        if np.max(label) >= len(self.colormap):
            raise ValueError('label value too large.')

        return self.colormap[label]

    def pad_image(self, cv_image):
        cv_image_height, cv_image_width, cv_image_depth = cv_image.shape

        padded_image = np.zeros([1025,2019,3],dtype=np.uint8)

        resize_ratio_y = 0.0
        resize_ratio_x = 0.0
        if cv_image_height > 1025:
            resize_ratio_y = 1025.0 / float(cv_image_height)
        if cv_image_width > 2019:
            resize_ratio_x = 2019.0 / float(cv_image_width)
        resize_ratio = max(resize_ratio_y, resize_ratio_y)
        if resize_ratio != 0.0:
            cv_image = cv2.resize(cv_image, None, fx=resize_ratio, fy=resize_ratio)

        cv_image_height, cv_image_width, cv_image_depth = cv_image.shape
        padded_image_height, padded_image_width, padded_image_depth = cv_image.shape
        offset_x = (padded_image_width - cv_image_width) / 2
        offset_y = (padded_image_height - cv_image_height) / 2

        padded_image[offset_y:offset_y + cv_image_height, offset_x:offset_x + cv_image_width] = cv_image
        return padded_image

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            #padded_image = pad_image(cv_image)
        except CvBridgeError as e:
            print(e)

        #t = time.time()

        pil_image = PILImage.fromarray(cv_image)
        resized_im, seg_map = self.run(pil_image)
        seg_image = self.label_to_color_image(seg_map).astype(np.uint8)

        #dt = time.time() - t
        #print("inference time: ", dt, " sec.")
        try:
            height, width = seg_image.shape[:2]
            scaled_image = cv2.resize(seg_image, (self.original_width, self.original_height))
            ros_image_msg = self.bridge.cv2_to_imgmsg(scaled_image, "bgr8")
            ros_image_msg.header = data.header
            self.segmentation_pub.publish(ros_image_msg)
        except CvBridgeError as e:
            print(e)




def main(args):
    rospy.init_node('vision_deeplab_segment', anonymous=True)

    parser = argparse.ArgumentParser(description="Autoware ROS node for Tensorflow's Deeplab v3 segmentation")
    parser.add_argument("model_path", help="Path to TAR file containing the pretrained model")
    parser.add_argument("--image_src", help="Image topic to subscribe", default="/image_raw")
    parser.add_argument("--label_format", help="Format of the Model: 'pascal' or 'cityscapes'", default="pascal")
    args = parser.parse_args()

    rospy.loginfo('[vision_deeplab_segment] Opening Deeplab pretrained model %s', args.model_path)
    rospy.loginfo('[vision_deeplab_segment] Image source topic (image_src): %s', args.image_src)
    rospy.loginfo('[vision_deeplab_segment] Label format (label_format): %s', args.label_format)

    deeplab = AutowareDeeplab(args.model_path, args.image_src, args.label_format)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main(sys.argv)