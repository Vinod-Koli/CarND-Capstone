from styx_msgs.msg import TrafficLight
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import rospy
import time

import label_map_util

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier

        self.detection_graph = None
        self.num_detections = None
        self.boxes = None
        self.scores = None
        self.classes = None

        self.label_map = None
        self.category_index = None

        self.MIN_SCORE_THRESHOLD = 0.40
        self.NUM_CLASSES = 3

        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        MODEL_NAME = 'faster_rcnn_v2_coco_light_classify'
        
        # path to model and label
        PATH_TO_CKPT = os.path.join(CWD_PATH, 'light_classification', MODEL_NAME,'frozen_inference_graph.pb')
        PATH_TO_LABELS = os.path.join(CWD_PATH,'light_classification', MODEL_NAME,'label_map.pbtxt')

        # Load the label map
        self.label_map      = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories          = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
        print("Done Loading classifier!")

    def get_classification(self, image):

        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction

        # Define input and output tensors (i.e. data) for the object detection classifier
        start = time.time()
        # Input tensor is the image
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (self.boxes, self.scores, self.classes, self.num_detections) = self.sess.run([detection_boxes, detection_scores, detection_classes, num_detections],\
            feed_dict={image_tensor: image_expanded})
        
        end = time.time()

        if self.classes[0][0] == 1:
            caption = 'Red: ' + str(self.scores[0][0] * 100)
        elif self.classes[0][0] == 2:
            caption = 'Yellow: '+ str(self.scores[0][0] * 100)
        elif self.classes[0][0] == 3:
            caption = 'Green '+ str(self.scores[0][0] * 100)
        else:
            caption = 'None'

        print("Light State: %s Time: %f ", caption, (end - start))

        return TrafficLight.UNKNOWN
