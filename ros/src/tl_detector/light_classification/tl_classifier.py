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
        
        rospy.loginfo("============ Classifier loaded! ============")

    def get_classification(self, image):

        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction

        start_time = time.time()
       
        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Expand image
        image_expanded = np.expand_dims(image, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (self.boxes, self.scores, self.classes, self.num_detections) = self.sess.run([detection_boxes, detection_scores, detection_classes, num_detections],\
            feed_dict={image_tensor: image_expanded})
        
        end_time = time.time()

        # Pick the light with high prediction score out of first 3 scores
        red_score = 0.
        green_score = 0.
        yellow_score = 0.

        for i in range (3):
            if self.classes[0][i] == 1:
                red_score = red_score + self.scores[0][i]

            if self.classes[0][i] == 2:
                yellow_score = yellow_score + self.scores[0][i]         

            if self.classes[0][i] == 3:
                green_score = green_score + self.scores[0][i]

        index = -1
        if red_score > yellow_score:
            index = 1
        elif yellow_score > green_score:
            index = 2
        else:
            index = 3

        if self.scores[0][0] < self.MIN_SCORE_THRESHOLD:
            index = 0

        if index == 1:
            light = TrafficLight.RED
            caption = 'Red: ' + str(self.scores[0][0] * 100)[:5] + '%'
        elif index == 2:
            light = TrafficLight.YELLOW
            caption = 'Yellow: ' + str(self.scores[0][0] * 100)[:5] + '%'
        elif index == 3:
            light = TrafficLight.GREEN
            caption = 'Green ' + str(self.scores[0][0] * 100)[:5] + '%'
        else:
            light = TrafficLight.UNKNOWN
            caption = 'UNKNOWN '

        rospy.loginfo("Light State: %s Inference Time: %.4f ", caption, (end_time - start_time))

        return light
