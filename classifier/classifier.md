## Traffic Detection and Classification

We used [Tensorflow Object Detecion](https://github.com/tensorflow/models/tree/master/research/object_detection) API for detecting and classifying the Red, Yellow and Green lights.

The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.
This repository includes different [models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) trained on COCO dataset over 90 different classes, which includes traffic light as well.

The pre-trained model is able to detect traffic lights but unfortunately cannot classify the Red, Yellow and Green lights. So to train model
for classification we chose **faster_rcnn_inception_v2_coco** from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 


### Collecting Training Data
The udacity's simulator provides images from car's camera on rostipic `/image_color`. We collected a set of images by subsribing to rostopic `/image_color` and running the car in the manual mode.

### Labeling and Generating Training Record

This official document [using_your_own_dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
was very helpful for creating the training record.

The [LabelImg](https://github.com/tzutalin/labelImg) is an exremely handy tool for labeling the images and generating the bounding boxes for the lights in the image. 
The tool saves the label and corresponding bounding box in a `.xml` file. 

This [Youtube tutorial](https://www.youtube.com/watch?v=HjiBbChYRDw) clearyly explains the procedure to train the Tensorflow Object Detection
API on custom dataset. In our case traffic lights.

After training for around 30000 steps we were able to achieve training loss as low as `0.04`
