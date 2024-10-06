#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import tensorflow as tf
import time
from ultralytics import YOLO

tf.config.experimental.set_visible_devices([], 'GPU')
model = YOLO("/home/rania/test/bestcone.pt")

class_names = {
    1: "Human",
    3: "Car",
}

def load_model(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

def process_frame(sess, image, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
    image_np_expanded = np.expand_dims(image, axis=0)
    start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    end_time = time.time()

    print("Elapsed Time:", end_time-start_time)

    im_height, im_width, _ = image.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[0,i,0] * im_height),
                    int(boxes[0,i,1]*im_width),
                    int(boxes[0,i,2] * im_height),
                    int(boxes[0,i,3]*im_width))

    return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    resized_image = cv2.resize(cv_image, (1280, 720))
    res = model(resized_image)
    results_object = res[0]
    boxes2 = results_object.boxes.xyxy
    if(len(results_object)!=0):
        for i in range(len(results_object)):
            confidence = float(results_object.boxes.conf[i].item())
            x_min, y_min, x_max, y_max = map(int, boxes2[i])
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"Cone {confidence:.2f}"  # Format score with 2 decimal places
            cv2.putText(resized_image, text, (x_min, y_min+10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    boxes, scores, classes, num = process_frame(sess, resized_image, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)

    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            class_name = class_names[classes[i]]
            cv2.rectangle(resized_image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            confidence_score = (scores[i])
            cv2.putText(resized_image, f"{class_name}  {confidence_score:.2f}", (box[1], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        if classes[i] == 3 and scores[i] > threshold:
            box = boxes[i]
            class_name = class_names[classes[i]]
            cv2.rectangle(resized_image, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            confidence_score = (scores[i])
            cv2.putText(resized_image, f"{class_name}  {confidence_score:.2f}", (box[1], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Video", resized_image)
    cv2.waitKey(1)

def main():
    # Initialize ROS node
    rospy.init_node("video_viewer", anonymous=True)

    # Load the detection model
    model_path = r'/home/rania/SVU_RT/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    
    global sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, threshold
    sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = load_model(model_path)
    threshold = 0.7

    # Subscribe to the camera topic
    rospy.Subscriber("/image", Image, image_callback)

    # Spin ROS node
    rospy.spin()

    # Close OpenCV window when done
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


