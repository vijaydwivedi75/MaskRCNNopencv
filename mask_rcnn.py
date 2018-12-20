import cv2 as cv
import numpy as np

import os.path
import random

import sys
import argparse

# Confidence and mask threshold
conf_threshold = 0.4
mask_threshold = 0.3

# COME BACK HERE TO WRITE PARSER CODES
parser = argparse.ArgumentParser(description='Mask RCNN for instance segmentation')
parser.add_argument('--image', help='Image file path')
parser.add_argument('--video', help='Video file path')
args = parser.parse_args()

# Draw the mask and show on the image
def draw_box(frame, class_id, conf, left, top, right, bottom, class_mask):
    # Draw a bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)

    label = '%.2f' % conf
    if classes:
        assert (class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

        # Display label at the top of the bounding box
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 0.5, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top+base_line), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_PLAIN, 0.5, 1)

        class_mask = cv.resize(class_mask, (right - left + 1, bottom - top + 1))
        filtering_mask = (class_mask > mask_threshold)
        roi = frame[top:bottom+1, left:right+1][filtering_mask]
        
        color = colors[class_id % len(colors)]

        frame[top:bottom+1, left:right+1][filtering_mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7*roi).astype(np.uint8)

        # Draw the contours on the image
        filtering_mask = filtering_mask.astype(np.uint8)
        im2, contours, hierarchy = cv.findContours(filtering_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)


# For each detected object in a frame, extract bounding box and mask
def postprocess(boxes, masks):
    # Mask output size: N x C x H x W; 
    # where; N: No of detected boxes
    #        C: No of classes
    #        H x W: Segmentation shape

    num_classes = masks.shape[1]
    num_detections = boxes.shape[2]

    frame_H = frame.shape[0]
    frame_W = frame.shape[1]

    for i in range(num_detections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > conf_threshold:
            class_id = int(box[1])

            # Extract the bounding box
            left = int(frame_W * box[3])
            top = int(frame_H * box[4])
            right = int(frame_W * box[5])
            bottom = int(frame_H * box[6])

            left = max(0, min(left, frame_W-1))
            top = max(0, min(top, frame_H-1))
            right = max(0, min(right, frame_W-1))
            bottom = max(0, min(bottom, frame_H-1))

            # Extract the mask for the object
            class_mask = mask[class_id]

            draw_box(frame, class_id, score, left, top, right, bottom, class_mask)


# Load classes
classes_file = "mscoco_labels.names"
classes = None

with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load colors
colors_file = "colors.txt"
with open(colors_file, 'rt') as f:
    colors_str = f.read().rstrip('\n').split('\n')

colors = []
for i in range(len(colors_str)):
    rgb = colors_str[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

# Load text graph and weight files for the model
text_graph = './mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
model_weights = './mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'

# Load network
net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Naming the OpenCV Window
win_name = 'Mask-RCNN Instance Segmentation in OpenCV'
cv.namedWindow(win_name, cv.WINDOW_NORMAL)

output_file = '_mask_rcnn_out.avi'

if(args.image):
    if not os.path.isfile(args.image):
        print("Input image file: ", args.image, "doesn't exist!")
        sys.exit()
    cap = cv.VideoCapture(args.image)
    output_file = args.image[:-4] + '_mask_rcnn_out.jpg'
else:
    # If there is no input image given, take the webcam as the input
    cap = cv.VideoCapture(0)

# Initialize video writer to save output video
if (not args.image):
    vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M','J','P','G'), 28, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


# Process each fram
while cv.waitKey(1) < 0:
    
    # Get frame from the video
    has_frame, frame = cap.read()

    # If video ends, stop and exit from the program
    if not has_frame:
        print("Done processing!")
        print("Output file is stored as: ", output_file)
        cv.waitKey(3000)
        break

    # create a 4D blob from  a frame
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # set input to the network
    net.setInput(blob)

    # Run the forward pass computation to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Extract the bounding boxes and masks for each of the detected objects
    postprocess(boxes, masks)

    t, _ = net.getPerfProfile()
    label = 'Mask RCNN, Inference time for a frame: %0.0f ms' % abs(t*1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0))

    if(args.image):
        cv.imwrite(output_file, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(win_name, frame)