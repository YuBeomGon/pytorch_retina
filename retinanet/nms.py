import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt

import cv2
import re
import time


def iou(box1, box2):
    """Compute the Intersection-Over-Union of two given boxes.
    Args:
    box1: array of 4 elements [cx, cy, width, height].
    box2: same as above
    Returns:
    iou: a float number in range [0, 1]. iou of the two boxes.
    """

    lr = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    if lr > 0:
        tb = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])
    if tb > 0:
        intersection = tb*lr
        union = box1[2]*box1[3]+box2[2]*box2[3]-intersection

        return intersection/union

    return 0

def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.
    Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
    Returns:
    ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

# def nms(boxes, probs, threshold):
#   """Non-Maximum supression.
#   Args:
#     boxes: array of [cx, cy, w, h] (center format)
#     probs: array of probabilities
#     threshold: two boxes are considered overlapping if their IOU is largher than
#         this threshold
#     form: 'center' or 'diagonal'
#   Returns:
#     keep: array of True or False.
#   """

# #   order = probs.argsort()[::-1]
#   keep = [True]*len(order)

#   for i in range(len(order)-1):
#     ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
#     for j, ov in enumerate(ovps):
#       if ov > threshold:
#         keep[order[j+i+1]] = False
#   return keep

# take the nms        
def get_nmsbox(bboxes, scores, isCenter=False, iou_threshold=0.5) : 
#     make xmin, ymin,  to centerx, centery    
    new_boxes = np.zeros((len(bboxes),4))  
    if isCenter == False :
        for i, box in enumerate(bboxes) :
            new_boxes[i, 0] = (box[0] + box[2])/2
            new_boxes[i, 1] = (box[1] + box[3])/2
            new_boxes[i, 2] = (box[2] - box[0])
            new_boxes[i, 3] = (box[3] + box[1])  
    else :
        for i, box in enumerate(bboxes) :
            new_boxes[i, 0] = box[0]
            new_boxes[i, 1] = box[1]
            new_boxes[i, 2] = box[2]
            new_boxes[i, 3] = box[3]
        
        
    keep = [True]*len(bboxes)
    for i in range(len(bboxes)-1):
        ovps = batch_iou(new_boxes[i+1:], new_boxes[i])
#         print(ovps)
        for j, ov in enumerate(ovps):
          if ov > iou_threshold:
            keep[j+i+1] = False   
            
    final_boxes = []
    final_scores = []
    for i, box, score in zip(keep, bboxes, scores) :
        if i == True :
            final_boxes.append(box) 
            final_scores.append(score)
    return final_boxes, final_scores