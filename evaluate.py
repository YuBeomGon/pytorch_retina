import argparse
import collections

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
import albumentations as A
import albumentations.pytorch
from tqdm.notebook import tqdm

import cv2
import re
import time

from retinanet import model
# from retinanet import retina
from retinanet.dataloader import *
from retinanet.anchors import Anchors
# from scheduler import *

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import Adam, lr_scheduler
import torch.optim as optim

from pycocotools.cocoeval import COCOeval
import json
import torch

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple paps training script for training a RetinaNet network.')
    parser.add_argument('--batch_size', help='Number of batchs', type=int, default=0)    
    parser.add_argument('--test_data', help='test data file', default='data/test.npy')   
    parser.add_argument('--model_dir', help='pretrained model dir', default='trained_models/resnet50_640/model.pt') 
    parser.add_argument('--threshold', help='pretrained model dir', type=float, default=0.1) 
    
    parser = parser.parse_args(args)
    
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('device', device)    

    retinanet = model.resnet50(num_classes=2, device=device)
    retinanet = torch.nn.DataParallel(retinanet, device_ids = [GPU_NUM], output_device=GPU_NUM).to(device)
    retinanet.load_state_dict(torch.load(parser.model_dir))    
#     retinanet.to(device)
    
    dataset_val = PapsDataset('data/', set_name='val_2class',
                                transform=val_transforms)

    val_data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )    

    retinanet.eval()
    start_time = time.time()
    threshold = parser.threshold
    results = []
    GT_results = []
    image_ids = []
    cnt = 0

    for index, data in enumerate(tqdm(val_data_loader)) :
        if cnt > 100 :
            break
        cnt += 1
        with torch.no_grad():        
            images, tbox, tlabel, targets = data
            batch_size = len(images)
    #         print(tbox)
    #         print(len(tbox[0]))

            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)

            outputs = retinanet(images)
            scores, labels, boxes = (outputs)

            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()  

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
    #             print(boxes)

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset_val.image_ids[index],
                        'category_id' : dataset_val.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            if len(tbox[0]) > 0:    

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(len(tbox[0])):
                    score = float(0.99)
                    label = (tlabel[0][box_id])
                    box = list(tbox[0][box_id])
                    box[2] -= box[0]
                    box[3] -= box[1]             

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset_val.image_ids[index],
                        'category_id' : dataset_val.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : list(box),
                    }

                    # append detection to results
                    GT_results.append(image_result)                

            # append image to list of processed images
            image_ids.append(dataset_val.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset_val)), end='\r')    

    if not len(results):
        print('No object detected')
    print('GT_results', len(GT_results))    
    print('pred_results', len(results))    

    # write output
    json.dump(results, open('trained_models/eval/{}_bbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)
    # write GT
    json.dump(GT_results, open('trained_models/eval/{}_GTbbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)     

    print('validation time :', time.time() - start_time)
    
    # load results in COCO evaluation tool
    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes('trained_models/eval/{}_bbox_results.json'.format(dataset_val.set_name))
    coco_gt = coco_true.loadRes('trained_models/eval/{}_GTbbox_results.json'.format(dataset_val.set_name))

    # run COCO evaluation
    # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    # coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()   
    
if __name__ == '__main__':
    main()        

    