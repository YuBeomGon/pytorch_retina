from pycocotools.cocoeval import COCOeval
import json
import torch
import time
from tqdm import tqdm
import os

def evaluate_paps(dataset, dataloader, model, saved_dir, device, threshold=0.5):
    
    model.eval()
    start_time = time.time()
    results = []
    GT_results = []
    image_ids = []
    cnt = 0
    scores_list = []

    for index, data in enumerate(tqdm(dataloader)) :
        with torch.no_grad():        
            images, tbox, tlabel, targets = data
            batch_size = len(images)

            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)

            outputs = model(images)
            scores, labels, boxes = (outputs)

            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()  

            scores_list.append(scores)

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
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
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
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : list(box),
                    }

                    # append detection to results
                    GT_results.append(image_result)                

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')    

    if not len(results):
        print('No object detected')
    print('GT_results', len(GT_results))    
    print('pred_results', len(results))    

    # write output
    os.remove(saved_dir + '{}_bbox_results.json'.format(dataset.set_name))
    os.remove(saved_dir + '{}_GTbbox_results.json'.format(dataset.set_name))
    json.dump(results, open(saved_dir + '{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)
    # write GT
    json.dump(GT_results, open(saved_dir + '{}_GTbbox_results.json'.format(dataset.set_name), 'w'), indent=4)     

    print('validation time :', time.time() - start_time)


    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes(saved_dir + '{}_bbox_results.json'.format(dataset.set_name))
    coco_gt = coco_true.loadRes(saved_dir + '{}_GTbbox_results.json'.format(dataset.set_name))

    # run COCO evaluation
    # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    print("******************total*********************")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    # coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    
    print("******************Abnormal*********************")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()   
    
#     print("******************Normal*********************")
#     coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
#     coco_eval.params.imgIds = image_ids
#     coco_eval.params.catIds = [1]
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()  

    model.train()