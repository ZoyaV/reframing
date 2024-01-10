import sys

import numpy as np
import torch
from tqdm import tqdm
sys.path.append("../")
sys.path.append("../GroundingDINO/")
from groundingdino.util.inference import load_model, load_image, predict
import pandas as pd
from rewarding import Dino_detector_based_reward
import matplotlib.pyplot as plt
from metrics.metrics import box_iou

import statistics
import re
from torchvision.ops import box_convert
tqdm.pandas()

def init_detector_model():
    Dino = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return Dino
    
def main():
    # Initialize the tokenizer, model and reference model
    correct_reward_iou = []  
    correct_reward_score = []
    means = []
    model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv('./dataset/modified_output_concatenated_long_with_bboxes.csv', sep=',', header=0)
    
    for i in range(len(df)):
        correct = df['response'][i]
        obj = df['item_id'][i]
        obj_split = obj.split('_')
        obj_split_len = len(obj_split)
        if obj_split_len == 3:
            obj_name = obj_split[0]
        elif obj_split_len == 4:
            obj_name = obj_split[0] + '_' + obj_split[1]
        elif obj_split_len == 5:
            obj_name = obj_split[0] + '_' + obj_split[1] + '_' + obj_split[2]
        name = obj_name+'/'+obj_name+'_'+obj_split[len(obj_split)-2]+'/'+obj+'.png'
        img_source, img = load_image('./dataset/gold/images/RGB_raw/'+name)
        real_bbox = []
        try:
            boxes, logits_detector, phrases = predict(
                model=model,
                image=img,
                caption=correct,
                box_threshold=0,
                text_threshold=0.25
            )
            h, w, _ = img_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            max_pos = np.where(np.max(logits_detector.numpy()))
            predicted_bbox = box_convert(boxes=boxes[max_pos], in_fmt="cxcywh", out_fmt="xyxy").numpy()[max_pos][0]
            dataset_bbox = torch.Tensor([[float(x) for x in re.split(',', df['true_bbox'][i][1:-1])]])
            real_bbox = box_convert(boxes=dataset_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()[0]
            iou_score = float(box_iou(real_bbox, predicted_bbox))
            pred_score = float(logits_detector.numpy()[max_pos])
        except Exception as e:
            iou_score = 0.0
            pred_score = 0.0
            print(f"Exception: {e}")
        correct_reward_iou.append(iou_score)
        correct_reward_score.append(pred_score)
        means.append(statistics.harmonic_mean([iou_score, pred_score]))
        print(i, '\\', iou_score)
        

    df['response_iou'] = correct_reward_iou
    df['response_score'] = correct_reward_score
    df['harmonic_mean'] = means

    df.to_csv("./dataset/modified_output_concatenated_long_with_scores.csv")
# Generate text
if __name__ == "__main__":
    main()

