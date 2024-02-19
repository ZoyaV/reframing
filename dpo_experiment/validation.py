import numpy as np
import torch
from tqdm import tqdm
from trl import PPOTrainer, create_reference_model
import wandb
import pandas as pd
import random
import sys
sys.path.append("../")
sys.path.append("../GroundingDINO/")
import matplotlib.pyplot as plt
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import peft
from groundingdino.util.inference import load_model, load_image, predict
import cv2
from datasets import Dataset, load_dataset
from torchvision.ops import box_convert
from utils import box_iou
tqdm.pandas()
wandb.init(dir="./wandb")
def Dino_detector_based_reward(logits, labels, model, images, img_sources):
    reward_metrics = []
    all_scores = {}
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)

        if len(prediction) == 0 or prediction == "word":
            prediction = "object"
        else:
            prediction = prediction.replace("or ", "")
            prediction = prediction.replace("of ", "")
            prediction = prediction.replace("to ", "").strip()

        print(prediction)
        real_bbox = []
        try:
            boxes, logits_detector, phrases = predict(
                 model=model,
                 image=images,
                 caption=prediction,
                 box_threshold=0,
                 text_threshold=0.25
             )
            h, w, _ = img_sources.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            max_pos = np.where(np.max(logits_detector.numpy()))
            predicted_bbox = box_convert(boxes=boxes[max_pos], in_fmt="cxcywh", out_fmt="xyxy").numpy()[max_pos][0]
            true_bbox =  torch.Tensor([float(x) for x in re.split(',', labels[1:-1])])
            real_bbox = box_convert(boxes=true_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()
            iou_score = float(box_iou(real_bbox, predicted_bbox))
            pred_score = float(logits_detector.numpy()[max_pos])
            total_reward = [pred_score, iou_score]
            reward_metrics.append(total_reward)

        except Exception as e:
            print(e)
            total_reward = [0.0,0.0]
            reward_metrics.append(total_reward)

    print(f"Reward metrics: {reward_metrics}")
    return total_reward
def get_images(paths):
    imgs = []
    img_sources = []
    for path in paths:
        img_source, image = load_image(path)
        imgs.append(image)
        img_sources.append(img_source)
    return imgs, img_sources
def prepare_data(path, v_range="train"):
    if v_range == "train":
        data = pd.read_csv(path, header=0)[:100]
    elif v_range == "test":
        data = pd.read_csv("../dataset/dpo_final_dataset.csv", header=0)
        data = data[len(data)-100:].reset_index(drop=True)
    print(data.size)
    data = data.dropna()
    print(data.size)
    return data
def main():
    model_type = "pretrained"
    path = "./dpo_final_dataset.csv"
    model  = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",  #NousResearch/Llama-2-7b-chat-hf
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=False,
    )
    if model_type == "pretrained":
        model = peft.PeftModel.from_pretrained(model,'./results/checkpoint-10', torch_dtype=torch.float16)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-10")
    else: 
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    Dino = load_model("/home/AI/yudin.da/avshalumov_ms/.local/lib/python3.8/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    v_range = "test"
    if v_range == "train":
        data = pd.read_csv(path, header=0)[:100]
    elif v_range == "test":
        data = pd.read_csv("./dpo_final_dataset.csv", header=0)
        data = data[len(data)-100:].reset_index(drop=True)
    print(data.size)
    data = data.dropna()
    ious = []
    scores = []
    for i in range(len(data)):
        prompt = tokenizer(data['prompt'][i],return_tensors="pt").input_ids
        output_tensor = model.generate(torch.Tensor(prompt),  max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
        obj = data['item_id'][i]
        obj_split = obj.split('_')
        obj_split_len = len(obj_split)
        if obj_split_len == 3:
            obj_name = obj_split[0]
        elif obj_split_len == 4:
            obj_name = obj_split[0] + '_' + obj_split[1]
        elif obj_split_len == 5:
            obj_name = obj_split[0] + '_' + obj_split[1] + '_' + obj_split[2]
        name = obj_name+'/'+obj_name+'_'+obj_split[len(obj_split)-2]+'/'+obj+'.png'
        img_sources, images = load_image('./dataset/gold/images/RGB_raw/'+name)
        rewards = Dino_detector_based_reward(output, data['true_bbox'][i], Dino, images, img_sources)
        ious.append(rewards[1])
        scores.append(rewards[0])
    
    iou = np.median(ious)
    iou_std = np.std(ious)
    score_std = np.std(scores)
    score = np.mean(scores)
    all_scores = []
    # all_scores["pred_score_median"] = score
    # all_scores["pred_score_std"] = score_std
    # all_scores["IOU_median"] = iou
    # all_scores["IOU_std"] = iou_std
    print(f"iou median = {iou}, iou std = {iou_std}, score = {score}, score std = {score_std}")
    old_table = pd.read_csv("./dpo_results_table_test.csv")
    old_table.loc[len(old_table.index)] = ['llama2 tuned 5000steps new', iou, iou_std, score, score_std] 
    old_table.to_csv("./dpo_results_table.csv", index=False)
if __name__ == "__main__":
    from huggingface_hub import login

    login("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")

    main()
