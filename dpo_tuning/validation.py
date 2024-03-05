import sys
sys.path.append("../")
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

os.system('pwd')
import numpy as np
import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict
import pandas as pd
import statistics
import re
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from training_arguments import ValidationArguments
import peft
from dpo_tuning.utils.metrics import box_iou
from one_peace.models import from_pretrained
from dpo_tuning.utils.data import prepare_data
from dpo_tuning.utils.detector import get_images, annotate_and_save
from detectors.detectors import BaseDetector

def get_predictions(i, data, prompt_name, detector, path_to_imgs, model=None, tokenizer=None):
        if not model:
            output = data['prompt'][i].replace(prompt_name, '')
        else: 
            prompt = tokenizer(data['prompt'][i],return_tensors="pt").input_ids
            output_tensor = model.generate(torch.Tensor(prompt),  max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
            output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
            output = str(output[0]).replace(str(prompt), '')       
            print(output)
        dataset_bbox = torch.Tensor([[float(x) for x in re.split(',', data['true_bbox'][i][1:-1])]])
        real_bbox = box_convert(boxes=dataset_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()[0] 
        name, img_sources, images = get_images(data['item_id'][i], path_to_imgs)
        image_metadata = {'image_np': images, 'image_pil': img_sources, 'correct': output, 'image_path': path_to_imgs+name}
        predicted_bbox, pred_score = detector.predict(image_metadata)
        print( predicted_bbox, pred_score, output)
        return real_bbox, predicted_bbox, pred_score, output

def main():
    parser = HfArgumentParser(ValidationArguments)
    val_args = parser.parse_args_into_dataclasses()[0]
    language_model_type = val_args.language_model_type
    language_model_name = val_args.language_model_name
    detector_model_name = val_args.detector_model_name
    path_to_source = val_args.path_to_source
    path_to_imgs = val_args.path_to_imgs
    path_to_output = val_args.path_to_output
    path_to_checkpoint = val_args.path_to_checkpoint
    prompt = val_args.prompt
    v_range = val_args.v_range
    run_name = val_args.run_name

    if language_model_type == "none":
        model=None
        tokenizer=None
    elif language_model_type == "tuned":
        model  = AutoModelForCausalLM.from_pretrained(
        language_model_name,  #NousResearch/Llama-2-7b-chat-hf
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=False,
        )
        model = peft.PeftModel.from_pretrained(model,path_to_checkpoint, torch_dtype=torch.float16)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
    elif language_model_type=="pretrained": 
        model  = AutoModelForCausalLM.from_pretrained(
        language_model_name,  #NousResearch/Llama-2-7b-chat-hf
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        tokenizer.pad_token = tokenizer.eos_token

    if detector_model_name == "Dino": 
        detector = BaseDetector("Dino")
    elif detector_model_name == "OnePeace":
        detector = BaseDetector("OnePeace")


    if v_range == "train":
        data = pd.read_csv(path_to_source, header=0)[:100]
    elif v_range == "test":
        data = pd.read_csv(path_to_source, header=0)
        data = data[len(data)-100:].reset_index(drop=True)
    print(data.size)
    data = data.dropna()
    ious = []
    scores = []
    for i in range(len(data)):
        real_bbox, predicted_bbox, pred_score, output = get_predictions(i, data, prompt, detector, path_to_imgs, model, tokenizer)
        if i==49:
            split = re.split(' ', re.split(',', data['description_bbox'][i][1:-1])[0])
            prompt_bbox = torch.Tensor([float(x) for x in split[0:len(split)-1]])
            annotate_and_save(i, predicted_bbox, real_bbox, prompt_bbox, path_to_output, path_to_imgs, name, run_name)
        iou_score = float(box_iou(real_bbox, predicted_bbox))
        ious.append(iou_score)
        scores.append(pred_score)
    
    iou = np.mean(ious)
    iou_std = np.std(ious)
    score_std = np.std(scores)
    score = np.mean(scores)
    print(f"iou median = {iou}, iou std = {iou_std}, score = {score}, score std = {score_std}")
    if os.path.exists(path_to_output):
        old_table = pd.read_csv(path_to_output)
    else:
        old_table = pd.DataFrame(columns = ["name", "iou_mean", "iou_std", "score_mean", "score_std"])
        
    old_table.loc[len(old_table.index)] = [run_name, iou, iou_std, score, score_std] 
    old_table.to_csv(path_to_output, index=False)
if __name__ == "__main__":
    from huggingface_hub import login

    login("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")

    main()
