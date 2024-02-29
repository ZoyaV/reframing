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
# from transformers import HfArgumentParser
# from dpo_experiment.training_arguments import ProcessingArguments
import statistics
import re
import random
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from training_arguments import ValidationArguments
import peft
from dpo_experiment.utils.metrics import box_iou
from one_peace.models import from_pretrained
from dpo_experiment.utils.data import prepare_data, get_images
from dpo_experiment.utils.detector import get_Dino_predictions, get_ONE_PEACE_predictions

def init_detector_model():
    Dino = load_model("/home/misha/anaconda3/envs/dpo/lib/python3.7/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",\
                       "/home/misha/code/cunning_manipulator/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return Dino

def init_onepeace():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_pretrained(
	    "ONE-PEACE_Grounding",
        model_type="one_peace_classify",
        device=device,
        dtype="float32"
        )
    return model

def main():
    parser = HfArgumentParser(ValidationArguments)
    val_args = parser.parse_args_into_dataclasses()[0]
    model_type = val_args.model_type
    model_name = val_args.model_name
    path = val_args.path_to_source
    path_to_imgs = val_args.path_to_imgs
    path_to_output = val_args.path_to_output
    v_range = val_args.v_range
    if model_type == "none":
        print("model=None")
    elif model_type == "tuned":
        model  = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",  #NousResearch/Llama-2-7b-chat-hf
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=False,
        )
        model = peft.PeftModel.from_pretrained(model,'./results/checkpoint-10', torch_dtype=torch.float16)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-10")
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type=="pretrained": 
        model  = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf",  #NousResearch/Llama-2-7b-chat-hf
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        load_in_4bit=False,
        )
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
        tokenizer.pad_token = tokenizer.eos_token
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
        if model_type == "none":
            output = data['prompt'][i].replace('Send ONLY a single sentence - a rewording of ', '')
        else: 
            prompt = tokenizer(data['prompt'][i],return_tensors="pt").input_ids
            output_tensor = model.generate(torch.Tensor(prompt),  max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
            output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)
        if model_name == "DINO":
            model = init_detector_model()
            device = torch.device("cuda" if torch.is_available() else "cpu")
            print(device)
            model.to(device)
        elif model_name == 'onepeace':
            model = init_onepeace()
        name, img_sources, images = get_images(data['item_id'][i], path_to_imgs)
        if model_name == 'DINO':
            predicted_bbox, pred_score = get_Dino_predictions(model, images, img_sources, output)
        elif model_name == 'onepeace':
            predicted_bbox = get_ONE_PEACE_predictions(model, path_to_imgs+name, str(output))[0]
            pred_score = 1
            try: 
                dataset_bbox = torch.Tensor([[float(x) for x in re.split(',', data['true_bbox'][i][1:-1])]])
                real_bbox = box_convert(boxes=dataset_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()[0] 
                iou_score = float(box_iou(real_bbox, predicted_bbox))
            except Exception as e:
                iou_score = 0.0
                pred_score = 0.0
                print(f"Exception: {e}")
        ious.append(iou_score)
        scores.append(pred_score)
    
    iou = np.mean(ious)
    iou_std = np.std(ious)
    score_std = np.std(scores)
    score = np.mean(scores)
    all_scores = []
    # all_scores["pred_score_median"] = score
    # all_scores["pred_score_std"] = score_std
    # all_scores["IOU_median"] = iou
    # all_scores["IOU_std"] = iou_std
    print(f"iou median = {iou}, iou std = {iou_std}, score = {score}, score std = {score_std}")
    old_table = pd.DataFrame(columns = ["name", "iou_mean", "iou_std", "score_mean", "score_std"])
    old_table.loc[len(old_table.index)] = ['onepeace_llama_notuning', iou, iou_std, score, score_std] 
    old_table.to_csv(path_to_output, index=False)
if __name__ == "__main__":
    from huggingface_hub import login

    login("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")

    main()
