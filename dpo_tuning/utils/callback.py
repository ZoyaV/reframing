from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import wandb
import pandas as pd
import random
import sys
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import numpy as np
from trl import DPOTrainer
sys.path.append("../")
sys.path.append("../../GroundingDINO/")
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import re
from training_arguments import ScriptArguments


from utils.metrics import box_iou
from utils.data import prepare_data, get_images
from utils.detector import get_Dino_predictions


from PIL import Image
import cv2
import matplotlib.pyplot as plt
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.integrations import WandbCallback
import os

class ValidationCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, val_dataset,accelerator, detector_model, eval_step, seed):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        self.val_dataset=val_dataset
        self.accelerator=accelerator
        self.step=eval_step
        if detector_model_name == "DINO": 
            self.detector = DinoDetector()
        elif detector_model_name == "onepeace":
            self.detector = OnePeaceDetector()
        self.seed=seed
        super().__init__()
        
    def on_save(self, args, state, control, **kwargs):
        super().on_save(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
            # generate predictions
        c = state.global_step-self.step
        print("EVALUATING EVALUATING EVALUATING")
        if c==0:
            return
        else:
            path="./results/checkpoint-"+str(c)
            model = AutoModelForCausalLM.from_pretrained(
            "NousResearch/Llama-2-7b-chat-hf",  #NousResearch/Llama-2-7b-chat-hf
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            load_in_4bit=False,
            )
            model = PeftModel.from_pretrained(model,path, torch_dtype=torch.float16)
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(path)
            ious = []
            scores = []
            data=self.val_dataset
            path_to_imgs="/datasets/gold/images/RGB_raw/"
            for i in range(50):
                prompt = tokenizer(data['prompt'][i],return_tensors="pt").input_ids
                output_tensor = model.generate(input_ids=prompt, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
                output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)[0]
                print("OUTPUT OF LM = ", output)
                name, img_sources, images = get_images(data['item_id'][i], path_to_imgs)
                image_metadata = {'image_np': images, 'image_pil': img_sources, 'output': output, 'image_path': path_to_imgs+name}
                predicted_bbox, pred_score = self.detector.predict(image_metadata)
                true_bbox =  torch.Tensor([float(x) for x in re.split(',', data["true_bbox"][i][1:-1])])
                real_bbox = box_convert(boxes=true_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()
                iou_score = float(box_iou(real_bbox, predicted_bbox))
                print(predicted_bbox)
                print(f"IOU = {iou_score}, score = {pred_score}")
                ious.append(iou_score)
                scores.append(pred_score)
                if i==49:
                    with open('./input_output_examples.txt', 'a') as f:
                        f.write("{} ||| {} ||| {} ||| {} ||| {}\n".format(c, data['prompt'][i],output, iou_score, pred_score))
                        image = plt.imread(path_to_imgs+name)
                        image = cv2.rectangle(image, (int(predicted_bbox[0]), int(predicted_bbox[1])), (int(predicted_bbox[2]), int(predicted_bbox[3])), (0, 0, 0), 2)
                        cv2.putText(image, 'predicted', (int(predicted_bbox[0]), int(predicted_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
                        image = cv2.rectangle(image, (int(real_bbox[0]), int(real_bbox[1])), (int(real_bbox[2]), int(real_bbox[3])), (36,255,12), 2)
                        cv2.putText(image, 'dataset', (int(real_bbox[0]), int(real_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        im = Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
                        im.save(f"./images/image_{c}.png")


            iou = np.mean(ious)
            iou_std = np.std(ious)
            score_std = np.std(scores)
            score = np.mean(scores)
            all_scores = {}
            all_scores["metrics/pred_score_mean"] = score
            all_scores["metrics/pred_score_std"] = score_std
            all_scores["metrics/IOU_mean"] = iou
            all_scores["metrics/IOU_std"] = iou_std
            print(all_scores)
            self.accelerator.log(all_scores)
