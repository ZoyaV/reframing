# 0. imports
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

c=0

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
        self.detector=detector_model
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
            for i in range(50):
                prompt = tokenizer(data['prompt'][i],return_tensors="pt").input_ids
                output_tensor = model.generate(input_ids=prompt, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
                output = tokenizer.batch_decode(output_tensor, skip_special_tokens=True)[0]
                print("OUTPUT OF LM = ", output)
                name, img_sources, images = get_images(data['item_id'][i])
                predicted_bbox, pred_score = get_Dino_predictions(self.detector, images, img_sources, output)
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
                        image = plt.imread("/datasets/gold/images/RGB_raw/"+name)
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

@record
def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # Create the custom configuration
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

# Instantiate Accelerator with the custom configuration
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs],log_with="wandb")
    Dino = load_model("/home/AI/yudin.da/avshalumov_ms/.local/lib/python3.8/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
    if accelerator.is_main_process:
        accelerator.init_trackers("dpo_llama2", init_kwargs={
            "wandb": {
                "name": "dpo_llama2_triplet_seed_"+str(script_args.seed)
            }
        })
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit = True,
    )
    model.config.use_cache = False
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    accelerator.prepare(model)
    # model_ref  = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit = True,
    # )
    # accelerator.prepare(model_ref)
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    data = prepare_data("dpo_final_dataset_triplet.csv")

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        #per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        #evaluation_strategy="epoch",
        num_train_epochs=15,
        seed=script_args.seed,
        #eval_steps=script_args.eval_steps,
        do_eval=False,
        logging_strategy="epoch",
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        fp16=True,
        remove_unused_columns=False,
        run_name="dpo_llama2_triplet_" + str(script_args.seed),
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        #generate_during_eval=True,
        beta=script_args.beta,
        train_dataset=data['train'],
        #eval_dataset=data['test'],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        #compute_metrics=compute_metrics
    )
    val_callback = ValidationCallback(
    val_dataset=data['test'],
    accelerator=accelerator,
    detector_model=Dino,
    eval_step=script_args.save_steps,
    seed=script_args.seed
    )

    # Add the callback to the trainer
    dpo_trainer.add_callback(val_callback)

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    from huggingface_hub import login

    login("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")

    main()

