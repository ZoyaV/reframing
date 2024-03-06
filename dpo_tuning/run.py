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
import re
from training_arguments import ScriptArguments


from utils.data import prepare_data


from torch.distributed.elastic.multiprocessing.errors import record
import os


@record
def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    # Create the custom configuration
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours

# Instantiate Accelerator with the custom configuration
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs],log_with="wandb")
    if accelerator.is_main_process:
        accelerator.init_trackers(str(script_args.run_name), init_kwargs={
            "wandb": {
                'name':str(script_args.run_name)+"_seed_"+str(script_args.seed)
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
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(script_args.path_to_source)
    # 2. Load the Stack-exchange paired dataset
    data = prepare_data(script_args.path_to_source)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=16,#script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        num_train_epochs=15,
        seed=script_args.seed,
        eval_steps=500,
        do_eval=False,
        logging_strategy="steps",
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        fp16=True,
        remove_unused_columns=False,
        run_name=script_args.run_name + str(script_args.seed),
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
        eval_dataset=data['test'],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        #compute_metrics=compute_metrics
    )
    # val_callback = ValidationCallback(
    # val_dataset=data['test'],
    # accelerator=accelerator,
    # detector_model=Dino,
    # eval_step=script_args.save_steps,
    # seed=script_args.seed
    # )

    # # Add the callback to the trainer
    # dpo_trainer.add_callback(val_callback)

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

