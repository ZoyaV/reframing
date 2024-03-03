from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import torch
import sys

import pandas as pd
from options import *
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
import tqdm
from transformers import GenerationConfig

def mixtral_model_and_tokenizer():
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
    return model, tokenizer
    
def llama_model_and_tockenizer():
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    return model, tokenizer

def generate(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=80, generation_config = GenerationConfig(do_sample = True, temperature= 1))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(data_path, model, tokenizer):
    data = pd.read_csv(data_path)
    original_prompts = data['preferenced']
    new_data = {'rejected':[], 'preferenced':[]}
    for i in range(3):
        new_data['preferenced'] += original_prompts.values.tolist()
        for i in tqdm.tqdm(range(original_prompts.size)):
            rewrited = generate(f"Send ONLY a single sentence - a rewording of '{original_prompts.iloc[i]}' [/INST] ", model, tokenizer).split(" [/INST] ")[1].split(".")[0]
            new_data['rejected'].append(rewrited)
    print (pd.DataFrame(new_data).head())
    return pd.DataFrame(new_data)

if __name__ == "__main__": 
    tockenizer, model = load_and_tockenizer()
    new_data = main("../dpo_experiment/modified_output.csv",model, tockenizer)
    new_data.to_csv("../dpo_experiment/modified_output2.csv")













