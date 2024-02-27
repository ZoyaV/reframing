from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from transformers import GenerationConfig

import torch
import pandas as pd
import sys
import tqdm

sys.path.append(".")
from data_processing.options import *
from data_processing.utils import llama_model_and_tockenizer, mixtral_model_and_tokenizer

def count_sentences_advanced(text):

    points_in_text = text.count(".")
    bad_points_in_text = 0
    bad_points_in_text += text.count("?")
    bad_points_in_text += text.count("!")
    bad_points_in_text += text.count(":")
    bad_points_in_text += text.count(";")
    
    return points_in_text<=1 and bad_points_in_text==0

    
def generate(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, generation_config = GenerationConfig(do_sample = True, temperature= 1))
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).split(" [/INST] ")[1].split(".")[0]
    try:
        result = result.split(":")[1]
    except:
        result = result
    return result, count_sentences_advanced(result)

def sample_santences(row_dict, sample_count, prompt, model, tokenizer):
    result = {key:[row_dict[key]] for key in row_dict}
    result['score'] = [0]
    sentence = row_dict['description']
    for i in range(sample_count):
        sentence, correct = generate(prompt.format(sentence), model, tokenizer)
        score = 0 if correct else -1
        result['description'].append(sentence)
        result['score'].append(score)
        for key in row_dict:
            if key not in ['description', 'score']:
                result[key].append(row_dict[key])
    return result

if __name__ == "__main__":
    data = pd.read_csv("./datasets/gold_descriptions_with_bboxes.csv")
    data = data[['description','item_id', 'true_bbox']]
    
    model, tokenizer = llama_model_and_tockenizer()
    prompt = "Paraphrase sentence: {} [/INST]" 
    dfs = []
    for i in tqdm.tqdm(range(data.shape[0])):
        sentences = sample_santences(dict(data.iloc[i]), 5, prompt, model, tokenizer)
        sentences = pd.DataFrame(sentences)
        dfs.append(sentences)
        if i%10 == 0:
            result = pd.concat(dfs, ignore_index=True)
            result.to_csv(f"./datasets/augmented_gold_{i}.csv")
    result = pd.concat(dfs, ignore_index=True)
    result.to_csv(f"./datasets/augmented_gold.csv")
    