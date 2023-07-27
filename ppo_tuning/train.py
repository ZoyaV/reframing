import sys

import numpy as np
import torch
from tqdm import tqdm
from trl import AutoModelForSeq2SeqLMWithValueHead
from trl import PPOTrainer, create_reference_model

sys.path.append("../")
from detectors.owlvit import OwlViTDetector
from data_loaders import SegmentationsDataset, HFDataset
from options import (TXT_IN_LEN, TXT_OUT_LEN, MODEL_NAME, PRETRAINED_MODEL,
                     config, generation_kwargs, INPUT, OUTPUT, PROMPTS, REWARD_MODEL)
from rewarding import detector_based_reward, hf_based_reward
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
tqdm.pandas()


# Collate function to be used for gathering batch data
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def get_images(paths):
    imgs = []
    for path in paths:
        img = plt.imread(f"../dataset/imgs/{path}")
        imgs.append(img)
    return imgs
# Function to run an epoch of training with PPO loss
def run_epoch(ppo_trainer, tokenizer, batch, model, reward_model = "detector", reward_tokenizer = None):
    # Prepare empty logs and game data
    logs, game_data = {}, {}

    # Get the queries from the batch
    game_data["query"] = batch["query"]

    # Prepare the queries as tensors
    query_tensors = [torch.from_numpy(np.array(input_ids)).cuda() for input_ids in batch["input_ids"]]

    # Generate responses from the model
    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-TXT_OUT_LEN:])
    game_data["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]


    # Calculate rewards
    if reward_model == "detector":
        images = get_images(batch['file_name'])
        rewards = [torch.from_numpy(np.array([r])) for r in
                   detector_based_reward(game_data["response"], batch[OUTPUT], model, images)]
    elif reward_model == "hf":
        prompt = batch["query"]
        rewards = [torch.from_numpy(np.array([r])) for r in
                   hf_based_reward( batch[OUTPUT], model, reward_tokenizer, prompt)]

    # Perform a PPO training step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Log the stats
    ppo_trainer.log_stats(stats, game_data, rewards)


def init_detector_model():
    return OwlViTDetector("google/owlvit-base-patch32")

def init_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("./human_feedback/reward_model/checkpoint-1000")
    model = AutoModelForSequenceClassification.from_pretrained("./human_feedback/reward_model/checkpoint-1000")
    return model, tokenizer
# Main function to execute the training
def main():
    # Initialize the tokenizer, model and reference model
    if REWARD_MODEL == 'detector':
        reward_model = init_detector_model()
    elif REWARD_MODEL == 'hf':
        reward_model, reward_tokenizer = init_hf_model()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(PRETRAINED_MODEL)
    model_ref = create_reference_model(model)
    model.cuda()
    tokenizer.pad_token = tokenizer.eos_token


    # Update the pad_token_id in generation kwargs
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    if REWARD_MODEL == 'detector':
        # Initialize the dataset and the PPO Trainer
        dataset = SegmentationsDataset(tokenizer=tokenizer, txt_in_len=TXT_IN_LEN,
                                       inp_column=INPUT, out_column=OUTPUT, prompts=PROMPTS).prepare_dataset()
    else:
        dataset = HFDataset(tokenizer=tokenizer, txt_in_len=TXT_IN_LEN,
                                       inp_column=INPUT, out_column=OUTPUT).prepare_dataset()
   # print(dataset)
    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset, data_collator=collator)

    # Execute training for several epochs
    for epoch in range(2):
        for batch in tqdm(ppo_trainer.dataloader):
            run_epoch(ppo_trainer, tokenizer, batch, model, reward_model = REWARD_MODEL, reward_tokenizer=reward_tokenizer)

    path_to_save = f"checkpoint/{MODEL_NAME.split('/')[1]}"
    model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)

if __name__ == "__main__":
    main()
