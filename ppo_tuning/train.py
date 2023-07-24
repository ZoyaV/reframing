import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead
from transformers import AutoModelForSeq2SeqLM
from trl import PPOTrainer, create_reference_model

from data_loaders import IgluDataset
from options import TXT_IN_LEN, TXT_OUT_LEN, MODEL_NAME, PRETRAINED_MODEL, config, generation_kwargs, INPUT, OUTPUT
from rewarding import reinforce_loss

tqdm.pandas()


# Collate function to be used for gathering batch data
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# Function to run an epoch of training with PPO loss
def run_epoch(ppo_trainer, tokenizer, batch):
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
    rewards = [torch.from_numpy(np.array([r])) for r in reinforce_loss(game_data["response"], batch["builded"])]

    # Perform a PPO training step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Log the stats
    ppo_trainer.log_stats(stats, game_data, rewards)

# Main function to execute the training
def main():
    # Initialize the tokenizer, model and reference model


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(PRETRAINED_MODEL)
   # model_ref = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(PRETRAINED_MODEL)
    model_ref = create_reference_model(model)
  #  model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(PRETRAINED_MODEL)
    model.cuda()
    tokenizer.pad_token = tokenizer.eos_token


    # Update the pad_token_id in generation kwargs
    generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    # Initialize the dataset and the PPO Trainer
    dataset = IgluDataset(tokenizer=tokenizer, txt_in_len=TXT_IN_LEN, inp_column=INPUT, out_column=OUTPUT).prepare_dataset()
   # print(dataset)
    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset, data_collator=collator)

    # Execute training for several epochs
    for epoch in range(2):
        for batch in tqdm(ppo_trainer.dataloader):
            run_epoch(ppo_trainer, tokenizer, batch)

    path_to_save = f"checkpoint/{MODEL_NAME.split('/')[1]}"
    model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)

if __name__ == "__main__":
    main()
