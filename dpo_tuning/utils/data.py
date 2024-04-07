import os
import torch
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from typing import Dict, Optional
import pandas as pd


def prepare_data(path, eos_token):
    print(path)
    data = pd.read_csv(path)#[6000:75000]
    data['chosen'] = data['correct'].apply(lambda x: "{}{}".format(x, eos_token))
    print(data['prompt'][:3])
    data['prompt'] = data['prompt'].apply(lambda x: "{}{}{}".format('[INST]', x, '[/INST]'))
    print(data['prompt'][:3])
    data['rejected'] = data['rejected'].apply(lambda x: "{}{}".format(x, eos_token))
    print(data.size)
    data = data.dropna()
    print(data.size)
    print(data.keys())
    dataset = Dataset.from_pandas(data)
    formatted_dataset = dataset.train_test_split()
    print(data[['prompt', 'chosen', 'rejected']].head())
    return formatted_dataset


def get_stack_exchange_paired(
        data_dir: str = "data/rl",
        sanity_check: bool = False,
        cache_dir: str = None,
        num_proc=24,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "lvwerra/stack-exchange-paired",
        split="train",
        cache_dir=cache_dir,
        data_dir=data_dir,
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [question + "\n\nAnswer: " for question in samples["question"]],
            "chosen": samples["response_j"],
            "rejected": samples["response_k"],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    
    
