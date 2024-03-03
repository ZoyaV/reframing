# :smirk: Reframing - Text Query Manipulator for Improved Detection

This repository contains an experiment aimed at enhancing the performance of a text query-based detection model using an NLP-based manipulator. The manipulator is designed to modify the user's text query in such a way that the detection results are improved. The training of this manipulator is implemented using the "trl" library, which facilitates training with either human feedback or the IoU (Intersection over Union) metric evaluation.

## Introduction

The main objective of this experiment is to explore the potential of NLP-based manipulation techniques in enhancing the performance of a text-based detection model. By iteratively improving the input query, the expectation is to achieve better detection results for various tasks, such as object detection or text-based image retrieval.

## Installation

To set up the environment and dependencies required to run the code, follow these steps:

1. Clone this repository:

```bash
git clone git@github.com:ZoyaV/reframing.git
cd reframing
```

2. Install the required packages using pip (settings for python3.9):

```bash
pip install -r requirements.txt
```

## Getting Started

The experiment consists of: creating dataset using **feedback from detector**, training language model using Direct Preference Optimization (DPO) using this dataset and validating results by running detector inference on tuned model predictions.

### Creating DPO dataset 
```bash
cd dpo_tuning
sh run_dpo.sh
```
In order to use your own config or setup parametrs in console, use the following

```bash
python3 data_processing/DPO_dataset_generation.py --path_to_source ./dpo_experiment/new_DINO_gold_dataset.csv --path_to_imgs /datasets/gold/images/RGB_raw/ --model_name DINO --path_to_output ./dpo_dataset.csv
```

### Train Reframing with DPO

To train the NLP-based manipulator using reward modeling, proceed as follows:

```bash
cd dpo_tuning
sh run_dpo.sh
```
In order to train with your own config or setup parametrs in console

```bash
python3.9 train.py --config config.yaml --reward_model detector --project cunman_detection_feedback
```


### Validate results

To train Reframing using DPO-like dataset, proceed as follows:

```bash
cd dpo_tuning
sh run_validation.sh
```
Use your own config or setup parametrs in console

```bash
python3 dpo_experiment/validation.py --path_to_source ./dpo_experiment/new_DINO_gold_dataset.csv --path_to_imgs /datasets/gold/images/RGB_raw/ --detector_model_name DINO --path_to_checkpoint ./dpo_experiment/results/checkpoint-4900/ --language_model_type tuned --run_name new_data_DINO_train_4900
```

## File Structure

The repository is organized as follows:

```
|- detectors/              # Detectors models under which the manipulator is trained
|- dpo_tuning/           
    |- run.py                # Main code for DPO_loss tuning
    |- training_arguments.py # Arguments list for all scripts
    |- validation.py         # Validation code
|- requirements.txt     # List of required Python packages
```

Please make sure to update the placeholders such as `your-username`, `your-repo`, and provide specific instructions on how to prepare the datasets, run the training scripts, and perform evaluations with the models.
