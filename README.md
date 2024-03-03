# :smirk: Reframing - Text Query Manipulator for Improved Detection

This repository contains an experiment aimed at enhancing the performance of a text query-based detection model using an NLP-based manipulator. The manipulator is designed to modify the user's text query in such a way that the detection results are improved. The training of this manipulator is implemented using the "trl" library, which facilitates training with either human feedback or the IoU (Intersection over Union) metric evaluation.

## Introduction

The main objective of this experiment is to explore the potential of NLP-based manipulation techniques in enhancing the performance of a text-based detection model. By iteratively improving the input query, the expectation is to achieve better detection results for various tasks, such as object detection or text-based image retrieval.

## Installation

To set up the environment and dependencies required to run the code, follow these steps:

1. Clone this repository:

```bash
git clone git@github.com:ZoyaV/cunning_manipulator.git
cd cunning_manipulator
```

2. Install the required packages using pip (settings for python3.9):

```bash
pip install -r requirements.txt
```

## Getting Started

The experiment consists of two main parts: training NLP-based manipulator using Proximal Policy Optimization (PPO) tuning with **feedback from detector**, training the NLP-based manipulator with reward based on human feedback.


### Training the NLP-based Manipulator with Reward from Detector Feedback

To train the NLP-based manipulator using reward modeling, proceed as follows:

```bash
cd ppo_tuning
```
Traing with your own config or setup parametrs in console

```bash
python3.9 train.py --config config.yaml --reward_model detector --project cunman_detection_feedback
```


### Training the NLP-based Manipulator with Reward from Human Feedback

1. At First train Human Feedback model *ppo_tuning/human_feedback/HFModel.ipynb*

2. To train the NLP-based manipulator using reward modeling, proceed as follows:

```bash
cd ppo_tuning
```
Traing with your own config or setup parametrs in console

```bash
python3.9 train.py --config config.yaml --reward_model hf --inp prompt --out text --txt_in_len 30 --txt_out_len 30 --project cunman_human_feedback
```


## Running the Models

Run the mine script from the root directory

```bash
python3.9 main.py 
```

## File Structure

The repository is organized as follows:

```
|- detectors/              # Detectors models under which the manipulator is trained
|- metrics/            # Metrics calculation such as IOU using for Detector feedback calculation
|- ppo_tuning/           # Main code for PPO_loss tuning
|- requirements.txt   # List of required Python packages
```

Please make sure to update the placeholders such as `your-username`, `your-repo`, and provide specific instructions on how to prepare the datasets, run the training scripts, and perform evaluations with the models.
