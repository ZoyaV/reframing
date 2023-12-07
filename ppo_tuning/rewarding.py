import sys
sys.path.append("../")
sys.path.append("../GroundingDINO/")
from metrics.metrics import calculate_iou, calculate_iou_Dino
from detectors.owlvit import OwlViTDetector
from groundingdino.util.inference import predict, load_image
from torchvision.ops import box_convert

import torch
import re
import wandb
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter



import GroundingDINO.groundingdino.datasets.transforms as T

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

repetition_pen = 0
prev_prediction = ""

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def extract_nouns(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return [word.lower() for word, tag in tagged if is_noun(tag)]

def normalize_hfreward(rewards):
    return [k/max(rewards) for k in rewards]

def calculate_similarity(original, predicted):
    nouns1 = extract_nouns(original)
    nouns2 = extract_nouns(predicted)

    if not nouns1 or not nouns2:
        return 0.0

    pred_words_set = dict(Counter(nouns2).most_common())
    repeats = np.mean([ value-1 for value in pred_words_set.values()])
    repeat_coaf = repeats/len(nouns2)
    sim_coef = len(set(nouns2))/len(set(nouns1))

    t = 0
    for word in pred_words_set:
        if word in nouns1:
            t += 1
    if t == 0:
        return -1

    return sim_coef-repeat_coaf*3

def calculate_similarity2(original, predicted):
    nouns1 = extract_nouns(original)
    nouns2 = extract_nouns(predicted)

    if not nouns1 or not nouns2:
        return 0.0

    pred_words_set = dict(Counter(nouns2).most_common())
    repeats = np.mean([ value-1 for value in pred_words_set.values()])
    repeat_coaf = repeats/len(nouns2)
    sim_coef = len(set(nouns2))/len(set(nouns1))
    return sim_coef-repeat_coaf


def detector_based_reward(logits, labels, model, images):
    reward_metrics = []
    all_scores = {}
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)

        if len(prediction) == 0 or prediction == "word":
            prediction = "object"
        else:
            prediction = prediction.replace("or ", "")
            prediction = prediction.replace("of ", "")
            prediction = prediction.replace("to ", "").strip()

        print(prediction)
        counter = 0
        global repetition_pen
        global prev_prediction
        for pred_word in prediction.split():
            if pred_word in prev_prediction.split():
                counter+=1
        if counter:
            repetition_pen -= 1
        else:
            repetition_pen = 0
        try:
            true_bbox = eval(labels[i])
            predicted_bbox = model.get_bboxes(images[i], [prediction])[0]
            print("predicted_bbox = ", predicted_bbox[1])
            iou_score = float(calculate_iou(true_bbox, predicted_bbox[1])) * 1.5
            bbox_score = float(predicted_bbox[0])
            all_scores["IOU"] = iou_score
            all_scores["pred_score"] = float(predicted_bbox[0])
            all_scores["repetition_pen"] = repetition_pen
            total_reward = bbox_score + iou_score + repetition_pen
        except Exception as e:
            total_reward = 0.0
            all_scores["IOU"] = 0
            all_scores["pred_score"] = 0
            all_scores["repetition_pen"] = repetition_pen
            print(f"Exception: {e}")
        wandb.log(all_scores)
        prev_prediction = prediction
        reward_metrics.append(total_reward)

    print(f"Reward metrics: {reward_metrics}")
    return reward_metrics


def Dino_detector_based_reward(logits, labels, model, images):
    reward_metrics = []
    all_scores = {}
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)

        if len(prediction) == 0 or prediction == "word":
            prediction = "object"
        else:
            prediction = prediction.replace("or ", "")
            prediction = prediction.replace("of ", "")
            prediction = prediction.replace("to ", "").strip()

        print(prediction)
        counter = 0
        global repetition_pen
        global prev_prediction
        for pred_word in prediction.split():
            if pred_word in prev_prediction.split():
                counter+=1
        if counter:
            repetition_pen -= 1
        else:
            repetition_pen = 0
        try:
            boxes, logits_detector, phrases = predict(
                model=model,
                image=images[i],
                caption=prediction,
                box_threshold=0,
                text_threshold=0.25
            )
            h, w, _ = images[i].shape
            resized_box = boxes * torch.Tensor([w, h, w, h])
            max_pos = np.where(np.max(logits_detector.numpy()))
            
            predicted_bbox = box_convert(boxes=resized_box, in_fmt="cxcywh", out_fmt="xyxy").numpy()[max_pos]
            true_bbox = eval(labels[i])
            print(f"predicted_bbox = {predicted_bbox}, logits_detector = {logits_detector}, phrases = {phrases}")
            iou_score = float(calculate_iou_Dino(true_bbox, predicted_bbox)) * 1.5
            pred_score = float(logits_detector.numpy()[max_pos])
            all_scores["pred_score"] = pred_score
            all_scores["IOU"] = iou_score
            all_scores["repetition_pen"] = 0
            total_reward = iou_score + pred_score
        except Exception as e:
            total_reward = 0.0
            all_scores["IOU"] = 0
            all_scores["pred_score"] = 0
            all_scores["repetition_pen"] = 0
            total_reward = 0.0
            print(f"Exception: {e}")

        
        wandb.log(all_scores)
        prev_prediction = prediction
        reward_metrics.append(total_reward)

    print(f"Reward metrics: {reward_metrics}")
    return reward_metrics

def get_score(model, tokenizer, prompt, response):
    # Tokenize the input sequences

  #  print(prompt)
  #  print(response)
    inputs = tokenizer.encode_plus(prompt, response, truncation=True, padding=False, max_length=512, return_tensors="pt")
 #   print(model)
    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()
def hf_based_reward(logits, reward_model, tokenizer, prompt):
    reward_metrics = []
    all_scores = {}
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)
       # nt(prediction)
        sim_score = calculate_similarity(prompt[i].replace("</s>", ""), prediction)
        #print("SIM SCORE: ", sim_score)
        hf_score = get_score(reward_model.cpu(), tokenizer, prompt[i].replace("</s>", ""), prediction)
       # print("HF SCORE: ", hf_score)
        score = hf_score + sim_score
        reward_metrics.append(score)
        all_scores["sim_score"] = sim_score
        all_scores["hf_score"] = hf_score
        wandb.log(all_scores)

   # print(f"Reward metrics: {reward_metrics}")
    return reward_metrics


