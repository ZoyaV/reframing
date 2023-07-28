from metrics.metrics import calculate_iou
from detectors.owlvit import OwlViTDetector

import torch
import re

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def extract_nouns(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    return [word.lower() for word, tag in tagged if is_noun(tag)]

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
            prediction = prediction.replace("to ", "")
        print(prediction)
        try:
            true_bbox = eval(labels[i])
            predicted_bbox = model.get_bboxes(images[i], [prediction])[0]
            total_reward = float(predicted_bbox[0]) + float(calculate_iou(true_bbox, predicted_bbox[1])) * 1.5
        except Exception as e:
            total_reward = 0.0
            print(f"Exception: {e}")

        reward_metrics.append(total_reward)

    print(f"Reward metrics: {reward_metrics}")
    return reward_metrics

def get_score(model, tokenizer, prompt, response):
    # Tokenize the input sequences

  #  print(prompt)
  #  print(response)
    inputs = tokenizer.encode_plus(prompt, response, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
 #   print(model)
    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()
def hf_based_reward(logits, reward_model, tokenizer, prompt):
    reward_metrics = []
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)
       # print(prediction)
        sim_score = calculate_similarity(prompt[i].replace("</s>", ""), prediction)
        #print("SIM SCORE: ", sim_score)
        hf_score = get_score(reward_model.cpu(), tokenizer, prompt[i].replace("</s>", ""), prediction)/8
       # print("HF SCORE: ", hf_score)
        score = hf_score + sim_score
        reward_metrics.append(score)

    print(f"Reward metrics: {reward_metrics}")
    return reward_metrics


