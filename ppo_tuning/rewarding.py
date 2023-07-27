from metrics.metrics import calculate_iou
from detectors.owlvit import OwlViTDetector

import torch
import re


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
            prediction = prediction[1:]

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
    inputs = tokenizer.encode_plus(prompt, response, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
 #   print(model)
    # Perform forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the logits
    logits = outputs.logits

    return logits.item()
def hf_based_reward(logits, model, tokenizer, prompt):
    reward_metrics = []
    for i in range(len(logits)):
        logit = logits[i]
        logit = logit.lower().replace("<pad>", "")
        logit_parts = logit.split("</s>")
        prediction = max(logit_parts, key=len)
        print(prediction)
        score = get_score(model.cpu(), tokenizer, prompt[i], prediction)
        reward_metrics.append(score)

    print(f"Reward metrics: {reward_metrics}")
    return reward_metrics


