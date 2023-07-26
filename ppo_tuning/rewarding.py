from metrics.metrics import calculate_iou
from detectors.owlvit import OwlViTDetector

import re


def reinforce_loss(logits, labels, model, image):
    preds = logits
    metrics = []
    stopwords = ['im', 'stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'i', 'that', 'had', 'on', 'for', 'were', 'was']
    t = 0
    for i in range(len(logits)):
        pred = preds[i].lower().replace("<pad>", "").split("</s>")
        pred = pred if pred not in stopwords else "word"
        pred = pred[0].split()[0]
        regex = re.compile('[^a-zA-Z]')
        pred = regex.sub('', pred)
        try:
            true_bbox = eval(labels[i])
            pred_bbox = (model.get_bboxes(image[i], pred)[0][1]).tolist()
            t += 0.3
            t += calculate_iou(true_bbox, pred_bbox)*1.5
        except Exception as e:
            t = -0.1
            print(e)
        metrics.append(t)
    return metrics
