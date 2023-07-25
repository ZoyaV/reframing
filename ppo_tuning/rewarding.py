from metrics.metrics import calculate_iou
from detectors.owlvit import OwlViTDetector


def reinforce_loss(logits, labels, model, image):
    preds = logits
    metrics = []
    for i in range(len(logits)):

        pred = preds[i].split("+")[0]

        print("!!!!!!!!!!!!!!!!1")
        print(preds[i])
        print(pred)
        print("!!!!!!!!!!!!!!!!1")

        try:
            true_bbox = eval(labels[i])
            pred_bbox = (model.get_bboxes(image[i], pred)[0][1]).tolist()
            metrics.append(calculate_iou(true_bbox, pred_bbox))
        except:
            metrics.append(0.)
    return metrics
