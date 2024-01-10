# Function to calculate IoU
import torch
from torchvision.ops.boxes import box_area
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_start = max(x1, x2)
    y_start = max(y1, y2)
    x_end = min(x1+w1, x2+w2)
    y_end = min(y1+h1, y2+h2)
    inter_area = max(0, x_end - x_start + 1) * max(0, y_end - y_start + 1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_iou_Dino(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, x22, y22 = box2[0]
    x_start = max(x1, x2)
    y_start = max(y1, y2)
    x_end = min(x1+w1, x22)
    y_end = min(y1+h1, y22)
    inter_area = max(0, x_end - x_start + 1) * max(0, y_end - y_start + 1)
    box1_area = w1 * h1
    box2_area = (x22-x2)*(y2-y22)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def box_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    print("a",boxA)
    print("b",boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou