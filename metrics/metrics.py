# Function to calculate IoU
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