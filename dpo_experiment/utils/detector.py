from torchvision.ops import box_convert
from groundingdino.util.inference import predict
import numpy as np
import torch



def get_Dino_predictions(Dino, images, img_sources, output):
    boxes, logits_detector, phrases = predict(
                            model=Dino,
                            image=images,
                            caption=output,
                            box_threshold=0,
                            text_threshold=0.25
                        )
    h, w, _ = img_sources.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    max_pos = np.where(np.max(logits_detector.numpy()))
    predicted_bbox = box_convert(boxes=boxes[max_pos], in_fmt="cxcywh", out_fmt="xyxy").numpy()[max_pos][0]
    pred_score = float(logits_detector.numpy()[max_pos])
    return predicted_bbox, pred_score