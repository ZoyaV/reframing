from torchvision.ops import box_convert
from groundingdino.util.inference import predict
import numpy as np
import torch
import cv2
from one_peace.models import from_pretrained




def get_Dino_predictions(Dino, images, img_sources, output):
    boxes, logits_detector, phrases = predict(
                            model=Dino,
                            image=images,
                            caption=str(output),
                            box_threshold=0,
                            text_threshold=0.25
                        )
    h, w, _ = img_sources.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    max_pos = np.where(np.max(logits_detector.numpy()))
    predicted_bbox = box_convert(boxes=boxes[max_pos], in_fmt="cxcywh", out_fmt="xyxy").numpy()[max_pos][0]
    pred_score = float(logits_detector.numpy()[max_pos])
    return predicted_bbox, pred_score

def get_ONE_PEACE_predictions(onepeace, img_path, text):
    (src_images, image_widths, image_heights), src_tokens  = onepeace.process_image_text_pairs(
    [(img_path, text)], return_image_sizes=True
    )
    with torch.no_grad():
    # extract features
        vl_features = onepeace.extract_vl_features(src_images, src_tokens).sigmoid()
        # extract coords
        vl_features[:, ::2] *= image_widths.unsqueeze(1)
        vl_features[:, 1::2] *= image_heights.unsqueeze(1)
        coords = vl_features.cpu().tolist()
    return coords