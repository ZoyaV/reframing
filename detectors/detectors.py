import numpy as np
import torch
import cv2

from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict
from one_peace.models import from_pretrained

class BaseDetector:
    def __init__(self, detector_class):
        self.detector_class = detector_class
        if self.detector_class == 'Dino':
            self.detector = DinoDetector()
        if self.detector_class == 'OnePeace':
            self.detector = OnePeaceDetector()
    def predict(self, image_metadata):
        raise self.detector.predict(image_metadata)

    def get_bboxes(self, image, text_queries):
        raise NotImplementedError

    def get_segmentation_coordinates(self, image, text_queries):
        raise NotImplementedError


class DinoDetector:
    def __init__(self):

        self.detector = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",\
                       "./GroundingDINO/weights/groundingdino_swint_ogc.pth")

    def _scrap_image_metadata(self, image_metadata):
        return (image_metadata['image_np'],
                                           image_metadata['image_pil'],
                                           image_metadata['correct'])


    def predict(self, image_metadata):
            images, img_sources, output = self._scrap_image_metadata(image_metadata)
            boxes, logits_detector, phrases = predict(
                model=self.detector,
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


class OnePeaceDetector:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = from_pretrained(
                        "ONE-PEACE_Grounding",
                        model_type="one_peace_classify",
                        device=device,
                        dtype="float32"
                        )

    def _scrap_image_metadata(self, image_metadata):
        return image_metadata['image_path'], image_metadata['correct']
    def predict(self, image_metadata):
            image, output = self._scrap_image_metadata(image_metadata)
            (src_images, image_widths, image_heights), src_tokens = self.detector.process_image_text_pairs(
                                                                        [(image, output)], return_image_sizes=True
                                                                        )
            with torch.no_grad():
                vl_features = self.detector.extract_vl_features(src_images, src_tokens).sigmoid()
                # extract coords
                vl_features[:, ::2] *= image_widths.unsqueeze(1)
                vl_features[:, 1::2] *= image_heights.unsqueeze(1)
                coords = vl_features.cpu().tolist()
            return coords, 1