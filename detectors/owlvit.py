import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from .base import BaseDetector


class OwlViTDetector(BaseDetector):
    def load_model(self):
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

    def get_bboxes(self, image, text_queries):
        # Process image and text inputs
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)

        # Set model in evaluation mode
        self.model.eval()

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get prediction logits
        logits = torch.max(outputs["logits"][0], dim=-1)
        scores = torch.sigmoid(logits.values).cpu().detach().numpy()

        # Get prediction labels and boundary boxes
        labels = logits.indices.cpu().detach().numpy()
        boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

        # Filter out low score predictions
        score_threshold = 0.1
        result_boxes = [(score, box, label) for score, box, label in zip(scores, boxes, labels) if
                        score >= score_threshold]

        return result_boxes