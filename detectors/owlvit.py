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
            target_sizes = torch.Tensor([image.shape[:2]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.0001)
    #    print(results)
        # Get prediction logits
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        # Filter out low score predictions
        score_threshold = 0.0
        result_boxes = [(score, box, label) for score, box, label in zip(scores, boxes, labels) if
                        score >= score_threshold]

        return result_boxes
