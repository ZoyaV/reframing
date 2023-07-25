import torch


class BaseDetector:
    def __init__(self, model_name):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def get_bboxes(self, image, text_queries):
        raise NotImplementedError

    def get_segmentation_coordinates(self, image, text_queries):
        pass


