import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import box_convert,nms
from groundingdino.util.inference import load_model, load_image, predict
#from one_peace.models import from_pretrained

class BaseDetector:
    def __init__(self, detector_class):
        self.detector_class = detector_class
        if self.detector_class == 'Dino':
            self.detector = DinoDetector()
        if self.detector_class == 'OnePeace':
            self.detector = OnePeaceDetector()
        if self.detector_class == 'YOLO':
            self.detector = YOLOWorldDetector()
    def predict(self, image_metadata):
        return self.detector.predict(image_metadata)

    def get_bboxes(self, image, text_queries):
        raise NotImplementedError

    def get_segmentation_coordinates(self, image, text_queries):
        raise NotImplementedError


class DinoDetector:
    def __init__(self):

        self.detector = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", \
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
    
class YOLOWorldDetector:
    def __init__(self):
        cfg = Config.fromfile(
            "../YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
        )
        cfg.work_dir = "."
        cfg.load_from = "../YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
        self.detector = Runner.from_cfg(cfg)
        self.detector.call_hook("before_run")
        self.detector.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        self.detector.pipeline = Compose(pipeline)
        self.detector.model.eval()

    def _scrap_image_metadata(self, image_metadata):
        return image_metadata['image_path'], image_metadata['correct']
    
    def predict(
        self,
        image_metadata,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
        output_image="output.png",
):
        image_path, output = self._scrap_image_metadata(image_metadata)
        # texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
        data_info = self.detector.pipeline(dict(img_id=0, img_path=image_path,
                                         texts=[[output]]))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        try: 
            with autocast(enabled=False), torch.no_grad():
                output = self.detector.model.test_step(data_batch)[0]
                self.detector.model.class_names = [[output]]
                pred_instances = output.pred_instances

            keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
            pred_instances = pred_instances[keep_idxs]
            pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

            if len(pred_instances.scores) > max_num_boxes:
                indices = pred_instances.scores.float().topk(max_num_boxes)[1]
                pred_instances = pred_instances[indices]
            output.pred_instances = pred_instances

            pred_instances = pred_instances.cpu().numpy()
            max_position = np.argmax(pred_instances['scores'])
            return pred_instances['bboxes'][max_position],pred_instances['scores'][max_position]
        except Exception as e:
            print(e) 
            return [0.0,0.0,0.0,0.0], 0.0