import sys
sys.path.append("../")
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

os.system('pwd')
import numpy as np
import torch
from torchvision.ops import box_convert,nms
# from groundingdino.util.inference import load_model, load_image, predict
import pandas as pd
from transformers import HfArgumentParser
from dpo_tuning.training_arguments import ProcessingArguments
import statistics
import re
import random
from dpo_tuning.utils.metrics import box_iou
# from one_peace.models import from_pretrained
from dpo_tuning.utils.data import prepare_data
from dpo_tuning.utils.detector import get_Dino_predictions, get_ONE_PEACE_predictions, get_images
import tqdm
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS



def init_detector_model():
    Dino = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",\
                       "./GroundingDINO/weights/groundingdino_swint_ogc.pth")
    return Dino

def init_onepeace():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = from_pretrained(
      "ONE-PEACE_Grounding",
        model_type="one_peace_classify",
        device=device,
        dtype="float32"
        )
    return model
def init_YOLO():
    # load config
    cfg = Config.fromfile(
        "../YOLO-World/configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    )
    cfg.work_dir = "."
    cfg.load_from = "../YOLO-World/weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)

    # run model evaluation
    runner.model.eval()
    return runner

def run_image(
        text,
        runner,
        input_image,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
        output_image="output.png",
):
    try:
        output_image = "runs/detect/"+output_image
        # texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
        print(text)
        data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                         texts=[[text]]))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            runner.model.class_names = [[text]]
            pred_instances = output.pred_instances

        # nms
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        # predictions
        pred_instances = pred_instances.cpu().numpy()

    # detections = sv.Detections(
    #     xyxy=pred_instances['bboxes'],
    #     class_id=pred_instances['labels'],
    #     confidence=pred_instances['scores']
    # )

    # # label ids with confidence scores
    # labels = [
    #     f"{class_id} {confidence:0.2f}"
    #     for class_id, confidence
    #     in zip(detections.class_id, detections.confidence)
    # ]
    
        print(pred_instances['bboxes'][0],pred_instances['scores'])
        return pred_instances['bboxes'][0],pred_instances['scores'][0]
    except:
        return [0.0,0.0,0.0,0.0], [0.0]

def get_objects_descriptions(ds: pd.DataFrame):
    object_descriptions = {}
    for i in range(len(ds)):
        if ds['item_id'][i] in object_descriptions.keys():
            object_descriptions[ds['item_id'][i]].append([i,ds['description'][i]])
        else:
            object_descriptions[ds['item_id'][i]] = [[i, ds['description'][i]]]
    return object_descriptions

def evaluate_descriptions(model_name, path_to_imgs, path_to_output, path_to_source, use_score, ranking_strategy,threshhold=None):
    correct_reward_iou = []  
    print(use_score)
    correct_reward_score = []
    means = []
    prompt_bboxes = []
    if model_name == "DINO":
        model = init_detector_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
    elif model_name == 'onepeace':
        model = init_onepeace()
    elif model_name == 'yolo':
        runner = init_YOLO()
    ds = pd.read_csv(path_to_source, sep=',', header=0)
    for i in tqdm.tqdm(range(len(ds))):
        correct = ds['description'][i]
        name, img_sources, images = get_images(ds['item_id'][i], path_to_imgs)
        if use_score == True and ds["score"][i] == -1:
            correct_reward_iou.append(-1.0)
            correct_reward_score.append(0.0)
            means.append(0.0)
            prompt_bboxes.append([0,0,0,0])
            continue
        elif model_name.lower() == 'dino':
            predicted_bbox, pred_score = get_Dino_predictions(model, images, img_sources, correct)
        elif model_name.lower() == 'onepeace':
            predicted_bbox = get_ONE_PEACE_predictions(model, str(path_to_imgs)+str(name), str(correct))[0]
            pred_score = 1
        elif model_name.lower() == 'yolo':
            predicted_bbox,pred_score = run_image(str(correct),runner, str(path_to_imgs)+str(name))
        try: 
            dataset_bbox = torch.Tensor([[float(x) for x in re.split(',', ds['true_bbox'][i][1:-1])]])
            real_bbox = box_convert(boxes=dataset_bbox, in_fmt="xywh", out_fmt="xyxy").numpy()[0] 
            iou_score = float(box_iou(real_bbox, predicted_bbox))
        except Exception as e:
            iou_score = 0.0
            pred_score = 0.0
            print(f"Exception: {e}")
        correct_reward_iou.append(iou_score)
        correct_reward_score.append(pred_score)
        prompt_bboxes.append(predicted_bbox)
        means.append(statistics.harmonic_mean([iou_score, pred_score]))
        print(i, '\\', iou_score)
    ds['response_iou'] = correct_reward_iou
    ds['response_score'] = correct_reward_score
    ds['harmonic_mean'] = means
    ds['description_bbox'] = prompt_bboxes
    if ranking_strategy.lower() == 'iou':
        ds['rank'] = correct_reward_iou
    elif ranking_strategy.lower() == 'score':
        ds['rank'] = correct_reward_score
    elif ranking_strategy.lower() == 'mean':
        ds['rank'] = means
    elif ranking_strategy.lower() == 'threshhold':
        values = []
        if not threshhold:
            threshhold=0.7
        for i in range(len(correct_reward_iou)):
            if correct_reward_iou[i]>threshhold:
                values.append(correct_reward_score[i])
            else:
                values.append(0.0)
        ds['rank'] = means
    ds.to_csv(path_to_output, index=False)
    return ds


def generate_triplets(path_to_output, ds=None):
    if ds is None:
        ds = pd.read_csv(path_to_output)
    df = pd.DataFrame(columns = ['id', 'item_id', 'true_bbox', 'prompt', 'correct', 'rejected', 
                             'iou_correct', 'score_correct', 'harmonic_correct',
                             'iou_rejected', 'score_rejected', 'harmonic_rejected', 'description_bbox'])
    object_descriptions = get_objects_descriptions(ds)
    for i in range(len(ds)):
        # try:
        if ds["score"][i]==-1:
            continue
        resp1 = ds['description'][i]
        prompt = "Paraphrase sentence: " + str(resp1)
        print(prompt)
        
        while True:     
            n1 = random.choice([k for k in range(0,len(object_descriptions[ds['item_id'][i]]))])
            if object_descriptions[ds['item_id'][i]][n1] != resp1:
                resp2 = object_descriptions[ds['item_id'][i]][n1][1]
                k1 = object_descriptions[ds['item_id'][i]][n1][0]
                break
        while True:     
            n2 = random.choice([k for k in range(0,len(object_descriptions[ds['item_id'][i]]))])
            if object_descriptions[ds['item_id'][i]][n2] != resp1 and object_descriptions[ds['item_id'][i]][n2] != resp2:
                k2 = object_descriptions[ds['item_id'][i]][n2][0]
                break
        
        if ds['harmonic_mean'][k1] >=  ds['harmonic_mean'][k2]:
            df.loc[len(df)] = [i, ds['item_id'][i], ds['true_bbox'][i], prompt, ds['description'][k1], ds['description'][k2],
                               ds['response_iou'][k1], ds['response_score'][k1], ds['harmonic_mean'][k1],
                              ds['response_iou'][k2], ds['response_score'][k2], ds['harmonic_mean'][k2], ds['description_bbox'][i]]
        else: 
            df.loc[len(df)] = [i, ds['item_id'][i], ds['true_bbox'][i], prompt, ds['description'][k2], ds['description'][k1],
                               ds['response_iou'][k2], ds['response_score'][k2], ds['harmonic_mean'][k2], 
                               ds['response_iou'][k1], ds['response_score'][k1], ds['harmonic_mean'][k1], ds['description_bbox'][i]]
        # except Exception as e: 
        #     print(e)
        #     continue
    df.to_csv(path_to_output, index=False)

def main():
    # Initialize the tokenizer, model and reference model
    parser = HfArgumentParser(ProcessingArguments)
    p_args = parser.parse_args_into_dataclasses()[0]
    model_name = p_args.model_name
    path_to_source = p_args.path_to_source
    path_to_imgs = p_args.path_to_imgs
    path_to_output = p_args.path_to_output
    ranking_strategy = p_args.ranking_strategy
    threshhold = p_args.ranking_strategy
    use_score = p_args.use_score


    # ds = evaluate_descriptions(model_name, path_to_imgs, path_to_output, path_to_source,use_score,ranking_strategy,threshhold)
    ds = pd.read_csv('./YOLOWorld_gold_dataset_evaluated.csv')
    generate_triplets(path_to_output,ds)
# Generate text
if __name__ == "__main__":
    main()