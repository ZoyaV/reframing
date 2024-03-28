from torchvision.ops import box_convert
from groundingdino.util.inference import predict, load_image
import numpy as np
import torch
import cv2
<<<<<<< HEAD
#from one_peace.models import from_pretrained
=======
from one_peace.models import from_pretrained
import os
>>>>>>> d082dea (recent changes)




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


def get_images(obj, path):
    obj_split = obj.split('_')
    obj_split_len = len(obj_split)
    if obj_split_len == 3:
        obj_name = obj_split[0]
    elif obj_split_len == 4:
        obj_name = obj_split[0] + '_' + obj_split[1]
    elif obj_split_len == 5:
        obj_name = obj_split[0] + '_' + obj_split[1] + '_' + obj_split[2]
    name = obj_name+'/'+obj_name+'_'+obj_split[len(obj_split)-2]+'/'+obj+'.png'
    img_sources, images = load_image(path+name)
    return name, img_sources, images

def annotate_and_save(i, predicted_bbox, real_bbox, prompt_bbox, output_dir, path_to_imgs, name, run_name):
    if os.path.exists(f'{output_dir}/input_output_examples.txt'):
        with open(f'{output_dir}/input_output_examples.txt', 'a') as f:
            print("output proxy")
            # f.write("{} ||| {} ||| {} ||| {} ||| {}\n".format(c, data['prompt'][i],output, iou_score, pred_score))
    else:
        os.system(f"touch {output_dir}/input_output_examples.txt")
        with open(f'{output_dir}/input_output_examples.txt', 'a') as f:
            print("output proxy")
            # f.write("{} ||| {} ||| {} ||| {} ||| {}\n".format(c, data['prompt'][i],output, iou_score, pred_score))        
    image = plt.imread(path_to_imgs+name)
    image = cv2.rectangle(image, (int(predicted_bbox[0]), int(predicted_bbox[1])), (int(predicted_bbox[2]), int(predicted_bbox[3])), (0, 0, 0), 2)
    cv2.putText(image, 'predicted', (int(predicted_bbox[0]), int(predicted_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    image = cv2.rectangle(image, (int(real_bbox[0]), int(real_bbox[1])), (int(real_bbox[2]), int(real_bbox[3])), (36,255,12), 2)
    cv2.putText(image, 'dataset', (int(real_bbox[0]), int(real_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    image = cv2.rectangle(image, (int(prompt_bbox[0]), int(prompt_bbox[1])), (int(prompt_bbox[2]), int(prompt_bbox[3])), (0,255,0), 2)
    cv2.putText(image, 'dataset', (int(prompt_bbox[0]), int(prompt_bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    im = Image.fromarray((image * 255).astype(np.uint8)).convert('RGB')
    im.save(f"./{output_dir}/images/images_{run_name}_{i}.png")