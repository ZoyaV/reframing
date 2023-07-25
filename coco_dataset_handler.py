import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pandas as pd

class CocoWrapper:
    def __init__(self, annotation_file_path):
        self.coco = COCO(annotation_file_path)

    def get_anns(self, image_id, category_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=category_id, iscrowd=None)
        return self.coco.loadAnns(ann_ids)

    def get_image(self, image_id, image_folder_path):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_folder_path, image_info['file_name'])
        return plt.imread(image_path)

    def get_image_by_name(self, image_name):
        img_ids = self.coco.getImgIds()
        for i in img_ids:
            img_info = self.coco.loadImgs(i)[0]
            if img_info['file_name'] == image_name:
                return self.get_image(i)
        return None

    def get_anns_by_image_name(self, image_name, category_id):
        img_ids = self.coco.getImgIds()
        for i in img_ids:
            img_info = self.coco.loadImgs(i)[0]
            if img_info['file_name'] == image_name:
                return self.get_anns(i, category_id)
        return None

    def to_pandas(self, keys=None):
        data = []
        for image_id in self.coco.getImgIds():
            image_info = self.coco.loadImgs(image_id)[0]
            if keys:
                image_info = {k: image_info[k] for k in keys if k in image_info}
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            if 'category_id' in keys:
                image_info['categories'] = list(set([ann['category_id'] for ann in anns]))
            data.append(image_info)
        return pd.DataFrame(data)

