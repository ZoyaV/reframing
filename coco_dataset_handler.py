import os
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pandas as pd
from categories import category_dict
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
            #print(image_info)
            if keys:
                image_info = {k: image_info[k] for k in keys if k in image_info}
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            if 'category_id' in keys:
                image_info['category_id'] = list(set([ann['category_id'] for ann in anns]))
            if 'bbox' in keys:
                image_info['bbox'] = [ann['bbox'] for ann in anns]
            data.append(image_info)

        df = pd.DataFrame(data)
        # Преобразование списка bbox и category_id в список кортежей
        df['tuples'] = df.apply(lambda row: list(zip(row['bbox'], row['category_id'])), axis=1)
        # Удаление столбцов 'bbox' и 'category_id'
        df = df.drop(columns=['bbox', 'category_id'])
        # Преобразование каждого элемента списка кортежей в отдельную строку
        df = df.explode('tuples')
        # Создание отдельных столбцов 'bbox' и 'category_id' из столбца 'tuples'
        df[['bbox', 'category_id']] = pd.DataFrame(df['tuples'].tolist(), index=df.index)
        # Удаление столбца 'tuples'

        df = df.drop(columns=['tuples'])
        df.reset_index(drop=True, inplace=True)
        df = df.dropna()
        prompt = lambda x: f"""In other words, 'table' is 'furniture' +
                    In other words, 'cat' is 'animal' +
                    In other words, 'laptop' is {x}"""

        df['category_id'] = df['category_id'].map(category_dict)
        df['category_id'] = df['category_id'].apply(prompt)
        df.to_csv('ppo_tuning/dataset/prompts.csv', index=False)

        return df

