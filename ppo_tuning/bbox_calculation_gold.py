import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from PIL import Image

def preprocess_crop(img_cropped):
    a = np.asarray(img_cropped)
    i_arr = []
    j_arr = []
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j]!=0:
                i_arr.append(i)
                j_arr.append(j)
                break
    return Image.fromarray(a[min(i_arr):max(i_arr),min(j_arr):])
    
def calculate_bbox(img_cropped, img_full):
    a = np.asarray(img_full)
    i_arr = []
    j_arr = []
    crop = preprocess_crop(img_cropped)
    for i in range(50, len(a)): #explicitly set number of pixels to start from in order to avoid dataset errors
        for j in range(len(a[i])):
            if a[i][j]!=0 and a[i][j+2]!=0: 
                i_arr.append(i)
                j_arr.append(j)
                break
    return [min(j_arr), min(i_arr), crop.size[0], crop.size[1]]

def main():
    df=pd.read_csv("./dataset/modified_output.csv", header=0)
    res = []
    for i in range(len(df)):
        obj = df['item_id'][i]
        obj_split = obj.split('_')
        obj_split_len = len(obj_split)
        if obj_split_len == 3:
            obj_name = obj_split[0]
        elif obj_split_len == 4:
            obj_name = obj_split[0] + '_' + obj_split[1]
        elif obj_split_len == 5:
            obj_name = obj_split[0] + '_' + obj_split[1] + '_' + obj_split[2]
        name = obj_name+'/'+obj_name+'_'+obj_split[len(obj_split)-2]+'/'+obj+'.png'
        with Image.open("./dataset/gold/images/RGB/"+name).convert('L') as img_for_bbox:
            img_for_bbox.load()
        with Image.open("./dataset/gold/images/RGB_cropped/"+name).convert('L') as img_cropped:
            img_cropped.load()
        res.append(calculate_bbox(img_cropped, img_for_bbox))
    df['true_bbox'] = res
    df.to_csv("./dataset/modified_output_with_bboxes.csv")
if __name__ == "__main__":
    main()