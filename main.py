from coco_dataset_handler import CocoWrapper
from detectors.owlvit import OwlViTDetector
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from metrics.metrics import calculate_iou

def plot_bbox(im, true_bbox, predicted_bbox):
    # Open the image
   # im = np.array(Image.open(image_path), dtype=np.uint8)

    # Create figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a rectangle for the true bbox
    true_rect = patches.Rectangle((true_bbox[0], true_bbox[1]), true_bbox[2], true_bbox[3],
                                  linewidth=1, edgecolor='r', facecolor='none')

    # Create a rectangle for the predicted bbox
    predicted_rect = patches.Rectangle((predicted_bbox[0], predicted_bbox[1]), predicted_bbox[2], predicted_bbox[3],
                                       linewidth=1, edgecolor='b', facecolor='none')

    # Add the rectangles to the image
    ax.add_patch(true_rect)
    ax.add_patch(predicted_rect)

    plt.show()

def main():
    coco_wrapper = CocoWrapper('./dataset/result.json')

    # Get all annotations for the given image and category
    annotations = coco_wrapper.get_anns(image_id=1, category_id=24)
    # Get image by id
    image = coco_wrapper.get_image(image_id=1, image_folder_path='./dataset/imgs')

    text_queries = ['door']
    try:
        model = OwlViTDetector("google/owlvit-base-patch32")
        predicted_bbox = (model.get_bboxes(image, text_queries)[0][1]).tolist()
    except IndexError:
        predicted_bbox = [100.0, 150.0, 200.0, 250.0]
        # Apply the function
    for annotation in annotations:
        true_bbox = annotation['bbox']
        iou = calculate_iou(true_bbox, predicted_bbox)
        print(f'IoU for the bbox = {iou}')
        plot_bbox(image, true_bbox, predicted_bbox)

        # Get pandas dataframe
   # df = coco_wrapper.to_pandas(keys=['id', 'file_name'])
    #print(df)


if __name__ == "__main__":
    main()