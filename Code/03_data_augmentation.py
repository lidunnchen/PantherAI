# There are many augmentations that can be accessed and applied from the albumentations repository; a more exhaustive list can be found here: https://github.com/albumentations-team/albumentations

## Note below there are 3 different augmentation pipelines. As part of the data preprocessing stage, it may be of interest to test model performance on datasets that are augmented (or not) with differing parameters. 

### Specify the pipeline you want to apply to images on Line 125 in the "augmented = " argument

### Refer to the associated article (Chen et al., 2025) and Supplemental Table 1 for descriptions of each script's intended usage and parameters. 


import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),  # Allows both zoom in and zoom out
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.7),  # Add rotation and shifting for symbols
 A.Resize(640, 640)  # Resize after scaling to ensure the correct output size
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

augmentation_pipeline1 = A.Compose([
   A.RandomScale(scale_limit=(-0.2, 0.1), p=1.0),  # Allows both zoom in and zoom out
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.5, rotate_limit=45, p=0.7),  # Add rotation and shifting for symbols
 A.Resize(640, 640)  # Resize after scaling to ensure the correct output size
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

augmentation_pipeline2 = A.Compose([
   A.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),  # Allows both zoom in and zoom out
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5, border_mode=0),  # Shift only, no additional rotation
 A.Resize(320, 320)  # Resize after scaling to ensure the correct output size
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def clip_bbox(bbox):
    """
    Clips bounding box values to be within [0.0, 1.0].
    Handles x_center, y_center, width, and height properly.
    """
    x_center, y_center, width, height = bbox

    # Ensure the box is within [0,1] for both center and edges.
    x0 = max(0.0, x_center - width / 2)
    y0 = max(0.0, y_center - height / 2)
    x1 = min(1.0, x_center + width / 2)
    y1 = min(1.0, y_center + height / 2)

    # Recalculate center, width, and height after clipping.
    new_x_center = (x0 + x1) / 2
    new_y_center = (y0 + y1) / 2
    new_width = x1 - x0
    new_height = y1 - y0

    return [new_x_center, new_y_center, new_width, new_height]

def is_touching_border(bbox, threshold=0.005):
    """
    Check if the bounding box is touching the border of the image.
    We add/subtract a small threshold to prevent near-border boxes.
    """
    x_center, y_center, width, height = bbox

    # Calculate box edges
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    # Check if the bounding box is touching or near the border (outside valid range)
    if x_min - threshold < 0 or y_min - threshold < 0 or x_max + threshold > 1 or y_max + threshold > 1:
        return True
    return False

def augment_and_save_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=5, remove_border_touching=False):
    """
    Augments and saves the dataset, with an option to remove bounding boxes that touch the border.

    Parameters:
    - image_dir: Directory of images.
    - label_dir: Directory of labels.
    - output_image_dir: Directory to save augmented images.
    - output_label_dir: Directory to save augmented labels.
    - num_augmentations: Number of augmentations per image.
    - remove_border_touching: If True, removes bounding boxes that touch the border of the image.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(image_dir, '*.PNG')):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue

        h, w = img.shape[:2]

        label_path = os.path.join(label_dir, img_name.replace('.PNG', '.txt'))
        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} does not exist. Skipping.")
            continue

        with open(label_path, 'r') as f:
            labels = f.readlines()

        bboxes = []
        class_labels = []
        for label in labels:
            cls, x_center, y_center, width, height = map(float, label.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(cls))

        for i in range(num_augmentations):
            try:
                augmented = augmentation_pipeline1(image=img, bboxes=bboxes, class_labels=class_labels)
                augmented_img = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_class_labels = augmented['class_labels']

                # Clip the bounding boxes after augmentation
                clipped_bboxes = [clip_bbox(bbox) for bbox in augmented_bboxes]

                # If remove_border_touching is True, filter out bounding boxes touching the border
                if remove_border_touching:
                    valid_bboxes = []
                    valid_class_labels = []
                    for augmented_bbox, clipped_bbox, cls in zip(augmented_bboxes, clipped_bboxes, augmented_class_labels):
                        if True: #not is_touching_border(clipped_bbox):
                            valid_bboxes.append(clipped_bbox)
                            valid_class_labels.append(cls)
                else:
                    valid_bboxes = clipped_bboxes
                    valid_class_labels = augmented_class_labels

                # If there are no valid bounding boxes after filtering, skip saving
                if not valid_bboxes:
                    print(f"Skipping augmentation {i} for {img_name} as no valid bounding boxes remain.")
                    continue

                # Convert the augmented image back to BGR for OpenCV
                augmented_img_bgr = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)

                augmented_img_name = img_name.replace('.PNG', f'_aug{i}.PNG')
                augmented_img_path = os.path.join(output_image_dir, augmented_img_name)
                augmented_label_path = os.path.join(output_label_dir, augmented_img_name.replace('.PNG', '.txt'))

                # Ensure the image is in the correct format for saving
                if augmented_img_bgr.dtype != np.uint8:
                    augmented_img_bgr = (augmented_img_bgr * 255).astype(np.uint8)

                cv2.imwrite(augmented_img_path, augmented_img_bgr)

#AFTER running augmented.py, copy over to the train folder (both the images and associated labels)
                with open(augmented_label_path, 'w') as f:
                    for bbox, cls in zip(valid_bboxes, valid_class_labels):
                        x_center, y_center, width, height = bbox
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            except Exception as e:
                print(f"Error during augmentation: {e}")

if __name__ == "__main__":
    image_dir = 'C:\\Users\\lchen\\Desktop\\master_combined_compiled\\images\\train'
    label_dir = 'C:\\Users\\lchen\\Desktop\\master_combined_compiled\\labels\\train'
    output_image_dir = 'C:\\Users\\lchen\\Desktop\\master_combined_compiled\\augmented\\images'
    output_label_dir = 'C:\\Users\\lchen\\Desktop\\master_combined_compiled\\augmented\\labels'
    
    augment_and_save_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=1, remove_border_touching=False)
