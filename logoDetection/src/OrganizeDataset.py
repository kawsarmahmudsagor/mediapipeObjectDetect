import os
import shutil
import json
import math
from collections import defaultdict

def move_images_to_subfolder(dataset_split_folder):
    images_folder = os.path.join(dataset_split_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    for item in os.listdir(dataset_split_folder):
        item_path = os.path.join(dataset_split_folder, item)
        # Skip annotations json and the images folder itself
        if os.path.isfile(item_path) and item.endswith((".jpg", ".jpeg", ".png")):
            shutil.move(item_path, os.path.join(images_folder, item))
    print(f"Images moved to {images_folder}")

# Example usage:
train_dataset_path = "/home/sagor/Projects/mediapipeObjectDetect/logoDetection/src/Logo_detection-12/train"
validation_dataset_path = "/home/sagor/Projects/mediapipeObjectDetect/logoDetection/src/Logo_detection-12/valid"

move_images_to_subfolder(train_dataset_path)
move_images_to_subfolder(validation_dataset_path)