import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import os

# Function to visualize detections (same as your current function)
def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_x = int(bbox.origin_x)
        start_y = int(bbox.origin_y)
        end_x = int(bbox.origin_x + bbox.width)
        end_y = int(bbox.origin_y + bbox.height)
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 8)
        if detection.categories:
            category_name = detection.categories[0].category_name
            score = detection.categories[0].score
            label = f"{category_name}: {score:.2f}"
            cv2.putText(image, label, (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Paths
input_folder = "/home/xr23/myenv/MediapipeObjectDetect/Human-Foot-Object-Detction-3/test"
detected_folder = "/home/xr23/myenv/MediapipeObjectDetect/DetectedImages/detected"
original_folder = "/home/xr23/myenv/MediapipeObjectDetect/DetectedImages/original"

# Create folders if they don't exist
os.makedirs(original_folder, exist_ok=True)
os.makedirs(detected_folder, exist_ok=True)

# Mediapipe ObjectDetector
base_options = python.BaseOptions(model_asset_path='/home/xr23/myenv/MediapipeObjectDetect/detection_exported_model/footDetection1.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.1)
detector = vision.ObjectDetector.create_from_options(options)

# Process all images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for idx, image_file in enumerate(image_files, start=1):
    IMAGE_FILE = os.path.join(input_folder, image_file)
    
    # Read image
    img = cv2.imread(IMAGE_FILE)
    
    # Save original image
    save_path_original = os.path.join(original_folder, f"original{idx}.png")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Original Image")
    plt.savefig(save_path_original)
    plt.close()
    
    # Load for Mediapipe
    image_mp = mp.Image.create_from_file(IMAGE_FILE)
    
    # Detect objects
    detection_result = detector.detect(image_mp)
    
    # Annotate image
    image_copy = np.copy(image_mp.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Save detected image
    save_path_detected = os.path.join(detected_folder, f"detected{idx}.png")
    plt.imshow(rgb_annotated_image)
    plt.axis("off")
    plt.title("Detected Objects")
    plt.savefig(save_path_detected)
    plt.close()
    
    print(f"✅ Processed {image_file} → original{idx}.png & detected{idx}.png")
