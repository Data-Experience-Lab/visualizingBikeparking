import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

sys.path.append(os.path.abspath('Segmentation_RLHS/DeepLabV3Plus'))
sys.path.append(os.path.abspath('Models'))
sys.path.append(os.path.abspath('rack_detection'))

from Segmentation_RLHS.DeepLabV3Plus.predict import segmentation
from Models.yolo import *
from rack_detection.detect import process_single


# Models
model = YOLO('yolov8x.pt')

import pandas as pd
import numpy as np

# Initialize an empty DataFrame with the required columns
df = pd.DataFrame(columns=[
    "image_id", "contrast", "Vegetation_area", "terrain_area", 
    "bench_conf", "bicycle_count", "confidence"
])

# Example data


# Function to process each image and return the required values
def process_image(image_path):
    image = Image.open(image_path) 
    contrast = np.std(image)
    
    #calculating vegetation and terrain area
    Vegetation_area, terrain_area = segmentation(image_path)
    results = model(image_path)
    classes = results#[0].boxes.cls.tolist()

    #calculating the bike and bench confidences
    out_dict, bench_conf, bike_conf = count_yolo_classes(classes, class_mapping, [1, 13])

    #calculating bike rack confidence
    confidence = process_single(image_path)
    
    bench_count = out_dict.get('bench', 0)
    bicycle_count = out_dict.get('bicycle', 0)
    
    return contrast, Vegetation_area, terrain_area, bench_conf, bicycle_count, bike_conf,  confidence


# Process each image and add a row to the DataFrame

# Specify theh folder path here
image_dir = "D:\\project\\Data\\final_data"
images = os.listdir(image_dir)
images = sorted(images, key=lambda x: int(os.path.splitext(x)[0]))

all_rows = []
for image in images:
    image_path = os.path.join(image_dir, image)
    contrast, Vegetation_area, terrain_area, bench_conf, bicycle_count,bike_conf, confidence = process_image(image_path)
    row = {
        "image_id": image,
        "contrast": contrast,
        "Vegetation_area": Vegetation_area,
        "terrain_area": terrain_area,
        "bench_conf": bench_conf,
        "bicycle_count": bicycle_count,
        "bicycle_conf": bike_conf,
        "confidence": confidence
    }
    all_rows.append(row)
    
df = pd.concat([df, pd.DataFrame(all_rows)], ignore_index=True)

# Display the DataFrame
df.to_csv("output_4.csv", index=False)
        
# pipeline("D:\\RLHS-MITACS\\Data\\final_data")
