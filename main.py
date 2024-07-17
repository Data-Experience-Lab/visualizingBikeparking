import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

# Add relevant directories to the Python path
sys.path.append(os.path.abspath('segmentation'))
sys.path.append(os.path.abspath('Segmentation_RLHS/DeepLabV3Plus'))
sys.path.append(os.path.abspath('Models'))
sys.path.append(os.path.abspath('rack_detection'))

import segmentation.models.model_1.model as Model
from segmentation.segmentation_inference import SegmentationInference
from Segmentation_RLHS.DeepLabV3Plus.predict import segmentation
from Models.yolo import *
from rack_detection.detect import process_single


# Models
model = YOLO('yolov8x.pt')



def compute(si, input_file_name_prefix):
    image       = Image.open(input_file_name_prefix)
    image       = image.convert('RGB')
    image       = image.resize((512, 256))
    image_np    = numpy.array(image)

    prediction_np, mask, result = si.process(image_np)
    image = (mask*255).astype(numpy.uint8)
    



def pipeline(image_dir):
    
    images = sorted(os.listdir(image_dir))
    for image in images:
        image_path = os.path.join(image_dir, image)
        image = cv2.imread(image_path)
        
        # contrast
        contrast = np.std(image)
        
        #segmentation area
        Vegetation_area, terrain_area = segmentation(image_path)
        
        #Bikes and Seating
        results = model(image_path)
        classes = results[0].boxes.cls.tolist()
        classes_to_count = [1, 13]  # 1 for bicycle, 13 for bench
        out_dict = count_yolo_classes(classes, class_mapping, classes_to_count)
        
        #Bike Rack
        confidence = process_single(image_path)
        
        print("Contrast", contrast, "  Vegetation Areas ", Vegetation_area, "  Terrain Areas",terrain_area, "Bikess and Seating ", out_dict, "  Bike rack ", confidence)
        
pipeline("D:\\RLHS-MITACS\\Data\\final_data")