import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
from utils import *

model_dir = "D:\\RLHS-MITACS\\Segmentation_RLHS\\model"
MODEL_NAME = 'resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_saved_model' 


LOADED_MODEL = tf.saved_model.load(os.path.join(model_dir, MODEL_NAME))


def segment(image_dir,output_dir, dataset_info, perturb_noise=60, alpha = 0.35):
    
    image_names = os.listdir(image_dir)
    
    for image_name in image_names:
        
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        print('image taken')
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = np.array(image)
        
        output = LOADED_MODEL(tf.cast(image, tf.uint8))
        print("out")
        
        panoptic_map, used_colors = color_panoptic_map(output['panoptic_pred'][0],dataset_info, perturb_noise)
        
        cv2.imwrite(os.path.join(output_dir,image_name),image)
        cv2.imwrite(os.path.join(output_dir, "_mask"+ image_name), panoptic_map)
  

        masked = (1.0 - alpha)*image + alpha*panoptic_map
        cv2.imwrite(os.path.join(output_dir, "_masked"+ image_name), masked)
        print("done")


image_dir = "D:/RLHS-MITACS/Segmentation_RLHS/dummy"
output_dir = "D:/RLHS-MITACS/Segmentation_RLHS/output"

segment(image_dir, output_dir,DATASET_INFO)