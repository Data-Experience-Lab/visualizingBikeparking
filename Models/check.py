import os
import cv2
import ultralytics
from ultralytics import YOLO
from collections import defaultdict

# Loading the custom model
model = YOLO('yolov8x.pt')

# Mappings as per requirement
class_mapping = {0: 'person', 1: 'bicycle', 13: 'bench'}

input_dir = 'D:\\RLHS-MITACS\\Data\\final_data'
output_dir = 'D:\\RLHS-MITACS\\Models\\output'


def write_text_opencv(image_path, text, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (10, 30)
    
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_path, image)


def count_yolo_classes(yolo_output, class_mapping, classes_to_count):
    class_counts = defaultdict(int)

    for class_index in yolo_output:
        if class_index in classes_to_count:
            class_name = class_mapping[class_index]
            class_counts[class_name] += 1

    return dict(class_counts)


# Process each image in the input directory
# for filename in os.listdir(input_dir):
#     if filename.endswith(".png") or filename.endswith(".jpg"):  # Add more extensions if needed
#         image_path = os.path.join(input_dir, filename)
        
#         results = model(image_path)
#         classes = results[0].boxes.cls.tolist()

#         # Specify only the classes you want to count
#         classes_to_count = [1, 13]  # 1 for bicycle, 13 for bench

#         out_dict = count_yolo_classes(classes, class_mapping, classes_to_count)
        
#         # Create the text to write on the image
#         text = f"Bicycles: {out_dict.get('bicycle', 0)}, Benches: {out_dict.get('bench', 0)}"
        
#         output_path = os.path.join(output_dir, filename)
        
#         # Write the text on the image and save it
#         write_text_opencv(image_path, text, output_path)
