import cv2
import ultralytics
from ultralytics import YOLO
from collections import defaultdict

# Loading the custom model
model = YOLO('yolov8x.pt')

# Mappings as per requirement
class_mapping = {0: 'person', 1: 'bicycle', 13: 'bench'}

input_dir ='D:\\RLHS-MITACS\\Data\\final_data'

output_dir = "D:\\RLHS-MITACS\\Models\\output"


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
    
    yolo_class = yolo_output[0].boxes.cls.tolist()
    yolo_conf = yolo_output[0].boxes.conf.tolist()
    for class_index in yolo_class:
        if class_index in classes_to_count:
            class_name = class_mapping[class_index]
            class_counts[class_name] += 1
    
    conf_index = [i for i, cls in enumerate(yolo_class) if cls == 13]
    bench_confs = [yolo_conf[i] for i in conf_index]
    # print(bench_confs)
    if len(bench_confs)==0:
        bench_conf = 0
    else:
        bench_conf = max(bench_confs)
        
    bicycle_conf_index = [i for i, cls in enumerate(yolo_class) if cls == 1]
    bicycle_confs = [yolo_conf[i] for i in bicycle_conf_index]
    bicycle_conf = max(bicycle_confs, default=0)
    
    return dict(class_counts), bench_conf, bicycle_conf
    

    # return dict(class_counts), bench_conf


# results = model("D:\\RLHS-MITACS\\Data\\final_data\\43.png")
# classes = results

# print(classes)
# # Specify only the classes you want to count
# classes_to_count = [1, 13]  # 1 for bicycle, 13 for bench

# out_dict,bench_conf = count_yolo_classes(classes, class_mapping, classes_to_count)

# print(out_dict, bench_conf)


