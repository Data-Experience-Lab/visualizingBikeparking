import ultralytics
import cv2
import os
from PIL import Image
from ultralytics import YOLO
import inference
import supervision as sv

# Loading the custom model
model1 = YOLO('D:/RLHS-MITACS/rack_detection/class-models/best_classify.pt')
model4 = inference.get_model(model_id="classify-racks/1", api_key="ZVyfCUbZT6qdSHQPyazV")


        
def detect_yolo_class(image_path):
    results = model1(image_path)
    
    # probs = results[0].boxes.conf.float().tolist()
    # print("results :", results[0].probs.top1)
    res = results[0].probs.top1 # 0 --> absent, 1--> present
    
    return res

def detect_roboflow_class(image_path):
    
    data_list = model4.infer(image = image_path)
    # detections = sv.Classifications.get_top_k(results[0])
    # print(results)
    
    
    data_dict = {
    "visualization": data_list[0].visualization,
    "frame_id": data_list[0].frame_id,
    "time": data_list[0].time,
    "image": {
        "width": data_list[0].image.width,
        "height": data_list[0].image.height
    },
    "predictions": [
        {
            "class_name": prediction.class_name,
            "class_id": prediction.class_id,
            "confidence": prediction.confidence
        }
        for prediction in data_list[0].predictions
    ],
    "top": data_list[0].top,
    "confidence": data_list[0].confidence,
    "parent_id": data_list[0].parent_id
}
    # confs = detections.confidence
    print(data_dict)
    # valid_probs = [prob for prob in confs if prob >= 0.4]
    
    # if valid_probs:
    #     return max(valid_probs)
    # else:
    #     return 0
    
detect_roboflow_class('D:/RLHS-MITACS/rack_detection/test_set/202_png.rf.36105c3d17466c226486189157e34d44.jpg')