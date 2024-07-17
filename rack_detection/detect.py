import ultralytics
import cv2
import os
from PIL import Image
from ultralytics import YOLO
import inference
import supervision as sv

# Loading the custom model
model1 = YOLO('D:/RLHS-MITACS/rack_detection/models/best.pt')
model3 = YOLO('D:/RLHS-MITACS/rack_detection/class-models/best_classify.pt')
model2 = inference.get_model(model_id="bike-rack-unayc/2", api_key="ZVyfCUbZT6qdSHQPyazV")
model4 = inference.get_model(model_id="classify-racks/1", api_key="ZVyfCUbZT6qdSHQPyazV")


        
def detect_yolo(image_path):
    results = model1(image_path)
    
    probs = results[0].boxes.conf.float().tolist()
    # valid_probs = [prob for prob in probs if prob >= 0.3]
    
    # if valid_probs:
    #     return max(valid_probs)
    # else:
    #     return 0
    if len(probs != 0):
        return max(probs)
    else:
        return 0

def detect_roboflow(image_path):
    print(image_path)
    results = model2.infer(image = image_path)
    detections = sv.Detections.from_inference(results[0])
    confs = detections.confidence
    # valid_probs = [prob for prob in confs if prob >= 0.4]
    
    # if valid_probs:
    #     return max(valid_probs)
    # else:
    #     return 0
    if len(confs != 0):
        return max(confs)
    else:
        return 0
   


def detect_yolo_class(image_path):
    results = model3(image_path)
    
    # probs = results[0].boxes.conf.float().tolist()
    # print("results :", results[0].probs.top1)
    res = results[0].probs.top1 # 0 --> absent, 1--> present
    
    return res



def extract_bike_rack_confidence(output):
    
        start_index = output.find("bike_rack', class_id=0, confidence=") + len("bike_rack', class_id=0, confidence=")
        end_index = output.find(")", start_index)
        confidence_value = float(output[start_index:end_index])
        return confidence_value
    
    
def detect_roboflow_class(image_path):
    data_list = model4.infer(image = image_path)
    # if data_list[0].top == "bike_rack":
    #     return data_list[0]
    # else:
    #     return data_list[0] 
    return extract_bike_rack_confidence(str(data_list))



    
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



def process(input_dir = "D:\\RLHS-MITACS\\rack_detection\\trial\\data", output_dir = "D:\\RLHS-MITACS\\rack_detection\\out_dir", model = detect_roboflow):
    
    images = sorted(os.listdir(input_dir))
    for image in images:
        image_path = os.path.join(input_dir, image)
        
        if model == "detect_roboflow":
            prob = detect_roboflow(image_path)
        elif model == "detect_yolo":
            prob = detect_yolo(image_path)
            
        # if (prob):
        #     write_text_opencv(image_path, 'present'+ str(prob), os.path.join(output_dir, image))
        # else:
        #     write_text_opencv(image_path, 'absent', os.path.join(output_dir, image))
        
        
        
        # ------------------------cls_modls---------------------------------------------------------
        elif model == "roboflow_class":
            prob = detect_roboflow_class(image_path)
        
        elif model == "yolo_class":
            prob = detect_yolo_class(image_path)
        
        return prob
            
        # if (prob == 1):
        #     write_text_opencv(image_path, 'present', os.path.join(output_dir, image))
        # elif (prob == 0):
        #     write_text_opencv(image_path, 'absent', os.path.join(output_dir, image))


def process_single(image_path, model = "roboflow_class"):
    
    
        
    if model == "detect_roboflow":
        prob = detect_roboflow(image_path)
    elif model == "detect_yolo":
        prob = detect_yolo(image_path)
            
       
        
        
        # ------------------------cls_modls---------------------------------------------------------
    elif model == "roboflow_class":
        prob = detect_roboflow_class(image_path)
        
    elif model == "yolo_class":
        prob = detect_yolo_class(image_path)
        
    return prob
            
        # if (prob == 1):
        #     write_text_opencv(image_path, 'present', os.path.join(output_dir, image))
        # elif (prob == 0):
        #     write_text_opencv(image_path, 'absent', os.path.join(output_dir, image))     
            

# process("D:\\RLHS-MITACS\\rack_detection\\internet", "D:\\RLHS-MITACS\\rack_detection\\internet_out")
# detect("D:\\RLHS-MITACS\\rack_detection\\trial\data\\2.png")

    
    
print(process_single("D:\\RLHS-MITACS\\Data\\final_data\\11.png"))
    
    
# print(results[0].boxes.cls)

# for i,result in enumerate(results):
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     # result.show()  # display to screen
#     result.save(filename="D:/RLHS-MITACS/rack_detection/trial/output/" + "result" + str(i)+ ".jpg")


# out_dict = count_yolo_classes(classes, class_mapping)

# [ClassificationPrediction(class_name='bike_rack', class_id=0, confidence=0.9217), ClassificationPrediction(class_name='absent', class_id=2, confidence=0.0394), ClassificationPrediction(class_name='Unlabeled', class_id=1, confidence=0.0389)], top='bike_rack', confidence=0.9217, parent_id=None)]