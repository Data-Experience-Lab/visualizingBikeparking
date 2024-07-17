import inference
import supervision as sv

model = inference.get_model(model_id="bike-rack-unayc/2", api_key="ZVyfCUbZT6qdSHQPyazV")
results = model.infer(image="D:\\RLHS-MITACS\\rack_detection\\internet\\input\\8a7592d84fd2db7030551d7243170567.jpg")
detections = sv.Detections.from_inference(results[0])


print(detections)