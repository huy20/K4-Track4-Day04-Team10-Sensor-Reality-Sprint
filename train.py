from ultralytics import YOLO
import yaml

model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/ADAS-1/data.yaml',
    epochs=10, 
    imgsz=640,
    batch=16,
    device= 0
)