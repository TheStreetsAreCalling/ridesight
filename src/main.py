import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
 
results = model("../traindata/video1.mp4", show=True)
print("all done")