import numpy as np
from ultralytics import YOLO
import cv2
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(script_directory)
video_path = ""
for f in os.listdir(script_directory):
    if '.mp4' in f:
        video_path = repr(f)
    print(repr(f))

# print("Script's Directory:", script_directory)

print(os.path.exists(video_path))

model = YOLO("yolov8n.pt")
 
results = model(video_path, show=True)
print("all done")