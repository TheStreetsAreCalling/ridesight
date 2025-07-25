import numpy as np
from ultralytics import YOLO
import cv2
import os


vision_models = ["YOLO", "OPENCV", "DLIB"]

script_directory = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(script_directory)
video_path = ""
for f in os.listdir(script_directory):
    if '.mp4' in f:
        video_path = f
    print(repr(f))

print("Exists?", os.path.exists(os.path.join(script_directory, video_path)))
print("Using " + vision_models[0] + " on video: " + video_path)
      
model = YOLO("yolov8n.pt")
 
results = model(video_path, show=True)
results.save()
print("all done")