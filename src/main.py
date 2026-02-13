import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import cv2
import os


vision_models = ["YOLO", "OPENCV", "DLIB"]

# Extract video path from current directory
script_directory = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(script_directory)
video_path = "video1.mp4"
for f in contents:
    if f.endswith('.mp4'):
        video_path = os.path.join(script_directory, f)
    print(repr(f))

if not video_path:
    raise FileNotFoundError(f"No .mp4 video found in {script_directory}")

print("Exists?", os.path.exists(video_path))
print("Using " + vision_models[0] + " on video: " + video_path)
      
# Load the YOLOv8n model (use local file if present)
model_path = os.path.join(script_directory, "yolov8n.pt")
model = YOLO(model_path)
results = model(video_path, show=True)

# Save the results to a file TODO

for r in results:
    print("Detected " + str(len(r.boxes)) + " objects in frame ")
results.save()
print("all done")

