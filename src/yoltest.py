from ultralytics import YOLO
import cv2
import os
import math
import time


# DRIVING INTERPRETER


class DrivingInterpreter:
    def __init__(self):
        self.target_speed = 30.0  
        self.current_speed = 0.0
        self.max_accel = 3.0      
        self.max_brake = 8.0      
        self.state = "CRUISE"

    def update(self, detections, dt):

        throttle = 0
        brake = 0
        self.state = "CRUISE"

        # Maintain 30 mph
        if self.current_speed < self.target_speed:
            throttle = 1

        #  Traffic Light 
        if detections["traffic_light"]["present"]:
            color = detections["traffic_light"]["color"]

            if color == "red":
                brake = 1
                throttle = 0
                self.state = "STOP_RED"

            elif color == "yellow":
                brake = 0.6
                throttle = 0
                self.state = "SLOW_YELLOW"

            elif color == "green":
                throttle = 1
                self.state = "GO_GREEN"

        #  Car Ahead 
        if detections["car"]["present"]:
            brake = 0.5
            throttle = 0
            self.state = "APPROACH_CAR"

            if detections["car"]["near"]:
                brake = 1
                self.state = "STOP_CAR"

        #  Pedestrian 
        if detections["person"]["present"]:
            brake = 0.7
            throttle = 0
            self.state = "SLOW_PEDESTRIAN"

            if detections["person"]["near"]:
                brake = 1
                self.state = "STOP_PEDESTRIAN"

        #  Bicycle 
        if detections["bicycle"]["present"]:
            brake = 0.5
            throttle = 0
            if detections["bicycle"]["near"]:
                brake = 1
                self.state = "STOP_BICYCLE"

        #  Train 
        if detections["train"]["near"]:
            brake = 1
            throttle = 0
            self.state = "STOP_TRAIN"

        # Barrier 
        if detections["barrier"]["present"]:
            brake = 0.7
            throttle = 0
            if detections["barrier"]["near"]:
                brake = 1
                self.state = "STOP_BARRIER"

        #  Stop Sign 
        if detections["stop_sign"]["present"]:
            brake = 0.6
            throttle = 0
            if detections["stop_sign"]["near"]:
                brake = 1
                self.state = "STOP_SIGN"

        #  Physics 
        if brake > 0:
            self.current_speed -= self.max_brake * brake * dt
        elif throttle > 0:
            self.current_speed += self.max_accel * throttle * dt

        self.current_speed = max(0, min(self.current_speed, 35))

        return {
            "speed": self.current_speed,
            "state": self.state
        }


# LOAD VIDEO + MODEL


script_directory = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_directory, "video2.mp4")

model = YOLO("yolov8n.pt")
results = model(video_path, stream=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()


# DETECTION ZONES (SCALED)


# Traffic Light Zone
tl_x_min = int(0.46 * w)
tl_x_max = int(0.54 * w)
tl_y_min = int(0.03 * h)
tl_y_max = int(0.75 * h)

# Stop Sign Zone (right side, wide)
ss_x_max = int(0.97 * w)
ss_x_min = int(0.70 * w)
ss_y_min = int(0.03 * h)
ss_y_max = int(0.75 * h)

# Pedestrian Mid Range Box
ped_x_min = int(0.20 * w)
ped_x_max = int(0.80 * w)
ped_y_min = int(0.15 * h)
ped_y_max = int(0.60 * h)

# =====================================================
# SENSOR RAYS
# =====================================================

origin = (w // 2, int(0.9 * h))
ray_length = int(0.65 * h)

angles = {"left": 10, "center": 0, "right": -10}

def make_ray(angle):
    rad = math.radians(angle)
    return origin, (
        int(origin[0] + ray_length * math.sin(rad)),
        int(origin[1] - ray_length * math.cos(rad))
    )

rays = {k: make_ray(v) for k, v in angles.items()}

# =====================================================
# INITIALIZE INTERPRETER
# =====================================================

interpreter = DrivingInterpreter()
previous_time = time.time()

# =====================================================
# MAIN LOOP
# =====================================================

class_names = model.names

for r in results:
    frame = r.plot()

    current_time = time.time()
    dt = current_time - previous_time
    previous_time = current_time

    vehicles = []
    pedestrians = []
    stop_sign_detected = False

    traffic_light_present = False
    traffic_light_color = "green"

    if r.boxes is not None:
        boxes = r.boxes.xywh.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for (cx, cy, _, _), cls in zip(boxes, classes):
            label = class_names[cls]
            cx_i, cy_i = int(cx), int(cy)

            # Traffic Light
            if label == "traffic light":
                traffic_light_present = True
                traffic_light_color = "red"  # placeholder until color classifier added

            # Stop Sign
            if label == "stop sign":
                if ss_x_min <= cx_i <= ss_x_max:
                    stop_sign_detected = True

            # Vehicles
            if label in ["car", "truck", "bus"]:
                vehicles.append((cx_i, cy_i))

            # Pedestrian
            if label == "person":
                pedestrians.append((cx_i, cy_i))

    pedestrian_near = any(
        ped_x_min <= x <= ped_x_max and ped_y_min <= y <= ped_y_max
        for x, y in pedestrians
    )

    car_near = len(vehicles) > 0

    detections = {
        "traffic_light": {"present": traffic_light_present, "color": traffic_light_color},
        "car": {"present": len(vehicles) > 0, "near": car_near},
        "person": {"present": len(pedestrians) > 0, "near": pedestrian_near},
        "bicycle": {"present": False, "near": False},
        "train": {"near": False},
        "barrier": {"present": False, "near": False},
        "stop_sign": {"present": stop_sign_detected, "near": stop_sign_detected}
    }

    control = interpreter.update(detections, dt)

    # =====================================================
    # DISPLAY OVERLAYS
    # =====================================================

    cv2.putText(frame, f"Speed: {control['speed']:.1f} mph",
                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"State: {control['state']}",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.rectangle(frame, (ped_x_min, ped_y_min),
                  (ped_x_max, ped_y_max), (0,255,255), 2)

    cv2.rectangle(frame, (ss_x_min, ss_y_min),
                  (ss_x_max, ss_y_max),
                  (255,0,0) if stop_sign_detected else (0,0,255), 2)

    for p1, p2 in rays.values():
        cv2.line(frame, p1, p2, (255,0,0), 2)

    cv2.imshow("Autonomous Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# 3D model
from ursina import *

app = Ursina()

DirectionalLight().look_at(Vec3(1,-1,-1))
AmbientLight(color=color.rgba(255, 255, 255, 0.8))

ground = Entity(model='plane', scale=50, texture='white_cube',
                texture_scale=(50,50), color=color.light_gray)

car = Entity(model='car.obj', scale=0.45, position=(-0.8,0,0))

# Camera settings
camera.position = (0, 5, -12)   # x, y, z
camera.rotation_x = 20          # tilt downward
camera.rotation_y = 0           # facing forward

def update():
    # Smooth follow
    pass

app.run()
