import cv2
import torch
import torchvision
import math
import os
import numpy as np

# -----------------------------
# Paths
# -----------------------------
script_directory = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_directory, "video1.mp4")
print(f"Using video: {video_path}")

# -----------------------------
# Load Faster R-CNN
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign"
]

# -----------------------------
# Video I/O
# -----------------------------
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Failed to open video"

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "result_faster_rcnn_fullscan.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# -----------------------------
# Detection boxes
# -----------------------------
# Traffic lights 
tl_y_min, tl_y_max = 20, 700
tl_box_width = int(0.08 * w)
tl_x_min = w // 2 - tl_box_width // 2
tl_x_max = w // 2 + tl_box_width // 2

# Stop signs (right, wider)
ss_y_min, ss_y_max = 20, 700
ss_box_width = int(0.28 * w)
ss_x_max = w - 20
ss_x_min = ss_x_max - ss_box_width

# Pedestrian zone 
ped_y_min = int(0.15 * h)
ped_y_max = int(0.6 * h)
ped_box_width = int(0.75 * w)
ped_x_min = w // 2 - ped_box_width // 2
ped_x_max = w // 2 + ped_box_width // 2

# -----------------------------
# Sensor rays (±10°)
# -----------------------------
origin = (w // 2, int(0.9 * h))
ray_length = int(0.65 * h)

angles = {"left": 10, "center": 0, "right": -10}

def make_ray(angle):
    rad = math.radians(angle)
    return origin, (
        int(origin[0] + ray_length * math.sin(rad)),
        int(origin[1] - ray_length * math.cos(rad))
    )

sensor_rays = {k: make_ray(v) for k, v in angles.items()}

# -----------------------------
# Parameters
# -----------------------------
CONF_THRESH = 0.6
PERSIST_FRAMES = 10
FOCAL_T = 0.45
VEHICLE_RADIUS = 140
PEDESTRIAN_RADIUS = 200

ray_persistence = {k: 0 for k in sensor_rays}

# -----------------------------
# Geometry helper
# -----------------------------
def point_to_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay

    if dx == dy == 0:
        return math.hypot(px - ax, py - ay), 0

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))

    cx = ax + t * dx
    cy = ay + t * dy

    return math.hypot(px - cx, py - cy), t

# -----------------------------
# MAIN LOOP
# -----------------------------
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Convert frame → tensor (EVERY FRAME)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    # Inference (EVERY FRAME)
    with torch.no_grad():
        output = model([img_tensor])[0]

    vehicles = []
    pedestrians = []
    stop_detected = False
    ped_zone = False

    # Draw static boxes
    cv2.rectangle(frame, (tl_x_min, tl_y_min), (tl_x_max, tl_y_max), (0, 255, 255), 2)
    cv2.rectangle(frame, (ped_x_min, ped_y_min), (ped_x_max, ped_y_max), (0, 255, 255), 2)

    # -----------------------------
    # Parse detections
    # -----------------------------
    for box, label_id, score in zip(
        output["boxes"],
        output["labels"],
        output["scores"]
    ):
        if score < CONF_THRESH:
            continue

        x1, y1, x2, y2 = box.int().cpu().numpy()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        label = COCO_CLASSES[label_id]
        conf_txt = f"{label} {int(score * 100)}%"

        # Draw bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.putText(
            frame,
            conf_txt,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        # Traffic lights
        if label == "traffic light":
            inside = tl_x_min <= cx <= tl_x_max and tl_y_min <= cy <= tl_y_max
            cv2.putText(
                frame, "TL",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255) if inside else (0, 255, 0),
                2
            )

        # Stop sign
        if label == "stop sign":
            if ss_x_min <= cx <= ss_x_max and ss_y_min <= cy <= ss_y_max:
                stop_detected = True

        # Pedestrians
        if label == "person":
            pedestrians.append((cx, cy))
            if ped_x_min <= cx <= ped_x_max and ped_y_min <= cy <= ped_y_max:
                ped_zone = True

        # Vehicles
        if label in ("car", "truck", "bus") and cy < int(0.35 * h):
            vehicles.append((cx, cy))

    # Stop sign box
    cv2.rectangle(
        frame,
        (ss_x_min, ss_y_min),
        (ss_x_max, ss_y_max),
        (255, 0, 0) if stop_detected else (0, 0, 255),
        2
    )

    # -----------------------------
    # Sensor ray logic
    # -----------------------------
    for name, (p1, p2) in sensor_rays.items():
        fp = (
            int(p1[0] + FOCAL_T * (p2[0] - p1[0])),
            int(p1[1] + FOCAL_T * (p2[1] - p1[1]))
        )

        detected = False
        urgency = 0.0

        for mid in vehicles + pedestrians:
            dist, t = point_to_segment_distance(mid, p1, p2)
            if dist < 45 and t > FOCAL_T:
                d = math.hypot(mid[0] - fp[0], mid[1] - fp[1])
                radius = PEDESTRIAN_RADIUS if mid in pedestrians else VEHICLE_RADIUS
                if d < radius:
                    urgency = max(urgency, 1.0 - d / radius)
                    detected = True

        ray_persistence[name] = ray_persistence[name] + 1 if detected else 0

        if ray_persistence[name] >= PERSIST_FRAMES:
            color = (0, 255, 255) if urgency < 0.4 else (0, 0, int(255 * urgency))
        else:
            color = (255, 0, 0)

        cv2.line(frame, p1, p2, color, 3)
        cv2.circle(frame, fp, 5, (0, 255, 255), -1)

    cv2.circle(frame, origin, 6, (255, 255, 255), -1)

    out.write(frame)
    cv2.imshow("ridesight rcnn", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("Saved result_faster_rcnn_fullscan.mp4")


