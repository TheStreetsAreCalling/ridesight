from ultralytics import YOLO
import cv2
import os
import math

# -----------------------------
# Locate video
# -----------------------------
script_directory = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_directory, "video1.mp4")
print(f"Using video: {video_path}")

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")
results = model(video_path, stream=True)

# -----------------------------
# Video properties
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out = cv2.VideoWriter(
    "result_with_midpoints.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

class_names = model.names

# =====================================================
# SCALED DETECTION BOXES
# =====================================================

# -----------------------------
# Traffic light box (thin, center)
# -----------------------------
tl_y_min = int(0.03 * h)
tl_y_max = int(0.75 * h)
tl_box_width = int(0.08 * w)

tl_x_center = w // 2
tl_x_min = tl_x_center - tl_box_width // 2
tl_x_max = tl_x_center + tl_box_width // 2

# -----------------------------
# Stop sign box (right side, wide)
# -----------------------------
ss_y_min = int(0.03 * h)
ss_y_max = int(0.75 * h)
ss_box_width = int(0.28 * w)
ss_margin = int(0.03 * w)

ss_x_max = w - ss_margin
ss_x_min = ss_x_max - ss_box_width

# -----------------------------
# Pedestrian awareness box (center, wide)
# -----------------------------
ped_y_min = int(0.15 * h)
ped_y_max = int(0.60 * h)
ped_box_width = int(0.60 * w)

ped_x_center = w // 2
ped_x_min = max(0, ped_x_center - ped_box_width // 2)
ped_x_max = min(w, ped_x_center + ped_box_width // 2)

# =====================================================
# SENSOR RAYS
# =====================================================
origin = (w // 2, int(0.9 * h))
ray_length = int(0.65 * h)

angles_deg = {"left": 10, "center": 0, "right": -10}

def make_ray(angle):
    rad = math.radians(angle)
    return origin, (
        int(origin[0] + ray_length * math.sin(rad)),
        int(origin[1] - ray_length * math.cos(rad))
    )

sensor_rays = {k: make_ray(v) for k, v in angles_deg.items()}

# -----------------------------
# Focal + urgency parameters
# -----------------------------
FOCAL_T = 0.45
VEHICLE_RADIUS = int(0.13 * h)
PEDESTRIAN_RADIUS = int(0.18 * h)

# -----------------------------
# Temporal persistence
# -----------------------------
PERSIST_FRAMES = 10
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
        return math.hypot(px - ax, py - ay), 0.0

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t

# =====================================================
# PROCESS FRAMES
# =====================================================
for r in results:
    frame = r.plot()

    stop_sign_detected = False
    pedestrian_in_zone = False
    vehicles, pedestrians = [], []

    # Draw static boxes
    cv2.rectangle(frame, (tl_x_min, tl_y_min), (tl_x_max, tl_y_max), (0, 255, 255), 2)
    cv2.rectangle(frame, (ped_x_min, ped_y_min), (ped_x_max, ped_y_max), (0, 255, 255), 2)

    if r.boxes is not None:
        boxes_xywh = r.boxes.xywh.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        for (cx, cy, _, _), cls_id in zip(boxes_xywh, class_ids):
            label = class_names[cls_id]
            cx_i, cy_i = int(cx), int(cy)

            cv2.circle(frame, (cx_i, cy_i), 4, (0, 0, 255), -1)

            # Traffic light
            if label == "traffic light":
                inside = tl_x_min <= cx_i <= tl_x_max and tl_y_min <= cy_i <= tl_y_max
                color = (255, 0, 255) if inside else (0, 255, 0)
                cv2.putText(frame, "TL", (cx_i + 6, cy_i - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Stop sign
            if label == "stop sign":
                inside = ss_x_min <= cx_i <= ss_x_max and ss_y_min <= cy_i <= ss_y_max
                stop_sign_detected |= inside
                cv2.putText(frame, "STOP", (cx_i + 6, cy_i - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0) if inside else (0, 0, 255), 2)

            # Pedestrian
            if label == "person":
                inside = ped_x_min <= cx_i <= ped_x_max and ped_y_min <= cy_i <= ped_y_max
                if inside:
                    pedestrian_in_zone = True
                    cv2.putText(frame, "PED", (cx_i + 6, cy_i - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                pedestrians.append((cx_i, cy_i))

            # Vehicles
            if label in ("car", "truck", "bus") and cy_i < int(0.35 * h):
                vehicles.append((cx_i, cy_i))

    # Stop sign box
    cv2.rectangle(
        frame,
        (ss_x_min, ss_y_min),
        (ss_x_max, ss_y_max),
        (255, 0, 0) if stop_sign_detected else (0, 0, 255),
        2
    )

    # Pedestrian box feedback
    if pedestrian_in_zone:
        cv2.rectangle(frame, (ped_x_min, ped_y_min), (ped_x_max, ped_y_max), (0, 255, 0), 3)

    # -----------------------------
    # SENSOR RAY LOGIC
    # -----------------------------
    for name, (p1, p2) in sensor_rays.items():
        fp = (
            int(p1[0] + FOCAL_T * (p2[0] - p1[0])),
            int(p1[1] + FOCAL_T * (p2[1] - p1[1]))
        )

        detected = False
        urgency = 0.0

        for mid in vehicles:
            dist, t = point_to_segment_distance(mid, p1, p2)
            if dist < int(0.05 * w) and t > FOCAL_T:
                d = math.hypot(mid[0] - fp[0], mid[1] - fp[1])
                if d < VEHICLE_RADIUS:
                    urgency = max(urgency, 1.0 - d / VEHICLE_RADIUS)
                    detected = True

        for mid in pedestrians:
            dist, t = point_to_segment_distance(mid, p1, p2)
            if dist < int(0.07 * w) and t > FOCAL_T:
                d = math.hypot(mid[0] - fp[0], mid[1] - fp[1])
                if d < PEDESTRIAN_RADIUS:
                    urgency = max(urgency, 0.7 * (1.0 - d / PEDESTRIAN_RADIUS))
                    detected = True

        if pedestrian_in_zone:
            urgency = max(urgency, 0.35)
            detected = True

        ray_persistence[name] = ray_persistence[name] + 1 if detected else 0

        if ray_persistence[name] >= PERSIST_FRAMES:
            color = (0, 255, 255) if urgency < 0.4 else (0, 0, int(255 * urgency))
        else:
            color = (255, 0, 0)

        cv2.line(frame, p1, p2, color, 3)
        cv2.circle(frame, fp, 5, (0, 255, 255), -1)

    cv2.circle(frame, origin, 6, (255, 255, 255), -1)

    cv2.imshow("ridesight", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
out.release()
cv2.destroyAllWindows()
print("Saved result_with_midpoints.mp4")
