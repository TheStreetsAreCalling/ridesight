import cv2
import os
import math
import time
import numpy as np
import torch
import torchvision
import sys
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# =====================================================
# LOAD VIDEO + MODEL
# =====================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = "ads.mp4"
MAX_VIDEO_SIZE_MB = 25

if len(sys.argv) > 1:
    input_path = sys.argv[1]
    video_path = input_path if os.path.isabs(input_path) else os.path.join(script_dir, input_path)
else:
    video_path = os.path.join(script_dir, DEFAULT_VIDEO)

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
if video_size_mb > MAX_VIDEO_SIZE_MB:
    raise ValueError(
        f"Video is too large ({video_size_mb:.2f} MB). "
        f"Max supported size is {MAX_VIDEO_SIZE_MB} MB."
    )

print(f"Loading video: {video_path}")
print(f"Video size: {video_size_mb:.2f} MB (max {MAX_VIDEO_SIZE_MB} MB)")

# Load Faster R-CNN model
print("Loading Faster R-CNN ResNet50-FPN model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()
model.roi_heads.detections_per_img = 50

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("Model loaded successfully")

# COCO class names (Faster R-CNN uses COCO dataset classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"OpenCV could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

# =====================================================
# DRIVING INTERPRETER
# =====================================================

class DrivingInterpreter:
    def __init__(self):
        self.speed = 0.0
        self.target = 40.0
        self.max_accel = 8.0
        self.natural_decel = 1.5
        self.max_brake = 20.0
        self.accel_ema = 0.0
        self.ema_tau = 0.25
        self.speed_tau_accel = 0.12
        self.speed_tau_brake = 0.3

    def update(self, urgency, proximity, growth, shrink, red_light_stop, stop_sign_stop, dt):
        CLOSE_PROXIMITY = 0.45
        MIN_SPEED_NO_STOP = 5.0

        urgency_factor = max(0.0, urgency)
        prox_factor    = max(0.0, proximity)
        growth_factor  = max(0.0, min(1.0, growth))
        shrink_factor  = max(0.0, min(1.0, shrink))

        inhibition = urgency_factor * 0.7 + prox_factor * 2.0

        # Follow shrinking boxes: when object appears to move away (shrinking),
        # reduce inhibition so the car can keep accelerating to follow.
        follow_term = 0.35 * shrink_factor * (1.0 - prox_factor)
        inhibition -= follow_term

        inhibition  = max(0.0, min(inhibition, 1.0))

        # FIX: when braking hard, immediately zero the accel EMA so it
        #      doesn't fight against the brake force
        if inhibition > 0.6:
            self.accel_ema = 0.0

        if self.speed < self.target and inhibition < 1.0:
            desired_accel = self.max_accel * (1.0 - inhibition)
        else:
            desired_accel = 0.0

        alpha = 1.0 - math.exp(-dt / max(self.ema_tau, 1e-6)) if dt > 0 else 1.0
        self.accel_ema += alpha * (desired_accel - self.accel_ema)
        a = self.accel_ema

        # Braking is based primarily on box growth (approaching objects),
        # with a small proximity safety term. Red light / stop sign can hard-override.
        if red_light_stop or stop_sign_stop:
            self.accel_ema = 0.0
            brake_factor = 1.0
            prox_factor = max(prox_factor, CLOSE_PROXIMITY)
        else:
            brake_factor = growth_factor * 0.9 + prox_factor * 0.1
            brake_factor = max(0.0, min(brake_factor, 1.0))
        brake_force  = self.max_brake * brake_factor

        speed_change = a * dt - self.natural_decel * dt - brake_force * dt
        desired = self.speed + speed_change

        # Determine state
        if red_light_stop and self.speed < 1.0:
            state = "RED_LIGHT_STOP"
        elif red_light_stop:
            state = "RED_LIGHT_BRAKE"
        elif stop_sign_stop and self.speed < 1.0:
            state = "STOP_SIGN_WAIT"
        elif stop_sign_stop:
            state = "STOP_SIGN_BRAKE"
        elif brake_factor > 0.7:
            state = "EMERGENCY_BRAKE"
        elif brake_factor > 0.25:
            state = "BRAKING"
        elif a > 0.5 and desired > self.speed:
            state = "ACCELERATE"
        else:
            state = "CRUISE"

        tau = self.speed_tau_accel if desired > self.speed else max(0.05, self.speed_tau_brake * (1.0 - 0.6 * prox_factor))
        speed_alpha = 1.0 - math.exp(-dt / max(tau, 1e-6)) if dt > 0 else 1.0
        self.speed += speed_alpha * (desired - self.speed)
        self.speed = max(0.0, self.speed)

        # Keep rolling at low speed for ray-based braking unless we are truly too close
        # or a traffic control requires a stop.
        if (not red_light_stop) and (not stop_sign_stop) and (prox_factor < CLOSE_PROXIMITY):
            dynamic_min = MIN_SPEED_NO_STOP
            if self.speed < dynamic_min and self.speed > 0.1:
                self.speed = dynamic_min

        # Full stop condition
        if (prox_factor >= CLOSE_PROXIMITY or red_light_stop or stop_sign_stop) and self.speed < 1.0:
            self.speed = 0.0
            if red_light_stop:
                state = "RED_LIGHT_STOP"
            elif stop_sign_stop:
                state = "STOP_SIGN_WAIT"
            else:
                state = "EMERGENCY_STOP"

        print(f"[Speed]: {self.speed:.1f} mph  [State]: {state}  [Brake]: {brake_factor:.2f}  [Prox]: {prox_factor:.2f}")
        return self.speed, state


# =====================================================
# DETECTION BOXES
# =====================================================

tl_x_min = int(0.46 * w); tl_x_max = int(0.54 * w)
tl_y_min = int(0.03 * h); tl_y_max = int(0.75 * h)

ss_x_min = int(0.70 * w); ss_x_max = int(0.97 * w)
ss_y_min = int(0.03 * h); ss_y_max = int(0.75 * h)

ped_x_min = int(0.20 * w); ped_x_max = int(0.80 * w)
ped_y_min = int(0.15 * h); ped_y_max = int(0.60 * h)

IGNORE_Y = int(0.75 * h)

# =====================================================
# SENSOR RAYS
# =====================================================

origin     = (w // 2, int(0.35 * h))
ray_length = int(0.65 * h)

angles = {
    "far_left":   120,
    "left":       168,
    "center":     180,
    "right":     -168,
    "far_right": -120,
}

RAY_URGENCY_WEIGHT = {
    "left":      0.015,
    "center":    1.0,
    "right":     0.015,
    "far_left":  0.00005,
    "far_right": 0.00005,
}

STOP_PROX_RAYS = {"left", "center", "right"}

def make_ray(angle):
    r = math.radians(angle)
    return origin, (
        int(origin[0] + ray_length * math.sin(r)),
        int(origin[1] - ray_length * math.cos(r)),
    )

rays = {k: make_ray(v) for k, v in angles.items()}
FOCAL_T       = 0.45
RAY_TOLERANCE = int(0.05 * w)
PERSIST_FRAMES = 6
ray_persist   = {k: 0 for k in rays}

# Global sideways drift turn detection (optical flow)
FLOW_SCALE = 0.5
FLOW_MIN_MAG = 0.35
TURN_DRIFT_THRESHOLD = 1.4
DRIFT_EMA_ALPHA = 0.25

# =====================================================
# GEOMETRY HELPERS
# =====================================================

def compute_area(box):
    x1, y1, x2, y2 = box
    return max(0, (x2 - x1) * (y2 - y1))

RED_LIGHT_RATIO_THRESHOLD = 0.08

def red_light_ratio(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = float(cv2.countNonZero(red_mask))
    total_pixels = float(red_mask.shape[0] * red_mask.shape[1])
    if total_pixels <= 0:
        return 0.0
    return red_pixels / total_pixels

MATCH_DIST = int(0.10 * w)
GROWTH_SCALE = 3.0
SHRINK_SCALE = 2.0

def growth_score(area, prev_area, dt):
    if prev_area <= 0 or dt <= 0:
        return 0.0
    rel = (area - prev_area) / prev_area / dt
    if rel <= 0:
        return 0.0
    return min(1.0, rel * GROWTH_SCALE)

def shrink_score(area, prev_area, dt):
    if prev_area <= 0 or dt <= 0:
        return 0.0
    rel = (area - prev_area) / prev_area / dt
    if rel >= 0:
        return 0.0
    return min(1.0, (-rel) * SHRINK_SCALE)

# =====================================================
# RAY vs BOX intersection
# Returns (hit: bool, raw_t: float, hit_y: int)
#   raw_t  = parametric position along ray [0..1], 0=origin, 1=tip
#             used directly as "closeness" — no y_factor squish
#   hit_y  = pixel y of first intersection (for proximity scoring)
# =====================================================

def box_hits_ray(box, a, b, step=4):
    x1, y1, x2, y2 = box
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return False, None, None

    steps = max(int(length / step), 1)
    for i in range(steps + 1):
        t  = i / steps
        px = int(a[0] + dx * t)
        py = int(a[1] + dy * t)
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True, t, py

    return False, None, None


# =====================================================
# URGENCY from proximity along the ray
# t=0 means origin (car), t=1 means ray tip (far end).
# We want HIGH urgency when the object is CLOSE (low t).
# =====================================================

def urgency_from_t(t):
    """Convert raw ray-t to urgency score. Closer = higher score."""
    # Invert: t close to 0 → urgency near 1; t near 1 → urgency near 0
    return max(0.0, min(1.0, 1.0 - t))


# =====================================================
# MAIN LOOP
# =====================================================

interpreter         = DrivingInterpreter()
prev_time           = time.time()
show_boxes          = True
turn_indicator      = "STRAIGHT"
prev_traffic_x_avg  = None
turn_drift_ema      = 0.0
prev_detections     = []

# Stop-sign behavior: full stop for 2s, then wait until road is clear
STOP_SIGN_HOLD_SECONDS = 2.0
ROAD_CLEAR_PROX_THRESHOLD = 0.12
stop_sign_hold_until = 0.0
stop_sign_waiting_clear = False
stop_sign_armed = True
exit_requested = False

# Inference speed tuning
INFER_SCALE = 0.75
INFER_EVERY_N_FRAMES = 2
CONF_THRESHOLD = 0.35
frame_count = 0
last_predictions = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    now       = time.time()
    dt        = now - prev_time
    prev_time = now

    # Run Faster R-CNN inference (downscaled + optional frame skipping)
    run_inference = (last_predictions is None) or (frame_count % INFER_EVERY_N_FRAMES == 1)
    if run_inference:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if INFER_SCALE != 1.0:
            infer_rgb = cv2.resize(
                rgb_frame,
                None,
                fx=INFER_SCALE,
                fy=INFER_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            infer_rgb = rgb_frame

        input_tensor = F.to_tensor(infer_rgb).to(device)

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model([input_tensor])[0]
            else:
                pred = model([input_tensor])[0]

        if INFER_SCALE != 1.0:
            scale_back = 1.0 / INFER_SCALE
            pred["boxes"] = pred["boxes"] * scale_back

        last_predictions = pred

    predictions = last_predictions

    pedestrians       = []
    objects           = []
    pedestrian_in_box = False
    red_light_detected = False
    stop_sign_detected = False

    # Draw static detection boxes
    if show_boxes:
        cv2.rectangle(frame, (tl_x_min, tl_y_min), (tl_x_max, tl_y_max), (0, 255, 255), 2)
        cv2.rectangle(frame, (ss_x_min, ss_y_min), (ss_x_max, ss_y_max), (255, 0, 0),   2)
        cv2.rectangle(frame, (ped_x_min, ped_y_min), (ped_x_max, ped_y_max), (0, 255, 255), 2)
        cv2.line(frame, (0, IGNORE_Y), (w, IGNORE_Y), (0, 0, 255), 2)

    # Parse Faster R-CNN detections
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    # Filter by confidence threshold
    conf_threshold = CONF_THRESHOLD
    for box, label_id, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            break

        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        label = COCO_CLASSES[label_id]
        conf_text = f"{label} {score:.2f}"

        if show_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if cy > IGNORE_Y:
            continue

        if label in ("car", "truck", "bus", "person", "bicycle"):
            area = compute_area((x1, y1, x2, y2))
            if score >= 0.4 and area >= int(0.0005 * w * h):
                objects.append({
                    "cx": cx, "cy": cy,
                    "area": area,
                    "box": (x1, y1, x2, y2),
                    "conf": float(score),
                })

        if label == "person":
            pedestrians.append((cx, cy))
            if ped_x_min <= cx <= ped_x_max and ped_y_min <= cy <= ped_y_max:
                pedestrian_in_box = True

        if label == "traffic light" and score >= 0.35:
            x1c = max(0, min(w - 1, x1))
            y1c = max(0, min(h - 1, y1))
            x2c = max(0, min(w, x2))
            y2c = max(0, min(h, y2))
            crop = frame[y1c:y2c, x1c:x2c]
            rr = red_light_ratio(crop)
            in_tl_zone = (tl_x_min <= cx <= tl_x_max) and (tl_y_min <= cy <= tl_y_max)
            if rr >= RED_LIGHT_RATIO_THRESHOLD and in_tl_zone:
                red_light_detected = True
                if show_boxes:
                    cv2.putText(frame, "RED LIGHT", (x1, max(20, y1 - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if label == "stop sign" and score >= 0.35:
            in_ss_zone = (ss_x_min <= cx <= ss_x_max) and (ss_y_min <= cy <= ss_y_max)
            if in_ss_zone:
                stop_sign_detected = True
                if show_boxes:
                    cv2.putText(frame, "STOP SIGN", (x1, max(20, y1 - 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Trigger stop-sign routine on first detection while armed
    if stop_sign_detected and stop_sign_armed:
        stop_sign_hold_until = now + STOP_SIGN_HOLD_SECONDS
        stop_sign_waiting_clear = True
        stop_sign_armed = False

    # --------------------------------------------------
    # TURN DETECTION
    # --------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow_gray = cv2.resize(gray, (0, 0), fx=FLOW_SCALE, fy=FLOW_SCALE)

    turn_indicator = "STRAIGHT"
    if prev_traffic_x_avg is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_traffic_x_avg,
            flow_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        valid = mag > FLOW_MIN_MAG

        if np.any(valid):
            mean_dx = float(np.mean(fx[valid])) / FLOW_SCALE
        else:
            mean_dx = 0.0

        turn_drift_ema = (1.0 - DRIFT_EMA_ALPHA) * turn_drift_ema + DRIFT_EMA_ALPHA * mean_dx

        if turn_drift_ema > TURN_DRIFT_THRESHOLD:
            turn_indicator = "LEFT TURN"
        elif turn_drift_ema < -TURN_DRIFT_THRESHOLD:
            turn_indicator = "RIGHT TURN"

    prev_traffic_x_avg = flow_gray

    # =====================================================
    # RAY DETECTION
    # =====================================================
    max_urgency  = 0.0
    max_proximity = 0.0
    max_growth   = 0.0
    max_shrink   = 0.0

    ray_results = {}

    for name, (p1, p2) in rays.items():
        ray_urgency = 0.0
        hit         = False

        for obj in objects:
            did_hit, raw_t, hit_y = box_hits_ray(obj["box"], p1, p2)
            if did_hit:
                # estimate growth from nearest prior detection
                prev_area = 0.0
                best_d = None
                for pd in prev_detections:
                    d = math.hypot(obj["cx"] - pd["cx"], obj["cy"] - pd["cy"])
                    if best_d is None or d < best_d:
                        best_d = d
                        prev_area = pd["area"]
                if best_d is None or best_d > MATCH_DIST:
                    prev_area = 0.0

                g = growth_score(obj["area"], prev_area, dt)
                s = shrink_score(obj["area"], prev_area, dt)
                max_growth = max(max_growth, g)
                max_shrink = max(max_shrink, s)

                score = urgency_from_t(raw_t)
                ray_urgency = max(ray_urgency, score)
                hit = True

        if pedestrian_in_box:
            ray_urgency = max(ray_urgency, 0.50)
            hit = True

        # Persistence
        if hit:
            ray_persist[name] = PERSIST_FRAMES
        elif ray_persist[name] > 0:
            ray_persist[name] -= 1
            hit = True
            ray_urgency = max(ray_urgency, 0.1)

        weighted = ray_urgency * RAY_URGENCY_WEIGHT[name]
        max_urgency   = max(max_urgency,  weighted)
        if name in STOP_PROX_RAYS:
            max_proximity = max(max_proximity, ray_urgency)

        ray_results[name] = (hit, ray_urgency)

    # Draw rays
    for name, (p1, p2) in rays.items():
        hit, ray_urgency = ray_results[name]
        focal = (
            int(p1[0] + FOCAL_T * (p2[0] - p1[0])),
            int(p1[1] + FOCAL_T * (p2[1] - p1[1])),
        )

        if show_boxes:
            if hit and ray_urgency > 0.65:
                color = (0, 0, 255)
            elif hit and ray_urgency > 0.25:
                color = (0, 165, 255)
            elif hit:
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)

            cv2.line(frame, p1, p2, color, 3)
            cv2.circle(frame, focal, 5, (0, 255, 255), -1)

    # update temporal cache for next frame growth matching
    prev_detections = [{"cx": o["cx"], "cy": o["cy"], "area": o["area"]} for o in objects]

    # Stop-sign state machine
    road_clear = (max_proximity < ROAD_CLEAR_PROX_THRESHOLD) and (not pedestrian_in_box) and (not red_light_detected)

    if stop_sign_waiting_clear and now >= stop_sign_hold_until and road_clear:
        stop_sign_waiting_clear = False

    stop_sign_active = (now < stop_sign_hold_until) or stop_sign_waiting_clear

    # Re-arm once sign is gone and routine is complete
    if (not stop_sign_detected) and (not stop_sign_active):
        stop_sign_armed = True

    # =====================================================
    # CONTROL
    # =====================================================
    speed, state = interpreter.update(max_urgency, max_proximity, max_growth, max_shrink, red_light_detected, stop_sign_active, dt)

    # HUD
    state_color = (0, 255, 255)
    if state in ("EMERGENCY_STOP", "EMERGENCY_BRAKE"):
        state_color = (0, 0, 255)
    elif state == "BRAKING":
        state_color = (0, 165, 255)
    elif state == "ACCELERATE":
        state_color = (0, 255, 0)

    cv2.putText(frame, f"Speed: {speed:.1f} mph", (40,  40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"State: {state}",          (40,  80), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color,    2)
    cv2.putText(frame, f"Turn:  {turn_indicator}",  (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),  2)
    cv2.putText(frame, f"Prox:  {max_proximity:.2f}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Urgency: {max_urgency:.2f}", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Growth: {max_growth:.2f}", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Shrink: {max_shrink:.2f}", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"RedLight: {'YES' if red_light_detected else 'NO'}", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if red_light_detected else (200, 200, 200), 2)
    cv2.putText(frame, f"StopSign: {'WAIT' if stop_sign_active else 'NO'}", (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255) if stop_sign_active else (200, 200, 200), 2)
    cv2.putText(frame, "Model: Faster R-CNN", (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("ridesight rcnn", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        show_boxes = not show_boxes
    elif key == ord(" "):
        paused = True
        while paused:
            paused_frame = frame.copy()
            cv2.putText(
                paused_frame,
                "PAUSED - Press SPACE to resume",
                (40, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("ridesight rcnn", paused_frame)
            pause_key = cv2.waitKey(30) & 0xFF
            if pause_key == ord(" "):
                paused = False
                prev_time = time.time()
            elif pause_key == ord("q"):
                paused = False
                exit_requested = True
        if exit_requested:
            break

cap.release()
cv2.destroyAllWindows()
