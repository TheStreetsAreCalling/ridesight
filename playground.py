from ultralytics import YOLO
import cv2
import os
import math
import time

# =====================================================
# LOAD VIDEO + MODEL
# =====================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "video2.mp4")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =====================================================
# DRIVING INTERPRETER
# =====================================================

class DrivingInterpreter:
    def __init__(self):
        self.speed = 0.0
        self.target = 40.0  # target cruising speed
        self.max_accel = 8.0  # mph/sec
        self.natural_decel = 1.5  # mph/sec (always active)
        self.max_brake = 12.0  # additional braking force (mph/sec) when obstacles detected
        # smoothing state to avoid jumping between speeds
        self.accel_ema = 0.0
        self.ema_tau = 0.25  # seconds for acceleration smoothing
        # asymmetric speed smoothing: faster when accelerating
        self.speed_tau_accel = 0.12
        self.speed_tau_brake = 0.5

    def update(self, urgency, proximity, dt):
        # proximity: 0..1 (1 is very close). Only allow full stop when proximity >= CLOSE_PROXIMITY
        CLOSE_PROXIMITY = 0.5
        MIN_SPEED_NO_STOP = 5.0

        # compute acceleration from urgency/proximity
        # high urgency/proximity -> low acceleration; zero urgency/proximity -> maintain target
        urgency_factor = max(0, urgency)  # clamp to [0, 1]
        prox_factor = max(0, proximity)  # clamp to [0, 1]
        
        # combined inhibition: proximity has stronger effect on braking
        inhibition = urgency_factor * 0.7 + prox_factor * 2.0
        inhibition = max(0, min(inhibition, 1.0))  # clamp to [0, 1]
        
        # compute base acceleration
        if self.speed < self.target and inhibition < 1.0:
            # below target: accelerate towards it (must overcome natural deceleration)
            desired_accel = self.max_accel * (1.0 - inhibition)
        else:
            # at/above target or full inhibition: let natural deceleration take effect
            # no additional acceleration, so speed naturally decreases
            desired_accel = 0.0
        
        # smooth acceleration (exponential moving average)
        if dt > 0:
            alpha = 1 - math.exp(-dt / max(self.ema_tau, 1e-6))
        else:
            alpha = 1.0
        self.accel_ema += alpha * (desired_accel - self.accel_ema)
        a = self.accel_ema

        # physics: speed change = acceleration - natural_deceleration - brake_force
        # brake force scales with urgency and proximity
        brake_factor = urgency_factor * 0.5 + prox_factor * 3.0
        brake_factor = max(0, min(brake_factor, 1.0))  # clamp to [0, 1]
        brake_force = self.max_brake * brake_factor
        
        accel_component = a * dt
        decel_component = self.natural_decel * dt
        brake_component = brake_force * dt
        speed_change = accel_component - decel_component - brake_component
        
        desired = self.speed + speed_change

        # determine state
        if a > 0:
            if desired > self.speed:
                state = "ACCELERATE"
            else:
                state = "CRUISE"
        else:
            state = "DECELERATE"

        # low-pass filter the speed change to avoid jumps
        if desired > self.speed:
            tau = self.speed_tau_accel
        else:
            tau = max(0.08, self.speed_tau_brake * (1.0 - 0.5 * proximity))

        if dt > 0:
            speed_alpha = 1 - math.exp(-dt / max(tau, 1e-6))
        else:
            speed_alpha = 1.0
        self.speed += speed_alpha * (desired - self.speed)

        # clamp to [0, inf) - no upper speed limit
        self.speed = max(0, self.speed)

        # minimum speed scales with proximity: high proximity allows lower speeds
        # at low proximity: maintain MIN_SPEED_NO_STOP
        # at high proximity: allow speed to drop to 0
        dynamic_min_speed = MIN_SPEED_NO_STOP * (1.0 - proximity)
        if self.speed < dynamic_min_speed:
            self.speed = dynamic_min_speed

        # emergency state if we actually stopped due to very close object
        if self.speed <= 0 and proximity >= CLOSE_PROXIMITY:
            state = "EMERGENCY_STOP"

        return self.speed, state

# =====================================================
# DETECTION BOXES
# =====================================================

tl_x_min = int(0.46 * w)
tl_x_max = int(0.54 * w)
tl_y_min = int(0.03 * h)
tl_y_max = int(0.75 * h)

ss_x_min = int(0.70 * w)
ss_x_max = int(0.97 * w)
ss_y_min = int(0.03 * h)
ss_y_max = int(0.75 * h)

ped_x_min = int(0.20 * w)
ped_x_max = int(0.80 * w)
ped_y_min = int(0.15 * h)
ped_y_max = int(0.60 * h)

IGNORE_Y = int(0.75 * h)

# =====================================================
# SENSOR RAYS
# =====================================================

origin = (w // 2, int(0.88 * h))
ray_length = int(0.65 * h)

angles = {"left": 12, "center": 0, "right": -12}

# Urgency weighting per ray (side rays deprioritized)
RAY_URGENCY_WEIGHT = {
    "left": 0.25,
    "center": 1.0,
    "right": 0.25
}

# =====================================================
# TURN DETECTION (traffic lights / stop signs)
# =====================================================

TURN_X_THRESHOLD = int(0.15 * w)  # pixels: threshold for detecting turn
prev_traffic_x_avg = None  # average x position of traffic lights/stop signs
def make_ray(angle):
    r = math.radians(angle)
    return origin, (
        int(origin[0] + ray_length * math.sin(r)),
        int(origin[1] - ray_length * math.cos(r))
    )

rays = {k: make_ray(v) for k, v in angles.items()}

FOCAL_T = 0.45
RAY_TOLERANCE = int(0.05 * w)
PERSIST_FRAMES = 6
ray_persist = {k: 0 for k in rays}

# =====================================================
# GEOMETRY
# =====================================================

def point_to_ray(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return False, None

    t = ((px - ax)*dx + (py - ay)*dy) / l2
    if t < 0 or t > 1:
        return False, None

    cx = ax + t*dx
    cy = ay + t*dy
    d = math.hypot(px - cx, py - cy)
    return d < RAY_TOLERANCE, t

# =====================================================
# SIZE / GROWTH-BASED URGENCY HELPERS
# =====================================================

# Normalized area thresholds (area / (w*h))
SIZE_SMALL = 0.001   # below this is considered small
SIZE_LARGE = 0.010   # above this is considered large
GROWTH_FACTOR = 5.0  # scales growth ratio into [0,1]
MATCH_DIST = int(0.10 * w)  # pixels: max distance to match objects across frames
CONF_MIN = 0.40
MIN_AREA_NORM = 0.0005
MIN_AREA_PIXELS = int(MIN_AREA_NORM * w * h)

def compute_area(box):
    x1, y1, x2, y2 = box
    return max(0, (x2 - x1) * (y2 - y1))

def size_score_from_area(area):
    norm = area / (w * h)
    if norm <= SIZE_SMALL:
        return 0.0
    if norm >= SIZE_LARGE:
        return 1.0
    return (norm - SIZE_SMALL) / (SIZE_LARGE - SIZE_SMALL)

def growth_score_from_prev(area, prev_area, dt):
    if prev_area <= 0 or dt <= 0:
        return 0.0
    # relative growth per second
    rel = (area - prev_area) / prev_area / dt
    gs = max(0.0, rel * GROWTH_FACTOR)
    return min(gs, 1.0)

# =====================================================
# MAIN LOOP
# =====================================================

interpreter = DrivingInterpreter()
prev_time = time.time()
class_names = model.names
# simple temporal cache of previous detections: list of dicts with cx,cy,area
prev_detections = []
# toggle visibility of detection boxes
show_boxes = True
# turn detection state
turn_indicator = "STRAIGHT"
prev_traffic_x_avg = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - prev_time
    prev_time = now

    result = model(frame, conf=0.35, verbose=False)[0]

    pedestrians = []
    objects = []
    traffic_signs = []  # traffic lights and stop signs for turn detection
    pedestrian_in_box = False

    # Draw static detection boxes (if visible)
    if show_boxes:
        cv2.rectangle(frame, (tl_x_min, tl_y_min), (tl_x_max, tl_y_max), (0,255,255), 2)
        cv2.rectangle(frame, (ss_x_min, ss_y_min), (ss_x_max, ss_y_max), (255,0,0), 2)
        cv2.rectangle(frame, (ped_x_min, ped_y_min), (ped_x_max, ped_y_max), (0,255,255), 2)
        cv2.line(frame, (0, IGNORE_Y), (w, IGNORE_Y), (0,0,255), 2)

    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_xywh = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), (cx, cy, _, _), cls, conf in zip(
            boxes_xyxy, boxes_xywh, classes, confidences
        ):
            cx, cy = int(cx), int(cy)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            label = class_names[cls]
            conf_text = f"{label} {conf:.2f}"

            # Draw YOLO bounding box (if visible)
            if show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, conf_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # Draw midpoint
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

            if cy > IGNORE_Y:
                continue

            if label in ("car", "truck", "bus", "person", "bicycle"):
                area = compute_area((x1, y1, x2, y2))
                # filter out tiny boxes and low-confidence detections
                if conf >= CONF_MIN and area >= MIN_AREA_PIXELS:
                    objects.append({
                        "cx": cx,
                        "cy": cy,
                        "area": area,
                        "box": (x1, y1, x2, y2),
                        "conf": float(conf),
                    })

            if label == "person":
                pedestrians.append((cx, cy))
                if ped_x_min <= cx <= ped_x_max and ped_y_min <= cy <= ped_y_max:
                    pedestrian_in_box = True

            # collect traffic lights and stop signs for turn detection
            if label in ("traffic light", "stop sign"):
                traffic_signs.append(cx)

    # =====================================================
    # TURN DETECTION
    # =====================================================

    turn_indicator = "STRAIGHT"
    if traffic_signs:
        traffic_x_avg = sum(traffic_signs) / len(traffic_signs)
        if prev_traffic_x_avg is not None:
            x_delta = traffic_x_avg - prev_traffic_x_avg
            if x_delta > TURN_X_THRESHOLD:
                turn_indicator = "LEFT TURN"
            elif x_delta < -TURN_X_THRESHOLD:
                turn_indicator = "RIGHT TURN"
        prev_traffic_x_avg = traffic_x_avg
    else:
        prev_traffic_x_avg = None

    # =====================================================
    # RAY DETECTION
    # =====================================================

    max_urgency = 0.0
    max_proximity = 0.0

    for name, (p1, p2) in rays.items():
        hit = False
        urgency = 0.0

        focal = (
            int(p1[0] + FOCAL_T*(p2[0]-p1[0])),
            int(p1[1] + FOCAL_T*(p2[1]-p1[1]))
        )

        for obj in objects:
            ox, oy = obj["cx"], obj["cy"]
            on_ray, t = point_to_ray(ox, oy, p1, p2)
            if on_ray and t > FOCAL_T:
                # base proximity urgency (closer along the ray -> larger u)
                u = (t - FOCAL_T) / (1 - FOCAL_T)

                # match to previous detection (simple nearest match)
                prev_area = 0.0
                best_d = None
                for pd in prev_detections:
                    d = math.hypot(ox - pd["cx"], oy - pd["cy"])
                    if best_d is None or d < best_d:
                        best_d = d
                        prev_area = pd.get("area", 0.0)

                if best_d is None or best_d > MATCH_DIST:
                    prev_area = 0.0

                area = obj.get("area", 0.0)
                ss = size_score_from_area(area)
                gs = growth_score_from_prev(area, prev_area, dt)

                # object scale: use size score so tiny objects contribute near-zero
                obj_scale = ss * (1.0 + gs)
                if obj_scale > 0:
                    urgency = max(urgency, u * u * obj_scale)
                max_proximity = max(max_proximity, u)
                hit = True

        if pedestrian_in_box:
            urgency = max(urgency, 0.35)
            hit = True

        ray_persist[name] = ray_persist[name] + 1 if hit else 0

        # apply per-ray weight when combining rays
        weighted_urgency = urgency * RAY_URGENCY_WEIGHT[name]
        max_urgency = max(max_urgency, weighted_urgency)

        color = (0,255,255) if ray_persist[name] >= PERSIST_FRAMES else (255,0,0)
        if urgency > 0.6:
            color = (0,0,255)

        if show_boxes:
            cv2.line(frame, p1, p2, color, 3)
            cv2.circle(frame, focal, 5, (0,255,255), -1)

    # update temporal cache of detections for next frame
    prev_detections = [{"cx": o["cx"], "cy": o["cy"], "area": o["area"]} for o in objects]

    # =====================================================
    # CONTROL
    # =====================================================

    speed, state = interpreter.update(max_urgency, max_proximity, dt)

    cv2.putText(frame, f"Speed: {speed:.1f} mph", (40,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(frame, f"State: {state}", (40,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Turn: {turn_indicator}", (40,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("ridesight", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        show_boxes = not show_boxes

cap.release()
cv2.destroyAllWindows()
