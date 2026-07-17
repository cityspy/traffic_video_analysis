import cv2
import csv
import math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# Paths
# =========================
video_path = r"C:\Users\CodyMa\Documents\traffic_video_analysis\videos\heaslip_rd_before.mp4"
output_frame_csv = r"C:\Users\CodyMa\Documents\traffic_video_analysis\vehicle_counts_per_frame.csv"
output_minute_csv = r"C:\Users\CodyMa\Documents\traffic_video_analysis\vehicle_counts_per_minute.csv"

# =========================
# Model
# =========================
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

frame_idx = 0

vehicle_classes = [2, 3, 5, 7]
class_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# =====================================
# ROI polygons - final tuned + merge buffer
# =====================================

# 1) NB mainline
NB_MAIN_POLYGON = np.array([
    (260, 760),
    (760, 760),
    (860, 610),
    (890, 500),
    (900, 390),
    (760, 250),
    (590, 210),
    (515, 250),
    (575, 320),
    (600, 430),
    (520, 560),
    (410, 690)
], dtype=np.int32)

# 2) NB ramp
NB_RAMP_POLYGON = np.array([
    (0, 760),
    (260, 760),
    (410, 690),
    (520, 560),
    (600, 430),
    (575, 320),
    (515, 250),
    (390, 290),
    (245, 420),
    (120, 560),
    (40, 680)
], dtype=np.int32)

# 3) SB mainline
SB_MAIN_POLYGON = np.array([
    (760, 760),
    (1460, 760),
    (1580, 610),
    (1580, 170),
    (1230, 150),
    (1010, 235),
    (900, 390),
    (890, 500),
    (860, 610)
], dtype=np.int32)

# 4) SB ramp
SB_RAMP_POLYGON = np.array([
    (1488, 205),
    (1655, 205),
    (1682, 205),
    (1682, 620),
    (1585, 620),
    (1515, 505),
    (1470, 360),
    (1478, 255)
], dtype=np.int32)

# 5) Merge buffer between SB_MAIN and SB_RAMP
MERGE_BUFFER_POLYGON = np.array([
    (1460, 760),
    (1535, 760),
    (1545, 610),
    (1545, 340),
    (1515, 250),
    (1475, 250),
    (1445, 335),
    (1440, 500)
], dtype=np.int32)

# 6) Neutral overlap / gore area near NB split
NEUTRAL_ZONE_POLYGON = np.array([
    (390, 290),
    (515, 250),
    (590, 210),
    (510, 145),
    (330, 165),
    (180, 250),
    (120, 340),
    (245, 420)
], dtype=np.int32)

ROI_CONFIG = {
    "NB_MAIN": {
        "polygon": NB_MAIN_POLYGON,
        "color": (0, 255, 0),
        "ppm": 8.5,
        "count_speed": True
    },
    "NB_RAMP": {
        "polygon": NB_RAMP_POLYGON,
        "color": (0, 200, 255),
        "ppm": 7.5,
        "count_speed": True
    },
    "SB_MAIN": {
        "polygon": SB_MAIN_POLYGON,
        "color": (0, 0, 255),
        "ppm": 10.0,
        "count_speed": True
    },
    "SB_RAMP": {
        "polygon": SB_RAMP_POLYGON,
        "color": (255, 0, 255),
        "ppm": 9.5,
        "count_speed": True
    },
    "MERGE_BUFFER": {
        "polygon": MERGE_BUFFER_POLYGON,
        "color": (255, 255, 0),
        "ppm": 9.0,
        "count_speed": False
    },
    "NEUTRAL_ZONE": {
        "polygon": NEUTRAL_ZONE_POLYGON,
        "color": (180, 180, 180),
        "ppm": 9.0,
        "count_speed": False
    }
}

track_history = defaultdict(lambda: deque(maxlen=8))
MIN_BOX_HEIGHT_FOR_SPEED_TEXT = 35

minute_rows = defaultdict(lambda: {
    "frames": 0,
    "vehicle_count_sum": 0,
    "car_count_sum": 0,
    "motorcycle_count_sum": 0,
    "bus_count_sum": 0,
    "truck_count_sum": 0,
    "unique_ids": set(),
    "speed_sum": 0.0,
    "speed_count": 0
})

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def get_roi(cx, cy):
    pt = (cx, cy)

    # Priority order matters
    # Buffer is checked before SB_MAIN/SB_RAMP final assignment to suppress edge ambiguity
    for roi_name in ["NB_MAIN", "NB_RAMP", "MERGE_BUFFER", "SB_RAMP", "SB_MAIN", "NEUTRAL_ZONE"]:
        polygon = ROI_CONFIG[roi_name]["polygon"]
        if point_in_polygon(pt, polygon):
            return roi_name, ROI_CONFIG[roi_name]

    return None, None

def draw_polygon_overlay(frame, polygon, color, label):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

    x, y, w, h = cv2.boundingRect(polygon)
    cv2.putText(
        frame,
        label,
        (x + 6, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )

def draw_label(frame, x1, y1, text, box_color=(0, 255, 0), text_color=(0, 0, 0)):
    (text_w, text_h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        1
    )

    text_x = x1
    text_y = max(y1 - 6, text_h + 4)

    cv2.rectangle(
        frame,
        (text_x, text_y - text_h - 4),
        (text_x + text_w + 4, text_y + 2),
        box_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (text_x + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        text_color,
        1,
        cv2.LINE_AA
    )

def estimate_speed_kmh(track_id, cx, cy, frame_idx, fps, pixels_per_meter):
    track_history[track_id].append((cx, cy, frame_idx))

    if len(track_history[track_id]) < 2:
        return None

    x0, y0, f0 = track_history[track_id][0]
    x1, y1, f1 = track_history[track_id][-1]

    dt = (f1 - f0) / fps
    if dt <= 0:
        return None

    dist_px = math.hypot(x1 - x0, y1 - y0)
    speed_px_s = dist_px / dt
    speed_m_s = speed_px_s / pixels_per_meter
    speed_kmh = speed_m_s * 3.6
    return speed_kmh

try:
    frame_file = open(output_frame_csv, mode="w", newline="", encoding="utf-8")
except PermissionError:
    print(f"Access denied: please close {output_frame_csv} in Excel and run again.")
    cap.release()
    exit()

with frame_file as f_frame:
    frame_writer = csv.writer(f_frame)
    frame_writer.writerow([
        "frame",
        "time_sec",
        "minute_bin",
        "vehicle_count",
        "car_count",
        "motorcycle_count",
        "bus_count",
        "truck_count",
        "track_ids",
        "track_speeds_kmh"
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished.")
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=vehicle_classes,
            conf=0.25,
            verbose=False
        )

        for roi_name, cfg in ROI_CONFIG.items():
            draw_polygon_overlay(frame, cfg["polygon"], cfg["color"], roi_name)

        r = results[0]
        boxes = r.boxes

        car_count = 0
        motorcycle_count = 0
        bus_count = 0
        truck_count = 0
        track_ids = []
        speed_pairs = []

        if boxes is not None and boxes.cls is not None and boxes.xyxy is not None:
            xyxy_list = boxes.xyxy.cpu().tolist()
            cls_list = boxes.cls.cpu().tolist()
            id_list = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(cls_list)

            for box, cls_id, track_id in zip(xyxy_list, cls_list, id_list):
                cls_id = int(cls_id)

                x1, y1, x2, y2 = map(int, box)
                box_h = y2 - y1
                cx = int((x1 + x2) / 2)
                cy = int(y2)

                roi_name, roi_cfg = get_roi(cx, cy)
                if roi_name is None or roi_name in ["MERGE_BUFFER", "NEUTRAL_ZONE"]:
                    continue

                if cls_id == 2:
                    car_count += 1
                elif cls_id == 3:
                    motorcycle_count += 1
                elif cls_id == 5:
                    bus_count += 1
                elif cls_id == 7:
                    truck_count += 1

                current_time = frame_idx / fps
                minute_bin = int(current_time // 60)

                if track_id is not None:
                    track_id = int(track_id)
                    track_ids.append(track_id)
                    minute_rows[minute_bin]["unique_ids"].add(track_id)

                    text = f"ID{track_id}"

                    if roi_cfg["count_speed"]:
                        speed_kmh = estimate_speed_kmh(
                            track_id=track_id,
                            cx=cx,
                            cy=cy,
                            frame_idx=frame_idx,
                            fps=fps,
                            pixels_per_meter=roi_cfg["ppm"]
                        )

                        if speed_kmh is not None:
                            speed_kmh = max(0, min(speed_kmh, 180))
                            speed_pairs.append(f"{track_id}:{speed_kmh:.1f}")

                            minute_rows[minute_bin]["speed_sum"] += speed_kmh
                            minute_rows[minute_bin]["speed_count"] += 1

                            if box_h >= MIN_BOX_HEIGHT_FOR_SPEED_TEXT:
                                text = f"ID{track_id} {round(speed_kmh)}"
                else:
                    text = "ID?"

                box_color = roi_cfg["color"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, box_color, -1)
                draw_label(frame, x1, y1, text, box_color=box_color, text_color=(0, 0, 0))

        vehicle_count = car_count + motorcycle_count + bus_count + truck_count
        time_sec = frame_idx / fps
        minute_bin = int(time_sec // 60)

        frame_writer.writerow([
            frame_idx,
            round(time_sec, 3),
            minute_bin,
            vehicle_count,
            car_count,
            motorcycle_count,
            bus_count,
            truck_count,
            ",".join(map(str, sorted(track_ids))),
            ",".join(speed_pairs)
        ])

        minute_rows[minute_bin]["frames"] += 1
        minute_rows[minute_bin]["vehicle_count_sum"] += vehicle_count
        minute_rows[minute_bin]["car_count_sum"] += car_count
        minute_rows[minute_bin]["motorcycle_count_sum"] += motorcycle_count
        minute_rows[minute_bin]["bus_count_sum"] += bus_count
        minute_rows[minute_bin]["truck_count_sum"] += truck_count

        cv2.putText(
            frame,
            f"Frame {frame_idx}  Total ROI Vehicles {vehicle_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Traffic Detection + Multi-ROI Estimated Speed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()

try:
    minute_file = open(output_minute_csv, mode="w", newline="", encoding="utf-8")
except PermissionError:
    print(f"Access denied: please close {output_minute_csv} in Excel and run again.")
    exit()

with minute_file as f_min:
    minute_writer = csv.writer(f_min)
    minute_writer.writerow([
        "minute_bin",
        "start_sec",
        "end_sec",
        "frames_in_minute",
        "avg_vehicle_count_per_frame",
        "avg_car_count_per_frame",
        "avg_motorcycle_count_per_frame",
        "avg_bus_count_per_frame",
        "avg_truck_count_per_frame",
        "unique_track_ids_in_minute",
        "avg_estimated_speed_kmh"
    ])

    for minute_bin in sorted(minute_rows.keys()):
        row = minute_rows[minute_bin]
        frames = row["frames"]
        avg_speed = row["speed_sum"] / row["speed_count"] if row["speed_count"] > 0 else 0

        minute_writer.writerow([
            minute_bin,
            minute_bin * 60,
            minute_bin * 60 + 59.999,
            frames,
            round(row["vehicle_count_sum"] / frames, 3) if frames else 0,
            round(row["car_count_sum"] / frames, 3) if frames else 0,
            round(row["motorcycle_count_sum"] / frames, 3) if frames else 0,
            round(row["bus_count_sum"] / frames, 3) if frames else 0,
            round(row["truck_count_sum"] / frames, 3) if frames else 0,
            len(row["unique_ids"]),
            round(avg_speed, 3)
        ])

print(f"Saved: {output_frame_csv}")
print(f"Saved: {output_minute_csv}")