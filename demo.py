import numpy as np

np.float = float  # Compatibility with ByteTrack
import cv2
import time
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict, deque
from types import SimpleNamespace


# ================= Configuration Class =================
class Config:
    # GPU Settings
    FORCE_GPU = True
    DEVICE = "cuda:0" if (FORCE_GPU and torch.cuda.is_available()) else "cpu"
    CUDA_FP16 = True  # Enable FP16 inference

    # File Paths
    VIDEO_PATH = "cut.mkv"
    YOLO_WEIGHT = "yolov8n.pt"

    # Detection & Tracking Params
    CONF_THRESH = 0.5
    IOU_THRESH = 0.5
    VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
    TRACK_BUFFER = 15

    # Counting & Speed Params
    COUNT_LINE_Y = 400
    DIRECTION = "both"  # both/up/down
    PIXEL_PER_METER = 60
    SPEED_SMOOTH_WINDOW = 5

    # Display Params
    SHOW_FPS = True
    MAX_DISPLAY_SIZE = (1280, 720)


# ================= GPU Environment Verification =================
def verify_gpu_env():
    if Config.DEVICE.startswith("cuda"):
        print(f"✅ GPU Environment Verified")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   FP16 Inference: {'Enabled' if Config.CUDA_FP16 else 'Disabled'}")
        return True
    else:
        print(f"⚠️ GPU Unavailable, Auto-switching to CPU Inference")
        Config.CUDA_FP16 = False
        return False


# ================= Model Initialization =================
def init_models():
    # Initialize YOLOv8 (FP32 for fusion, then switch to FP16 if enabled)
    yolo = YOLO(Config.YOLO_WEIGHT)
    yolo.to(Config.DEVICE)  # Move model to target device

    # Model fusion (must be done in FP32)
    yolo.fuse()

    # Convert to FP16 if GPU is available
    if Config.CUDA_FP16 and Config.DEVICE.startswith("cuda"):
        yolo.half()
        print(f"🔧 YOLOv8 Converted to FP16 Inference")
    else:
        print(f"🔧 YOLOv8 Running in FP32 Mode")

    # Initialize ByteTrack
    tracker_args = SimpleNamespace(
        track_thresh=Config.CONF_THRESH,
        track_buffer=Config.TRACK_BUFFER,
        match_thresh=0.7,
        min_box_area=100,
        mot20=False
    )
    tracker = BYTETracker(tracker_args)

    print(f"✅ Models Initialized Successfully (Device: {Config.DEVICE})")
    return yolo, tracker


# ================= Core Functions =================
def recognize_plate(frame, bbox):
    """License plate recognition (simulated)"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        if (x2 - x1) < 30 or (y2 - y1) < 10:
            return "Unknown"
        return "Sim-Plate"
    except Exception:
        return "Unknown"


def calculate_speed(position_history, fps, pixel_per_meter, smooth_window):
    """Calculate smoothed vehicle speed (km/h)"""
    if len(position_history) < 2:
        return 0.0

    pos_list = np.array(position_history, dtype=np.float32)
    take_count = min(smooth_window, len(pos_list))
    recent_pos = pos_list[-take_count:]

    diffs = recent_pos[1:] - recent_pos[:-1]
    pixel_dists = np.hypot(diffs[:, 0], diffs[:, 1])
    time_diffs = diffs[:, 2] / fps

    total_dist = np.sum(pixel_dists)
    total_time = np.sum(time_diffs)

    if total_time <= 0:
        return 0.0

    meter_dist = total_dist / pixel_per_meter
    speed = (meter_dist / total_time) * 3.6
    return round(max(0, min(200, speed)), 1)


# ================= Main Function =================
def main():
    verify_gpu_env()
    yolo, tracker = init_models()

    # Open video file
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Failed to Open Video: {Config.VIDEO_PATH}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Adjust count line position
    if Config.COUNT_LINE_Y >= frame_height or Config.COUNT_LINE_Y <= 0:
        Config.COUNT_LINE_Y = frame_height // 2

    # State storage
    pos_history = defaultdict(deque)
    id2plate = {}
    id2speed = {}
    counted_ids = set()
    total_count = 0
    frame_id = 0
    start_time = time.time()

    print(f"🚀 Starting Processing (Resolution: {frame_width}x{frame_height} | FPS: {video_fps:.1f})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        current_time = time.time()
        frame_copy = frame.copy()
        online_targets = []  # Initialize empty to avoid loop errors

        # 1. YOLOv8 Detection
        detections = yolo(
            frame_copy,
            device=Config.DEVICE,
            conf=Config.CONF_THRESH,
            iou=Config.IOU_THRESH,
            classes=Config.VEHICLE_CLASSES,
            verbose=False
        )[0]

        # 2. Process detection results (KEY FIX: Format validation)
        dets = detections.boxes.data.cpu().numpy() if Config.DEVICE.startswith(
            "cuda") else detections.boxes.data.numpy()
        # Convert to [x1, y1, x2, y2, conf] format and filter invalid
        dets = [[*box[:4], box[4]] for box in dets if len(box) >= 5]  # Ensure each box has 5+ elements
        dets_np = np.array(dets, dtype=np.float32)

        # 3. Fix: Skip tracking if no valid detections (avoid empty array error)
        if dets_np.size > 0 and dets_np.shape[1] == 5:
            online_targets = tracker.update(
                dets_np,
                (frame_height, frame_width),
                (frame_height, frame_width)
            )
        else:
            # Update tracker with empty to maintain state (prevent crash)
            online_targets = tracker.update(
                np.empty((0, 5), dtype=np.float32),
                (frame_height, frame_width),
                (frame_height, frame_width)
            )

        # 4. Process tracked targets
        for target in online_targets:
            tx1, ty1, tx2, ty2 = target.tlbr
            track_id = target.track_id
            center_x = (tx1 + tx2) / 2
            center_y = (ty1 + ty2) / 2

            # Update position history
            pos_history[track_id].append((center_x, center_y, frame_id))
            if len(pos_history[track_id]) > Config.SPEED_SMOOTH_WINDOW * 2:
                pos_history[track_id].popleft()

            # License plate recognition
            if frame_id % 5 == 0:
                id2plate[track_id] = recognize_plate(frame_copy, (tx1, ty1, tx2, ty2))

            # Calculate speed
            id2speed[track_id] = calculate_speed(
                pos_history[track_id],
                video_fps,
                Config.PIXEL_PER_METER,
                Config.SPEED_SMOOTH_WINDOW
            )

            # Vehicle counting
            if track_id not in counted_ids and len(pos_history[track_id]) >= 2:
                prev_y = pos_history[track_id][-2][1]
                curr_y = center_y
                count_trigger = False

                if Config.DIRECTION == "both":
                    count_trigger = (prev_y < Config.COUNT_LINE_Y and curr_y >= Config.COUNT_LINE_Y) or \
                                    (prev_y > Config.COUNT_LINE_Y and curr_y <= Config.COUNT_LINE_Y)
                elif Config.DIRECTION == "down":
                    count_trigger = prev_y < Config.COUNT_LINE_Y and curr_y >= Config.COUNT_LINE_Y
                elif Config.DIRECTION == "up":
                    count_trigger = prev_y > Config.COUNT_LINE_Y and curr_y <= Config.COUNT_LINE_Y

                if count_trigger:
                    total_count += 1
                    counted_ids.add(track_id)

        # 5. Draw visualization
        # Draw count line
        cv2.line(frame_copy, (0, Config.COUNT_LINE_Y), (frame_width, Config.COUNT_LINE_Y), (0, 0, 255), 2)

        # Draw count & FPS
        display_texts = [f"Total Vehicles: {total_count}"]
        if Config.SHOW_FPS:
            current_fps = frame_id / (current_time - start_time)
            display_texts.append(f"FPS: {current_fps:.1f} ({Config.DEVICE})")

        for i, text in enumerate(display_texts):
            cv2.putText(frame_copy, text, (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw tracked boxes
        for target in online_targets:
            tx1, ty1, tx2, ty2 = target.tlbr
            track_id = target.track_id
            plate = id2plate.get(track_id, "Unknown")
            speed = id2speed.get(track_id, 0.0)

            cv2.rectangle(frame_copy, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 0, 0), 2)
            info_text = f"ID:{int(track_id)} | {plate} | {speed}km/h"
            text_pos_y = max(10, int(ty1) - 10)
            cv2.putText(frame_copy, info_text, (int(tx1), text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. Resize for display
        scale = min(Config.MAX_DISPLAY_SIZE[0] / frame_width, Config.MAX_DISPLAY_SIZE[1] / frame_height, 1.0)
        if scale < 1.0:
            display_frame = cv2.resize(frame_copy, (int(frame_width * scale), int(frame_height * scale)),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame_copy

        cv2.imshow("GPU-Accelerated Traffic Monitoring", display_frame)
        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    if Config.DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

    # Final stats
    total_time = time.time() - start_time
    print(f"👋 Processing Completed")
    print(f"   Total Vehicles Detected: {total_count}")
    print(f"   Total Frames Processed: {frame_id}")
    print(f"   Average FPS: {frame_id / total_time:.1f}")


if __name__ == "__main__":
    main()