# --- 基础依赖导入 ---
import numpy as np

np.float = float  # 兼容ByteTrack
import sys, os, cv2, torch, math
from ultralytics import YOLO
from collections import deque, defaultdict
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont

# --- 模块路径设置 ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'Licensee', 'LPRNet', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Licensee', 'MTCNN'))
from LPRNET import LPRNet, CHARS
from STN import STNet
from MTCNN import create_mtcnn_net
from yolox.tracker.byte_tracker import BYTETracker
from Licensee.LPRNet.Evaluation import decode


# ================= 配置优化 =================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VIDEO_PATH = r"D:\studay\CV6\Visual Object Tracking\Traffic\cut.mkv"

    # 模型路径
    YOLO_WEIGHT = "yolov8n.pt"
    LPRNET_WEIGHT = os.path.join("Licensee", "LPRNet", "weights", "Final_LPRNet_model.pth")
    STN_WEIGHT = os.path.join("Licensee", "LPRNet", "weights", "Final_STN_model.pth")
    MTCNN_PNET = os.path.join("Licensee", "MTCNN", "weights", "pnet_Weights")
    MTCNN_ONET = os.path.join("Licensee", "MTCNN", "weights", "onet_Weights")

    # 车牌识别增强参数
    PLATE_MIN_CONF = 0.7  # 车牌检测置信度阈值
    PLATE_RECOG_THRESH = 0.6  # 车牌识别置信度阈值
    PLATE_MIN_AREA = 800  # 最小车牌面积（过滤小目标）
    PLATE_ASPECT_RATIO = (2.5, 5.0)  # 车牌宽高比范围（标准车牌约3.1-4.0）
    PLATE_RECOG_REPEAT = 3  # 重复识别次数（取多数结果）
    PLATE_HISTORY_LEN = 5  # 车牌历史缓存长度（用于投票）

    # 其他参数
    PIXEL_PER_METER = 80
    FRAME_RATE = 30
    SPEED_SMOOTH_WINDOW = 5


# ================= 车牌预处理增强 =================
def preprocess_plate_image(plate_img):
    """
    车牌图像预处理：增强对比度、去噪、字符锐化
    """
    if plate_img.size == 0:
        return None

    # 1. 转换为灰度图
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 2. 自适应阈值二值化（增强字符与背景对比）
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 3. 去除噪声（保留字符细节）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. 边缘增强（突出字符轮廓）
    edges = cv2.Canny(binary, 50, 150)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. 合并增强结果（灰度图+边缘）
    enhanced = cv2.addWeighted(gray, 0.8, edges, 0.2, 0)

    # 6. 转换回BGR格式（保持通道数一致）
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ================= 模型初始化 =================
def init_models(config):
    print("🔧 初始化模型中...")

    # YOLOv8
    yolo = YOLO(config.YOLO_WEIGHT)

    # ByteTrack
    tracker = BYTETracker(SimpleNamespace(
        track_thresh=0.5,
        track_buffer=50,
        match_thresh=0.7,
        min_box_area=100,
        mot20=False
    ))

    # 车牌识别模型
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0.5)
    stn = STNet()
    lprnet.load_state_dict(torch.load(config.LPRNET_WEIGHT, map_location=config.DEVICE, weights_only=True))
    stn.load_state_dict(torch.load(config.STN_WEIGHT, map_location=config.DEVICE, weights_only=True))
    lprnet.to(config.DEVICE).eval()
    stn.to(config.DEVICE).eval()

    print("✅ 模型加载完成！")
    return yolo, tracker, lprnet, stn


# ================= 车牌识别增强 =================
def recognize_plate_enhanced(lprnet, stn, plate_img, device, config):
    """
    增强版车牌识别：多次识别+结果投票+置信度过滤
    """
    if plate_img is None or plate_img.size == 0:
        return "", 0.0

    # 预处理增强
    processed = preprocess_plate_image(plate_img)
    if processed is None:
        return "", 0.0

    # 多次识别取稳定结果
    results = []
    for _ in range(config.PLATE_RECOG_REPEAT):
        # 尺寸标准化
        resized = cv2.resize(processed, (94, 24))
        img_tensor = torch.tensor(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        # 推理
        with torch.no_grad():
            stn_out = stn(img_tensor)
            preds = lprnet(stn_out)

        # 解码并计算置信度
        labels, confs = decode(preds.cpu().detach().numpy(), CHARS)
        if labels and len(labels[0]) >= 7:  # 标准车牌长度校验
            results.append((labels[0], np.mean(confs[0])))

    if not results:
        return "", 0.0

    # 过滤低置信度结果
    filtered = [(l, c) for l, c in results if c >= config.PLATE_RECOG_THRESH]
    if not filtered:
        return "", 0.0

    # 结果投票（取出现次数最多的结果）
    label_counts = defaultdict(float)
    for label, conf in filtered:
        label_counts[label] += conf  # 加权计数（置信度高的权重高）

    best_label = max(label_counts.items(), key=lambda x: x[1])[0]
    best_conf = max(c for l, c in filtered if l == best_label)

    return best_label, best_conf


# ================= 车牌跟踪匹配增强 =================
def match_plate_to_vehicle(plates, tracks, frame, lprnet, stn, device, config):
    """
    增强版车牌-车辆匹配：基于位置约束+历史匹配+IOU加权
    """
    id2plate = {}
    valid_plates = []

    # 1. 过滤无效车牌（面积、宽高比校验）
    for bbox in plates:
        x1, y1, x2, y2, score = bbox[:5]
        if score < config.PLATE_MIN_CONF:
            continue

        # 计算宽高比和面积
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        # 过滤不符合标准的车牌
        if (area < config.PLATE_MIN_AREA or
                not (config.PLATE_ASPECT_RATIO[0] <= aspect_ratio <= config.PLATE_ASPECT_RATIO[1])):
            continue

        valid_plates.append((x1, y1, x2, y2, score))

    # 2. 识别有效车牌并匹配
    for (x1, y1, x2, y2, score) in valid_plates:
        # 截取车牌区域
        plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        plate_text, conf = recognize_plate_enhanced(lprnet, stn, plate_crop, device, config)

        if not plate_text or conf < config.PLATE_RECOG_THRESH:
            continue

        # 3. 智能匹配到车辆跟踪框
        best_match = None
        max_score = 0

        for track in tracks:
            tx1, ty1, tx2, ty2 = track.tlbr
            track_id = track.track_id

            # 计算IOU
            ix1 = max(x1, tx1)
            iy1 = max(y1, ty1)
            ix2 = min(x2, tx2)
            iy2 = min(y2, ty2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
            iou = inter / union if union > 0 else 0

            # 加权评分（IOU + 位置约束：车牌通常在车辆下部）
            position_score = 1.0 if (ty1 <= y1 <= ty2 and y2 <= ty2) else 0.5
            total_score = iou * 0.7 + position_score * 0.3

            if total_score > max_score and total_score > 0.4:
                max_score = total_score
                best_match = track_id

        if best_match is not None:
            id2plate[best_match] = (plate_text, conf)

    return id2plate


# ================= 工具函数 =================
def convert_pixel_speed_to_kmh(pixel_speed, pixel_per_meter, frame_rate):
    meter_per_second = (pixel_speed / pixel_per_meter) * frame_rate
    return max(0, round(meter_per_second * 3.6, 1))


def smooth_speed(speeds, window_size):
    if len(speeds) < window_size:
        window_size = len(speeds)
    return sum(speeds[-window_size:]) / window_size if window_size > 0 else 0


def cv2_add_text(img, text, pos, color=(0, 255, 255), size=12):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, color, font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================= 主函数 =================
def main():
    config = Config()
    yolo, tracker, lprnet, stn = init_models(config)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {config.VIDEO_PATH}")
        return

    # 跟踪状态存储
    track_pts = defaultdict(lambda: deque(maxlen=30))  # 轨迹
    track_speeds = defaultdict(list)  # 速度历史
    track_plates = defaultdict(lambda: deque(maxlen=config.PLATE_HISTORY_LEN))  # 车牌历史
    frame_id = 0

    print("🚀 开始处理 (按ESC退出)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_copy = frame.copy()

        # 1. 车辆检测
        dets = yolo(frame, device=config.DEVICE, conf=0.5, classes=[2, 3, 5, 7], verbose=False)[0]
        vehicle_dets = [[*box[:4], box[4]] for box in dets.boxes.data.cpu().numpy()]

        # 2. 目标跟踪
        tracks = tracker.update(np.array(vehicle_dets, dtype=np.float32),
                                frame.shape[:2], frame.shape[:2])

        # 3. 车牌检测与匹配（每3帧一次，平衡性能）
        if frame_id % 3 == 0:
            plates = create_mtcnn_net(
                frame, mini_lp_size=(50, 15), device=config.DEVICE,
                p_model_path=config.MTCNN_PNET, o_model_path=config.MTCNN_ONET
            )
            plate_matches = match_plate_to_vehicle(plates, tracks, frame, lprnet, stn, config.DEVICE, config)

            # 更新车牌历史（用于稳定结果）
            for track_id, (plate, conf) in plate_matches.items():
                track_plates[track_id].append((plate, conf))

        # 4. 绘制结果
        for track in tracks:
            tx1, ty1, tx2, ty2 = track.tlbr
            track_id = track.track_id
            center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
            track_pts[track_id].append(center)

            # 计算速度
            speed = 0.0
            if len(track_pts[track_id]) >= 2:
                px, py = track_pts[track_id][-2]
                dx, dy = center[0] - px, center[1] - py
                pixel_dist = math.hypot(dx, dy)
                track_speeds[track_id].append(pixel_dist)
                speed = convert_pixel_speed_to_kmh(
                    smooth_speed(track_speeds[track_id], config.SPEED_SMOOTH_WINDOW),
                    config.PIXEL_PER_METER, config.FRAME_RATE
                )

            # 确定最终车牌（从历史中选置信度最高的）
            final_plate = "未知"
            if track_plates[track_id]:
                best_plate = max(track_plates[track_id], key=lambda x: x[1])
                final_plate = best_plate[0]

            # 绘制跟踪框和信息
            cv2.rectangle(frame_copy, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (255, 0, 0), 2)
            info = f"ID:{track_id} 车牌:{final_plate} 速度:{speed}km/h"
            frame_copy = cv2_add_text(frame_copy, info, (int(tx1), max(10, int(ty1) - 30)))

            # 绘制轨迹
            for i in range(1, len(track_pts[track_id])):
                cv2.line(frame_copy,
                         (int(track_pts[track_id][i - 1][0]), int(track_pts[track_id][i - 1][1])),
                         (int(track_pts[track_id][i][0]), int(track_pts[track_id][i][1])),
                         (0, 255, 0), 2)

        cv2.imshow("增强版智能交通监控", frame_copy)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("👋 处理完成")


if __name__ == "__main__":
    main()