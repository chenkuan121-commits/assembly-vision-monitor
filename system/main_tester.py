import sys
import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import json
import cv2
from workflow_monitor import WorkflowMonitor  # 🌟 新增导入
from ng_tracker import NGProductTracker   # 🌟 NG 产品追踪
from derived_obb_relation import (
    ObbInternalOrderTracker,
    derived_obb_virtual_mapping,
    normalize_derived_obb_relation,
)
from state_monitor import ToggleStateMonitorEngine
from slot_monitor import (
    AnchorSlotMonitorEngine,
    ResultSequenceMonitorEngine,
    normalize_step_wiring_check,
)
import shutil
from intent_engine import IntentEngine
import time
import math
import re
import html
from collections import deque

def _init_runtime_base_path():
    if not getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(__file__))

    # Folder-mode EXE: keep user-created files beside VisionCodex.exe so they are easy to find.
    # PyInstaller dependencies and bundled defaults stay in _internal / _MEIPASS.
    app_data = os.path.dirname(sys.executable)
    bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    os.makedirs(app_data, exist_ok=True)

    def copy_missing_tree(src, dst):
        if not os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            return
        for root, _, files in os.walk(src):
            rel = os.path.relpath(root, src)
            target_root = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(target_root, exist_ok=True)
            for filename in files:
                source_file = os.path.join(root, filename)
                target_file = os.path.join(target_root, filename)
                if not os.path.exists(target_file):
                    shutil.copy2(source_file, target_file)

    for name in ("models", "configs", "aoi_captures", "projects", "logs", "video-photo"):
        src = os.path.join(bundle_dir, name)
        dst = os.path.join(app_data, name)
        copy_missing_tree(src, dst)
    for name in ("simhei.ttf", "mobile_sam.pt", "yolo11n.pt", "yolo26n.pt", "yolov8n.pt", "yolov8n-obb.pt"):
        src = os.path.join(bundle_dir, name)
        dst = os.path.join(app_data, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    os.chdir(app_data)
    return app_data




base_path = _init_runtime_base_path()
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from datetime import datetime  # 🌟 新增：用于记录关闭时的时间
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
# 如果没有安装 pyrealsense2，记得 pip install pyrealsense2
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,QSlider,
                               QHBoxLayout, QWidget, QFileDialog, QLabel, QCheckBox,
                               QGroupBox, QMessageBox, QInputDialog, QComboBox, QGridLayout, QTextBrowser, QDialog,
                               QScrollArea)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import model_manager
import process_editor
import fast_trainer
from logic_engine import ProcessLogicEngine
from alarm_light import AlarmLightController, list_serial_port_options


DETECTION_CLASS_COLORS_RGB = (
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (61, 219, 134), (26, 147, 52),
    (0, 212, 187), (44, 153, 168), (0, 194, 255), (52, 69, 147),
    (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133),
    (203, 56, 255), (255, 149, 200), (255, 55, 199),
)


def detection_class_color_rgb(class_id, class_name=""):
    """Return a stable, high-contrast RGB color for one model class."""
    try:
        color_index = int(class_id)
    except (TypeError, ValueError):
        encoded = str(class_name or "").encode("utf-8")
        color_index = sum((index + 1) * value for index, value in enumerate(encoded))
    return DETECTION_CLASS_COLORS_RGB[color_index % len(DETECTION_CLASS_COLORS_RGB)]


def apply_frame_transform(frame, transform_mode):
    if frame is None or transform_mode in (None, "", "none"):
        return frame
    if transform_mode == "flip_v":
        return cv2.flip(frame, 0)
    if transform_mode == "flip_h":
        return cv2.flip(frame, 1)
    if transform_mode == "rotate_180":
        return cv2.flip(frame, -1)
    return frame


def transform_bbox_to_display(bbox, transform_mode, frame_width, frame_height):
    """Map an AABB detected on the native camera frame onto the transformed display frame."""
    x1, y1, x2, y2 = bbox
    if transform_mode == "flip_v":
        return [x1, frame_height - y2, x2, frame_height - y1]
    if transform_mode == "flip_h":
        return [frame_width - x2, y1, frame_width - x1, y2]
    if transform_mode == "rotate_180":
        return [frame_width - x2, frame_height - y2, frame_width - x1, frame_height - y1]
    return [x1, y1, x2, y2]


def transform_points_to_display(points, transform_mode, frame_width, frame_height):
    """Map OBB polygon points from native camera coordinates onto the transformed display frame."""
    pts = np.asarray(points, dtype=np.float32).copy()
    if transform_mode == "flip_v":
        pts[:, 1] = frame_height - 1 - pts[:, 1]
    elif transform_mode == "flip_h":
        pts[:, 0] = frame_width - 1 - pts[:, 0]
    elif transform_mode == "rotate_180":
        pts[:, 0] = frame_width - 1 - pts[:, 0]
        pts[:, 1] = frame_height - 1 - pts[:, 1]
    return pts


def _detection_polygon(detection):
    points = detection.get("points")
    if points is not None:
        polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(polygon) >= 3:
            return cv2.convexHull(polygon).reshape(-1, 2)
    x1, y1, x2, y2 = map(float, detection.get("bbox", (0, 0, 0, 0)))
    return np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)


def detection_overlap_metrics(first, second):
    """Return polygon IoU and intersection/min-area containment for two detections."""
    try:
        first_polygon = _detection_polygon(first)
        second_polygon = _detection_polygon(second)
        first_area = abs(float(cv2.contourArea(first_polygon)))
        second_area = abs(float(cv2.contourArea(second_polygon)))
        if first_area <= 1e-6 or second_area <= 1e-6:
            return 0.0, 0.0
        intersection, _ = cv2.intersectConvexConvex(first_polygon, second_polygon)
        intersection = max(0.0, float(intersection))
        union = first_area + second_area - intersection
        iou = intersection / union if union > 1e-6 else 0.0
        containment = intersection / min(first_area, second_area)
        return iou, containment
    except (TypeError, ValueError, cv2.error):
        return 0.0, 0.0


def filter_overlapping_detections(detections, same_class=False, cross_class=False,
                                  iou_threshold=0.45, containment_threshold=0.8):
    """Suppress lower-confidence overlapping boxes only for explicitly enabled modes."""
    if not same_class and not cross_class:
        return detections

    ranked = sorted(
        enumerate(detections),
        key=lambda item: (-float(item[1].get("confidence", 0.0)), item[0]),
    )
    kept = []
    for original_index, candidate in ranked:
        candidate_class = candidate.get("class")
        should_suppress = False
        for _, existing in kept:
            classes_match = candidate_class == existing.get("class")
            mode_enabled = same_class if classes_match else cross_class
            if not mode_enabled:
                continue
            iou, containment = detection_overlap_metrics(candidate, existing)
            if iou >= float(iou_threshold) or containment >= float(containment_threshold):
                should_suppress = True
                break
        if not should_suppress:
            kept.append((original_index, candidate))
    return [detection for _, detection in sorted(kept, key=lambda item: item[0])]


def _aligned_polygon_points(current_points, previous_points):
    """Align cyclic/reversed OBB vertex order before temporal averaging."""
    current = np.asarray(current_points, dtype=np.float32).reshape(-1, 2)
    previous = np.asarray(previous_points, dtype=np.float32).reshape(-1, 2)
    if current.shape != previous.shape or len(current) < 3:
        return current
    candidates = []
    for source in (current, current[::-1]):
        candidates.extend(np.roll(source, offset, axis=0) for offset in range(len(source)))
    return min(candidates, key=lambda candidate: float(np.mean(np.linalg.norm(candidate - previous, axis=1))))


def get_safe_torch_device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.get_device_properties(0)
            print(
                f"[VisionCodex][Device] CUDA available: {torch.cuda.get_device_name(0)} "
                f"(torch CUDA {torch.version.cuda})",
                flush=True,
            )
            return "cuda:0"
    except Exception as e:
        print(f"[Device] CUDA 检测异常，已回退 CPU: {e}")
    print("[VisionCodex][Device] CUDA unavailable, using CPU", flush=True)
    return "cpu"
def get_ultralytics_device_arg(torch_device):
    return "0" if str(torch_device).startswith("cuda") else "cpu"
def get_onnxruntime_device_arg(torch_device):
    if not str(torch_device).startswith("cuda"):
        return "cpu"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        log_runtime_device(f"ONNXRuntime providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            return "0"
    except Exception as e:
        print(f"[ONNXRuntime] CUDA provider 检测失败，回退 CPU: {e}")
    return "cpu"


def get_torch_device_name(torch_device):
    if not str(torch_device).startswith("cuda"):
        return "CPU"
    try:
        import torch
        index = int(str(torch_device).split(":")[1]) if ":" in str(torch_device) else 0
        return torch.cuda.get_device_name(index)
    except Exception as e:
        return f"CUDA ({e})"


def describe_infer_device(engine_name, infer_device, torch_device=None):
    if str(infer_device).lower() == "cpu":
        return "CPU"
    if torch_device is None:
        torch_device = "cuda:0"
    device_name = get_torch_device_name(torch_device)
    if "ONNX" in str(engine_name):
        return f"{device_name} / CUDAExecutionProvider"
    return device_name


def log_runtime_device(message):
    print(f"[VisionCodex][Device] {message}", flush=True)


class VisionThread(QThread):
    # 信号传递参数：画面, 步骤列表, 当前索引, 暂停标识, 进度, 警报文字, 完成的轮数, 当前子次数
    update_ui_signal = Signal(np.ndarray, list, int, bool, int, str, int, int)
    recording_time_signal = Signal(str)
    alarm_clip_saved_signal = Signal(str)
    aoi_update_signal = Signal(float, str, bool)  # similarity, state, is_blocked
    aoi_capture_done_signal = Signal(np.ndarray, np.ndarray, tuple)  # feature_vector, crop_image, (frame_w, frame_h, crop_w, crop_h)
    aoi_capture_failed_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.model = None
        self.engine = ProcessLogicEngine()
        # 👇 新增：专门用来死死盯着最后一步的专属引擎
        self.final_step_engine = ProcessLogicEngine()
        self.logic_engine = ProcessLogicEngine()
        self.intent_engine = IntentEngine()
        self.derived_obb_tracker = ObbInternalOrderTracker()
        self.toggle_state_monitor = ToggleStateMonitorEngine()
        self.slot_monitor = AnchorSlotMonitorEngine()
        self.result_sequence_monitor = ResultSequenceMonitorEngine()
        self.step_wiring_monitor = AnchorSlotMonitorEngine()
        self.step_wiring_configs = {}
        self.active_step_wiring_idx = None
        self.step_wiring_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.toggle_state_statuses = []
        self.slot_monitor_statuses = []
        self.result_sequence_statuses = []
        self.slot_expectation_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.monitor_event_log = []
        self.workflow_monitor = WorkflowMonitor(self.engine)  # 🌟 挂载全局监控器
        self.show_jump_progress = process_editor.DEFAULT_JUMP_PROGRESS_VISIBLE
        self.ng_tracker = NGProductTracker()  # 🌟 NG 产品追踪器
        self.remediation_engines = {}  # 跳步补救引擎: step_idx -> ProcessLogicEngine
        self.remediation_mode = False
        self.remediation_step_idx = None
        self.remediation_resume_idx = 0
        self.remediation_request_idx = None
        self.remediation_cancel_signal = False
        self.remediation_status_msg = ""
        self.aoi_context = "normal"
        self.unordered_step_engines = {}
        self.step_progress_by_idx = {}
        self.step_detach_status_by_idx = {}
        self.step_action_status_by_idx = {}
        self.step_release_status_by_idx = {}
        self.step_cooldown_until_by_idx = {}
        self.step_cooldown_status_by_idx = {}
        self.step_prereq_status_by_idx = {}
        self.step_result_status_by_idx = {}
        self.step_result_hit_counts = {}
        self.step_post_action_latched = set()
        self.post_completion_check = None
        self.unordered_completion_order = {}
        self.unordered_completion_seq = 0
        self.unordered_active_idx = None
        self.wrong_pair_counters = {}
        self.pending_group_final_checks = []
        self.group_final_alert_message = ""
        self.group_final_alert_frames = 0
        self.group_final_alert_key = ""

        # AOI 特征比对状态机
        self.aoi_extractor = None
        self.aoi_state = None  # None | 'finding_anchor' | 'checking' | 'blocked'
        self.aoi_step_idx = None
        self.aoi_similarity = 0.0
        self.aoi_stable_count = 0
        self.aoi_check_start_time = 0.0
        self.aoi_standard_vector = None
        self.aoi_threshold = 0.85
        self.aoi_timeout = 5.0
        self.aoi_anchor_class = None
        self.aoi_force_signal = False
        self.aoi_pass_flash = 0      # AOI 通过后绿色边框闪烁帧数 (1s=30帧)
        self.aoi_pass_msg_frames = 0  # AOI 通过后提示消息保持帧数 (3s=90帧)
        self.aoi_pending_restart = False  # AOI 通过后延迟重启循环
        self.req_aoi_capture = False
        self._aoi_capture_anchor = None
        self._aoi_capture_ttl = 0  # 抓拍超时倒计时（帧数），0=未激活
        self.source = None
        self.source_type = "video"
        self.speed_multiplier = 1
        self.infer_device = "cpu"
        self.selected_class_ids = []
        self.model_names = {}       # 类名缓存：{id: eng_name}，避免 ONNX 触发 CUDA 崩溃
        self.name_to_id_cache = {}  # 反向映射：{eng_name: id}
        self.process_steps = []
        self.forbidden_targets = []
        self.state_alarm_rules = []
        self.state_alarm_confirm_frames = process_editor.DEFAULT_STATE_ALARM_CONFIRM_FRAMES
        self.state_alarm_release_frames = process_editor.DEFAULT_STATE_ALARM_RELEASE_FRAMES
        self.state_alarm_padding_ratio = process_editor.DEFAULT_STATE_ALARM_PADDING_RATIO
        self.state_alarm_hit_counts = {}
        self.state_alarm_active_idx = None
        self.state_alarm_clear_frames = 0
        self.state_alarm_message = ""
        self.current_step_idx = 0
        self.current_sub_count = 0
        self.completed_cycles = 0
        self.just_restarted_cycle = False
        self.step_start_time = 0
        self.final_sub_count = 0  # 最后一步累计完成了几次
        self.final_is_pausing = False  # 最后一步单次完成后的冷却判定状态
        self.final_pause_start_time = 0  # 冷却计时器
        self.final_last_action_time = 0  # 上一次拧螺丝的时间（用于防误触清零）
        self.step_timeout = process_editor.DEFAULT_STEP_TIMEOUT
        self.current_conf = 0.25  # YOLO 置信度阈值 (动态可调)
        # 检测框后处理默认全部关闭；关闭时不添加推理参数，也不修改模型原始结果。
        self.same_class_box_filter_enabled = False
        self.cross_class_box_filter_enabled = False
        self.detection_filter_iou = 0.45
        self.detection_filter_containment = 0.8
        self.detection_smoothing_current_weight = 0.72
        self._previous_stable_detections = []
        self.is_pausing = False
        self.pause_start_time = 0
        self.cycle_pausing = False
        self.use_chinese_labels = True  # 🌟 默认启用中文渲染，避免 OpenCV 显示乱码
        self.MAX_LOST_FRAMES = 3000
        # 🌟 性能优化：PIL 字体只加载一次，不每帧重复 IO
        self.font_label = None
        self.font_hand = None
        self._text_patch_cache = {}
        self._text_patch_cache_limit = 512
        self._init_fonts()
        # UI 通信的控制开关
        self.force_skip_signal = False
        self.reset_signal = False
        self.req_take_photo = False
        self.req_record_action = None  # 'start', 'pause', 'resume', 'stop'
        self.is_recording = False
        self.is_record_paused = False
        self.video_writer = None
        self.record_start_time = 0
        self.total_paused_time = 0
        self.pause_start_tick = 0
        # 🌟 分辨率设置（默认值）
        self.capture_width = 1920
        self.capture_height = 1080
        self.yolo_imgsz = 1280
        self.yolo_input_size = None  # 实际送入模型的 (width, height)
        self._yolo_input_size_key = None
        self.record_width = 1920
        self.record_height = 1080
        # 报警追溯录像：常驻保存最近 5 秒的低内存 JPEG 环形缓冲区，
        # 报警触发后再继续写 5 秒，并把当时工序上下文写进画面和 JSON。
        self.alarm_clip_enabled = True
        self.alarm_clip_pre_seconds = 5.0
        self.alarm_clip_post_seconds = 5.0
        self.alarm_clip_target_fps = 15.0
        self.alarm_clip_width = 1280
        self.alarm_clip_height = 720
        self.alarm_clip_root = os.path.join(base_path, "alarm-recordings")
        self._alarm_frame_buffer = deque()
        self._alarm_clip_writer = None
        self._alarm_clip_event = None
        self._alarm_clip_end_time = 0.0
        self._alarm_clip_last_written_ts = 0.0
        self._alarm_last_present_key = ""
        self._alarm_key_last_triggered = {}
        self._alarm_context_signature = None
        self._alarm_clip_product_ref = None
        self._alarm_last_buffered_ts = 0.0
        self.frame_transform = "none"
        self.alarm_light = AlarmLightController()
        self._alarm_forbidden_active = False
        self._alarm_forbidden_buzzer = True
        self._alarm_aoi_blocked_active = False

    def _safe_stop_pipeline(self, pipeline):
        if pipeline:
            try:
                pipeline.stop()
            except Exception as e:
                print(f"[RealSense] stop ignored: {e}")

    def _safe_release_capture(self, cap):
        if cap:
            try:
                cap.release()
            except Exception as e:
                print(f"[Camera] release ignored: {e}")

    def _update_yolo_input_size(self, native_frame):
        """Cache the actual preprocessed tensor size used by Ultralytics."""
        if native_frame is None or self.model is None:
            self.yolo_input_size = None
            self._yolo_input_size_key = None
            return None
        frame_h, frame_w = native_frame.shape[:2]
        predictor = getattr(self.model, "predictor", None)
        key = (
            int(frame_w),
            int(frame_h),
            int(self.yolo_imgsz) if self.yolo_imgsz is not None else None,
            id(predictor),
        )
        if self._yolo_input_size_key == key and self.yolo_input_size:
            return self.yolo_input_size
        try:
            transformed = predictor.pre_transform([native_frame])[0]
            input_h, input_w = transformed.shape[:2]
            self.yolo_input_size = (int(input_w), int(input_h))
        except Exception:
            fallback = int(self.yolo_imgsz or 640)
            self.yolo_input_size = (fallback, fallback)
        self._yolo_input_size_key = key
        return self.yolo_input_size

    def _build_infer_kwargs(self, active_class_ids):
        kwargs = {
            "classes": active_class_ids,
            "verbose": False,
            "conf": self.current_conf,
            "device": self.infer_device,
        }
        if self.yolo_imgsz is not None:
            kwargs["imgsz"] = self.yolo_imgsz
        if self.same_class_box_filter_enabled or self.cross_class_box_filter_enabled:
            kwargs["iou"] = self.detection_filter_iou
        if self.cross_class_box_filter_enabled:
            kwargs["agnostic_nms"] = True
        return kwargs

    def reset_detection_postprocessing(self):
        self._previous_stable_detections = []

    def _stabilize_detection_positions(self, detections):
        if not self.same_class_box_filter_enabled:
            self.reset_detection_postprocessing()
            return detections

        used_previous = set()
        stabilized = []
        current_weight = float(self.detection_smoothing_current_weight)
        previous_weight = 1.0 - current_weight
        for detection in detections:
            best_index = None
            best_score = 0.0
            for index, previous in enumerate(self._previous_stable_detections):
                if index in used_previous or previous.get("class") != detection.get("class"):
                    continue
                iou, containment = detection_overlap_metrics(detection, previous)
                score = max(iou, containment * 0.6)
                if score > best_score:
                    best_index = index
                    best_score = score

            current = dict(detection)
            if best_index is not None and best_score >= 0.2:
                previous = self._previous_stable_detections[best_index]
                used_previous.add(best_index)
                if detection.get("points") is not None and previous.get("points") is not None:
                    previous_points = np.asarray(previous["points"], dtype=np.float32).reshape(-1, 2)
                    current_points = _aligned_polygon_points(detection["points"], previous_points)
                    if current_points.shape == previous_points.shape:
                        smoothed_points = current_weight * current_points + previous_weight * previous_points
                        current["points"] = smoothed_points.tolist()
                        current["bbox"] = [
                            float(np.min(smoothed_points[:, 0])),
                            float(np.min(smoothed_points[:, 1])),
                            float(np.max(smoothed_points[:, 0])),
                            float(np.max(smoothed_points[:, 1])),
                        ]
                else:
                    current_bbox = np.asarray(detection.get("bbox", ()), dtype=np.float32)
                    previous_bbox = np.asarray(previous.get("bbox", ()), dtype=np.float32)
                    if current_bbox.shape == (4,) and previous_bbox.shape == (4,):
                        current["bbox"] = (
                            current_weight * current_bbox + previous_weight * previous_bbox
                        ).tolist()
            stabilized.append(current)

        self._previous_stable_detections = [dict(item) for item in stabilized]
        return stabilized

    def _postprocess_model_detections(self, detections):
        filtered = filter_overlapping_detections(
            detections,
            same_class=self.same_class_box_filter_enabled,
            cross_class=self.cross_class_box_filter_enabled,
            iou_threshold=self.detection_filter_iou,
            containment_threshold=self.detection_filter_containment,
        )
        return self._stabilize_detection_positions(filtered)

    def _init_fonts(self):
        """预加载字体，避免每帧重复 IO。优先用打包/同目录字体，再回退系统字体"""
        # 优先同目录/打包路径
        font_label_paths = [
            os.path.join(base_path, "simhei.ttf"),
            "simhei.ttf",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        self.font_label = ImageFont.load_default()
        for fp in font_label_paths:
            if os.path.exists(fp):
                try:
                    self.font_label = ImageFont.truetype(fp, 40)
                    break
                except IOError:
                    continue
        # 手部渲染字体
        font_hand_paths = [
            os.path.join(base_path, "msyh.ttc"),
            "C:/Windows/Fonts/msyh.ttc",
        ]
        self.font_hand = self.font_label
        for fp in font_hand_paths:
            if os.path.exists(fp):
                try:
                    self.font_hand = ImageFont.truetype(fp, 35)
                    break
                except IOError:
                    continue

    def _get_text_patch(self, text, rgb_color, font=None, background=None):
        """Render and cache a small Unicode text patch instead of converting a full frame to PIL."""
        font = font or self.font_label
        rgb_color = tuple(int(channel) for channel in rgb_color)
        background_key = None if background is None else tuple(int(channel) for channel in background)
        cache_key = (str(text), rgb_color, id(font), background_key)
        cached = self._text_patch_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            left, top, right, bottom = font.getbbox(str(text))
        except AttributeError:
            width, height = font.getsize(str(text))
            left, top, right, bottom = 0, 0, width, height
        stroke_width = 1 if background is None else 0
        padding = max(2, stroke_width + 1)
        width = max(1, int(right - left + padding * 2))
        height = max(1, int(bottom - top + padding * 2))
        if background is None:
            pil_patch = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        else:
            pil_patch = Image.new("RGBA", (width, height), (*background_key, 255))
        draw = ImageDraw.Draw(pil_patch)
        draw.text(
            (padding - left, padding - top), str(text), font=font,
            fill=(*rgb_color, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 255),
        )
        rgba = np.asarray(pil_patch)
        bgr = np.ascontiguousarray(rgba[:, :, :3][:, :, ::-1])
        alpha = None if background is not None else np.ascontiguousarray(rgba[:, :, 3])
        patch = (bgr, alpha)
        if len(self._text_patch_cache) >= self._text_patch_cache_limit:
            self._text_patch_cache.clear()
        self._text_patch_cache[cache_key] = patch
        return patch

    @staticmethod
    def _paste_text_patch(frame, patch, x, y):
        """Paste a cached BGR/alpha text patch into a frame with boundary clipping."""
        patch_bgr, alpha = patch
        frame_h, frame_w = frame.shape[:2]
        patch_h, patch_w = patch_bgr.shape[:2]
        x, y = int(x), int(y)
        dst_x1, dst_y1 = max(0, x), max(0, y)
        dst_x2, dst_y2 = min(frame_w, x + patch_w), min(frame_h, y + patch_h)
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return
        src_x1, src_y1 = dst_x1 - x, dst_y1 - y
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        source = patch_bgr[src_y1:src_y2, src_x1:src_x2]
        target = frame[dst_y1:dst_y2, dst_x1:dst_x2]
        if alpha is None:
            target[:] = source
            return
        mask = alpha[src_y1:src_y2, src_x1:src_x2, None].astype(np.uint16)
        target[:] = (
            (source.astype(np.uint16) * mask
             + target.astype(np.uint16) * (255 - mask) + 127) // 255
        ).astype(np.uint8)

    def _reset_aoi_runtime(self):
        self.aoi_state = None
        self.aoi_step_idx = None
        self.aoi_context = "normal"
        self.aoi_similarity = 0.0
        self.aoi_stable_count = 0
        self.aoi_check_start_time = 0.0
        self.aoi_standard_vector = None
        self.aoi_threshold = 0.85
        self.aoi_timeout = 5.0
        self.aoi_anchor_class = None
        self.aoi_pass_flash = 0
        self.aoi_pass_msg_frames = 0
        self.aoi_pending_restart = False

    def _reset_transient_requests(self):
        self.force_skip_signal = False
        self.reset_signal = False
        self.req_take_photo = False
        self.req_record_action = None
        self.aoi_force_signal = False
        self.req_aoi_capture = False
        self._aoi_capture_anchor = None
        self._aoi_capture_ttl = 0
        self.remediation_request_idx = None
        self.remediation_cancel_signal = False
        self._set_forbidden_alarm(False)
        self._alarm_aoi_blocked_active = False

    def _reset_workflow_runtime(
        self,
        clear_group_final_checks=True,
        preserve_jump_alarm=False,
    ):
        self.current_step_idx = 0
        self.current_sub_count = 0
        self.final_sub_count = 0
        self.final_is_pausing = False
        self.final_pause_start_time = 0
        self.final_last_action_time = 0
        self.is_pausing = False
        self.pause_start_time = 0
        self.cycle_pausing = False
        self.engine.reset()
        self.final_step_engine.reset()
        self.derived_obb_tracker.reset()
        self.remediation_engines.clear()
        self.state_alarm_hit_counts.clear()
        self.state_alarm_active_idx = None
        self.state_alarm_clear_frames = 0
        self.state_alarm_message = ""
        self.remediation_mode = False
        self.remediation_step_idx = None
        self.remediation_resume_idx = 0
        self.remediation_status_msg = ""
        self.unordered_step_engines.clear()
        self.step_progress_by_idx.clear()
        self.step_detach_status_by_idx.clear()
        self.step_action_status_by_idx.clear()
        self.step_release_status_by_idx.clear()
        self.step_cooldown_until_by_idx.clear()
        self.step_cooldown_status_by_idx.clear()
        self.step_prereq_status_by_idx.clear()
        self.step_result_status_by_idx.clear()
        self.step_result_hit_counts.clear()
        self.step_post_action_latched.clear()
        self.post_completion_check = None
        self.step_wiring_monitor.configure([])
        self.active_step_wiring_idx = None
        self.step_wiring_status = {
            "configured": False, "satisfied": True, "settled": True,
            "mismatches": [], "progress": 100,
        }
        self.unordered_completion_order.clear()
        self.unordered_completion_seq = 0
        self.unordered_active_idx = None
        self.wrong_pair_counters.clear()
        if clear_group_final_checks:
            self.pending_group_final_checks.clear()
            self.group_final_alert_message = ""
            self.group_final_alert_frames = 0
            self.group_final_alert_key = ""
        self._reset_jump_monitors(
            clear_restart_guard=True,
            preserve_alarm=preserve_jump_alarm,
        )
        if hasattr(self.intent_engine, "reset_runtime_state"):
            self.intent_engine.reset_runtime_state()
        self._reset_aoi_runtime()
        self._alarm_aoi_blocked_active = False
        self.step_start_time = time.time()

    def _set_forbidden_alarm(self, active, buzzer=True):
        active = bool(active)
        buzzer = bool(buzzer)
        if (self._alarm_forbidden_active == active
                and (not active or self._alarm_forbidden_buzzer == buzzer)):
            return
        self._alarm_forbidden_active = active
        self._alarm_forbidden_buzzer = buzzer
        self.alarm_light.set_forbidden_alarm(active, buzzer=buzzer)

    def _flash_red_alarm(self, key, buzzer=True):
        self.alarm_light.flash_red(key=key, times=3, interval=0.3, buzzer=buzzer)

    def prepare_for_new_stream(self, interrupted_reason='切换视频源时产品未完成'):
        """开启新相机/视频前清理旧运行态，避免上一轮按钮指令串到下一轮。"""
        self._finish_alarm_clip("stream_changed")
        self._alarm_frame_buffer.clear()
        self._alarm_last_present_key = ""
        self._alarm_last_buffered_ts = 0.0
        if self.ng_tracker.current_product:
            if self.ng_tracker.has_product_activity(min_elapsed_sec=5):
                self.ng_tracker.finalize_as_ng(interrupted_reason)
            else:
                self.ng_tracker.current_product = None
        self._reset_transient_requests()
        self._reset_workflow_runtime()
        self.slot_monitor.reset()
        self.result_sequence_monitor.reset()
        self.slot_monitor_statuses = []
        self.result_sequence_statuses = []
        self.slot_expectation_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.yolo_input_size = None
        self._yolo_input_size_key = None
        self.reset_detection_postprocessing()

    def _finish_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.is_recording or self.is_record_paused:
            self.recording_time_signal.emit("00:00")
        self.is_recording = False
        self.is_record_paused = False
        self.record_start_time = 0
        self.total_paused_time = 0
        self.pause_start_tick = 0

    @staticmethod
    def _safe_record_path_part(value, fallback="未命名"):
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
        text = text.strip(" ._")
        return (text or fallback)[:80]

    @staticmethod
    def _recordable_alarm_key(message):
        """Return an event key only for real alarms, not ordinary guidance text."""
        text = re.sub(r"\s+", " ", str(message or "")).strip()
        if not text or text.startswith(("🔎", "✅", "🧪")):
            return ""
        alarm_terms = (
            "报警", "预警", "违禁", "跳步", "错误", "失败", "异常", "超时",
            "耗时过长", "强制跳过", "作废", "无法", "中断", "禁止", "不通过",
            "阻塞", "强制放行", "状态违规",
            "布局错误", "接线顺序错误", "接线结果错误", "错误装配",
        )
        if text.startswith(("⚠", "❌", "🚫")) or any(term in text for term in alarm_terms):
            # 进度、相似度等会改变的数字不应把同一次报警拆成多个录像事件。
            return re.sub(r"\d+(?:\.\d+)?%?", "#", text)[:180]
        return ""

    def _build_alarm_record_context(self, alert_message, progress):
        cp = self.ng_tracker.current_product or {}
        post_check = dict(self.post_completion_check or {})
        step_idx = int(post_check.get("step_idx", self.current_step_idx) or 0)
        step_text = ""
        if 0 <= step_idx < len(self.process_steps):
            step_text = str(self.process_steps[step_idx].get("text", "") or "")
        completed_steps = [
            i + 1 for i, rec in enumerate(cp.get("step_records", []) or [])
            if rec.get("status") in {"completed", "remedied"}
        ]
        step_statuses = [
            {
                "step": int(rec.get("step", index + 1) or index + 1),
                "text": str(rec.get("text", "") or ""),
                "status": str(rec.get("status", "pending") or "pending"),
                "reason": str(rec.get("reason", "") or ""),
            }
            for index, rec in enumerate(cp.get("step_records", []) or [])
        ]
        jump_alarms = list(cp.get("jump_alarms", []) or [])
        wiring = dict(self.step_wiring_status or {})
        return {
            "captured_at": datetime.now().isoformat(timespec="milliseconds"),
            "profile": self.ng_tracker.active_profile,
            "product_id": cp.get("id"),
            "product_status": cp.get("status", "未跟踪"),
            "step_index": step_idx,
            "step_number": step_idx + 1 if self.process_steps else None,
            "step_text": step_text,
            "step_progress": 100 if post_check else max(0, min(100, int(progress or 0))),
            "sub_count": int(self.current_sub_count or 0),
            "completed_steps": completed_steps,
            "step_statuses": step_statuses,
            "latest_jump_alarm": dict(jump_alarms[-1]) if jump_alarms else {},
            "alert_message": str(alert_message or ""),
            "post_completion_check": {
                "active": bool(post_check),
                "name": post_check.get("name", ""),
                "type": post_check.get("type", ""),
                "progress": int(post_check.get("progress", 0) or 0),
                "message": post_check.get("message", ""),
            },
            "wiring_check": {
                "configured": bool(wiring.get("configured")),
                "satisfied": bool(wiring.get("satisfied")),
                "phase": wiring.get("phase", ""),
                "expected_order": list(wiring.get("expected_order", []) or []),
                "actual_order": list(
                    wiring.get("actual_order", []) or wiring.get("visible_order", []) or []
                ),
                "mismatches": list(wiring.get("mismatches", []) or []),
            },
        }

    def _render_alarm_context_frame(self, frame, context):
        target_w, target_h = self.alarm_clip_width, self.alarm_clip_height
        src_h, src_w = frame.shape[:2]
        scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
        resized = cv2.resize(
            frame, (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
        )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - resized.shape[0]) // 2
        x0 = (target_w - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (target_w, 150), (18, 18, 18), -1)
        canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
        profile = str(context.get("profile") or "未选择方案")
        product_id = context.get("product_id")
        product_text = f"产品#{product_id}" if product_id is not None else "未跟踪产品"
        step_number = context.get("step_number")
        step_label = f"步骤{step_number}" if step_number is not None else "无普通工序"
        step_text = str(context.get("step_text") or "")[:36]
        alert = str(context.get("alert_message") or "无报警（报警前画面）")[:52]
        post = context.get("post_completion_check", {}) or {}
        stage = (
            f"独立验收 {int(post.get('progress', 0) or 0)}%"
            if post.get("active") else f"工序进度 {int(context.get('step_progress', 0) or 0)}%"
        )
        lines = (
            (f"{context.get('captured_at', '')}  |  {profile}  |  {product_text}", (255, 255, 255)),
            (f"{step_label}  {step_text}  |  {stage}  |  产品状态:{context.get('product_status', '')}", (180, 230, 255)),
            (f"事件: {alert}", (255, 210, 80) if not context.get("alert_message") else (255, 110, 110)),
        )
        for line_idx, (text_value, color) in enumerate(lines):
            patch = self._get_text_patch(text_value, color, font=self.font_hand)
            self._paste_text_patch(canvas, patch, 16, 6 + line_idx * 47)
        return canvas

    @staticmethod
    def _alarm_context_key(context):
        post = context.get("post_completion_check", {}) or {}
        wiring = context.get("wiring_check", {}) or {}
        return (
            context.get("product_id"), context.get("product_status"),
            context.get("step_number"), context.get("step_progress"),
            tuple(context.get("completed_steps", []) or []), context.get("alert_message"),
            tuple(
                (item.get("step"), item.get("status"), item.get("reason"))
                for item in (context.get("step_statuses", []) or [])
            ),
            bool(post.get("active")), post.get("progress"), post.get("message"),
            wiring.get("phase"), wiring.get("satisfied"),
            tuple(wiring.get("actual_order", []) or []),
        )

    def _append_alarm_buffer(self, now, frame, context):
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        self._alarm_frame_buffer.append((now, encoded.tobytes(), context))
        cutoff = now - self.alarm_clip_pre_seconds
        while self._alarm_frame_buffer and self._alarm_frame_buffer[0][0] < cutoff:
            self._alarm_frame_buffer.popleft()

    def _append_alarm_context_change(self, timestamp, context):
        if not self._alarm_clip_event:
            return
        signature = self._alarm_context_key(context)
        if signature == self._alarm_context_signature:
            return
        self._alarm_context_signature = signature
        snapshot = dict(context)
        snapshot["offset_seconds"] = round(
            max(0.0, timestamp - self._alarm_clip_event["buffer_started_epoch"]), 3
        )
        self._alarm_clip_event["context_timeline"].append(snapshot)

    def _start_alarm_clip(self, now, alarm_message, alarm_key, current_fps):
        stamp = datetime.fromtimestamp(now)
        profile_part = self._safe_record_path_part(self.ng_tracker.active_profile, "默认方案")
        cp = self.ng_tracker.current_product
        product_id = cp.get("id") if cp else None
        product_part = f"product_{int(product_id):06d}" if product_id is not None else "untracked"
        event_dir = os.path.join(
            self.alarm_clip_root, stamp.strftime("%Y-%m-%d"), profile_part, product_part
        )
        os.makedirs(event_dir, exist_ok=True)
        basename = f"alarm_{stamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        video_path = os.path.abspath(os.path.join(event_dir, basename + ".mp4"))
        metadata_path = os.path.abspath(os.path.join(event_dir, basename + ".json"))

        buffer_span = (
            self._alarm_frame_buffer[-1][0] - self._alarm_frame_buffer[0][0]
            if len(self._alarm_frame_buffer) > 1 else 0.0
        )
        measured_fps = (
            (len(self._alarm_frame_buffer) - 1) / buffer_span if buffer_span > 0.2
            else float(current_fps or 15.0)
        )
        output_fps = max(5.0, min(30.0, measured_fps))
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
            (self.alarm_clip_width, self.alarm_clip_height),
        )
        if not writer.isOpened():
            writer.release()
            print(f"[AlarmRecording] 无法创建报警录像: {video_path}")
            return False

        buffer_start = self._alarm_frame_buffer[0][0] if self._alarm_frame_buffer else now
        self._alarm_clip_writer = writer
        self._alarm_clip_end_time = now + self.alarm_clip_post_seconds
        self._alarm_clip_last_written_ts = 0.0
        self._alarm_context_signature = None
        self._alarm_clip_event = {
            "schema_version": 1,
            "event_type": "alarm_clip",
            "started_at": stamp.isoformat(timespec="milliseconds"),
            "triggered_at": stamp.isoformat(timespec="milliseconds"),
            "ended_at": "",
            "pre_seconds": self.alarm_clip_pre_seconds,
            "post_seconds": self.alarm_clip_post_seconds,
            "fps": round(output_fps, 3),
            "resolution": [self.alarm_clip_width, self.alarm_clip_height],
            "video_path": video_path,
            "metadata_path": metadata_path,
            "buffer_started_epoch": buffer_start,
            "profile": self.ng_tracker.active_profile,
            "product_id": product_id,
            "triggers": [{
                "time": stamp.isoformat(timespec="milliseconds"),
                "message": str(alarm_message or ""),
                "key": alarm_key,
            }],
            "context_timeline": [],
        }
        for frame_ts, encoded, buffered_context in self._alarm_frame_buffer:
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                writer.write(decoded)
                self._alarm_clip_last_written_ts = frame_ts
                self._append_alarm_context_change(frame_ts, buffered_context)

        self._alarm_clip_product_ref = None
        if cp is not None:
            product_entry = {
                "triggered_at": stamp.isoformat(timespec="milliseconds"),
                "alarm_message": str(alarm_message or ""),
                "video_path": video_path,
                "metadata_path": metadata_path,
                "status": "recording",
            }
            cp.setdefault("alarm_clips", []).append(product_entry)
            self._alarm_clip_product_ref = product_entry
        return True

    def _finish_alarm_clip(self, reason="post_window_complete"):
        if self._alarm_clip_writer is not None:
            self._alarm_clip_writer.release()
            self._alarm_clip_writer = None
        event = self._alarm_clip_event
        if not event:
            return
        event["ended_at"] = datetime.now().isoformat(timespec="milliseconds")
        event["finish_reason"] = reason
        event["duration_seconds"] = round(
            max(0.0, self._alarm_clip_last_written_ts - event.get("buffer_started_epoch", 0.0)), 3
        )
        event.pop("buffer_started_epoch", None)
        metadata_path = event["metadata_path"]
        temp_path = metadata_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(event, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, metadata_path)
            if self._alarm_clip_product_ref is not None:
                self._alarm_clip_product_ref.update({
                    "status": "saved",
                    "ended_at": event["ended_at"],
                    "duration_seconds": event["duration_seconds"],
                })
            self.alarm_clip_saved_signal.emit(event["video_path"])
            print(f"[AlarmRecording] 已保存: {event['video_path']}")
        except Exception as exc:
            print(f"[AlarmRecording] 保存元数据失败: {exc}")
        self._alarm_clip_event = None
        self._alarm_clip_end_time = 0.0
        self._alarm_clip_last_written_ts = 0.0
        self._alarm_context_signature = None
        self._alarm_clip_product_ref = None

    def _update_alarm_clip(self, now, annotated_frame, context, alert_message, current_fps):
        if not self.alarm_clip_enabled:
            if self._alarm_clip_event:
                self._finish_alarm_clip("disabled")
            self._alarm_frame_buffer.clear()
            self._alarm_last_present_key = ""
            self._alarm_last_buffered_ts = 0.0
            return

        alarm_key = self._recordable_alarm_key(alert_message)
        new_trigger = bool(alarm_key and alarm_key != self._alarm_last_present_key)
        if alarm_key:
            self._alarm_last_present_key = alarm_key
        else:
            self._alarm_last_present_key = ""

        # 追溯录像固定抽样到约 15 FPS，避免 4K/高帧率相机为了环形缓冲
        # 每帧都做 JPEG 编码，仍能完整覆盖按时间计算的前后 5 秒。
        sample_interval = 1.0 / max(1.0, float(self.alarm_clip_target_fps or 15.0))
        should_capture = bool(
            new_trigger
            or self._alarm_last_buffered_ts <= 0
            or now - self._alarm_last_buffered_ts >= sample_interval * 0.95
        )
        if not should_capture:
            if self._alarm_clip_event and now >= self._alarm_clip_end_time:
                self._finish_alarm_clip()
            return
        record_frame = self._render_alarm_context_frame(annotated_frame, context)
        self._append_alarm_buffer(now, record_frame, context)
        self._alarm_last_buffered_ts = now

        last_trigger = float(self._alarm_key_last_triggered.get(alarm_key, 0.0) or 0.0)
        if new_trigger and now - last_trigger >= 2.0:
            self._alarm_key_last_triggered[alarm_key] = now
            if len(self._alarm_key_last_triggered) > 128:
                oldest = sorted(self._alarm_key_last_triggered, key=self._alarm_key_last_triggered.get)[:32]
                for key in oldest:
                    self._alarm_key_last_triggered.pop(key, None)
            if self._alarm_clip_event:
                self._alarm_clip_event["triggers"].append({
                    "time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                    "message": str(alert_message or ""),
                    "key": alarm_key,
                })
                self._alarm_clip_end_time = max(
                    self._alarm_clip_end_time, now + self.alarm_clip_post_seconds
                )
            else:
                self._start_alarm_clip(now, alert_message, alarm_key, current_fps)

        if self._alarm_clip_writer is not None and now > self._alarm_clip_last_written_ts:
            self._alarm_clip_writer.write(record_frame)
            self._alarm_clip_last_written_ts = now
            self._append_alarm_context_change(now, context)
        if self._alarm_clip_event and now >= self._alarm_clip_end_time:
            self._finish_alarm_clip()

    def load_config(self, config_data):
        mapping = config_data.get("mapping", {})
        relation_config = normalize_derived_obb_relation(
            config_data.get("derived_obb_relation")
        )
        mapping_lookup = {}
        for info in mapping.values():
            eng_name = str(info.get("eng_name", "") or "").strip()
            zh_name = str(info.get("zh_name", "") or "").strip()
            if eng_name:
                mapping_lookup[eng_name.lower()] = eng_name
            if zh_name and eng_name:
                mapping_lookup[zh_name.lower()] = eng_name
        for key in ("anchor_class", "first_class", "second_class"):
            configured_name = relation_config.get(key, "")
            resolved_name = mapping_lookup.get(str(configured_name).lower())
            if resolved_name:
                relation_config[key] = resolved_name
        runtime_mapping = dict(mapping)
        runtime_mapping.update(derived_obb_virtual_mapping(relation_config))

        old_steps = self.process_steps
        self.process_steps = config_data.get("process_steps", [])
        self.step_wiring_configs = {
            index: normalize_step_wiring_check(step.get("step_wiring_check", {}), index)
            for index, step in enumerate(self.process_steps)
            if isinstance(step, dict)
        }
        self.step_timeout = config_data.get("step_timeout", process_editor.DEFAULT_STEP_TIMEOUT)
        self.show_jump_progress = bool(config_data.get(
            "jump_progress_visible", process_editor.DEFAULT_JUMP_PROGRESS_VISIBLE
        ))
        self.workflow_monitor.configure(
            monitor_scope=config_data.get("jump_monitor_scope", process_editor.DEFAULT_JUMP_MONITOR_SCOPE),
            strong_action_enabled=config_data.get(
                "jump_strong_action_enabled", process_editor.DEFAULT_JUMP_STRONG_ACTION_ENABLED
            ),
            strong_action_frames=config_data.get(
                "jump_strong_action_frames", process_editor.DEFAULT_JUMP_STRONG_ACTION_FRAMES
            ),
            ignore_static_intersection=config_data.get(
                "jump_ignore_static_intersection", process_editor.DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
            ),
        )

        # 缓存类名映射，避免每帧访问 self.model.names（ONNX 模型会触发 CUDA 设备检测导致崩溃）
        self.model_names = {}
        for class_id_str, info in mapping.items():
            try:
                class_id = int(class_id_str)
            except (TypeError, ValueError):
                continue
            self.model_names[class_id] = info.get("eng_name", "")
        self.name_to_id_cache = {v: k for k, v in self.model_names.items() if v}
        self.derived_obb_tracker.configure(relation_config)

        self.engine.build_parser(runtime_mapping)
        self.engine.reset()
        self.toggle_state_monitor.configure(
            config_data.get("toggle_state_monitors", ""),
            config_data.get("state_conditional_rules", ""),
        )
        self.slot_monitor.configure(config_data.get("slot_monitors", ""))
        self.result_sequence_monitor.configure(
            config_data.get("result_monitor_stages", [])
        )
        self.step_wiring_monitor.configure([])
        self.active_step_wiring_idx = None
        self.toggle_state_statuses = []
        self.slot_monitor_statuses = []
        self.result_sequence_statuses = []
        self.slot_expectation_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.step_wiring_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.monitor_event_log = []
        # 👇 新增：同步给末步引擎
        self.final_step_engine.build_parser(runtime_mapping)
        self.final_step_engine.reset()
        # 同步所有补救引擎的字典映射（引擎实例按需创建）
        for eng in self.remediation_engines.values():
            eng.build_parser(runtime_mapping)
        self.remediation_engines.clear()
        self.unordered_step_engines.clear()
        self.step_action_status_by_idx.clear()
        self.step_cooldown_until_by_idx.clear()
        self.step_cooldown_status_by_idx.clear()
        self.step_prereq_status_by_idx.clear()
        self.step_result_status_by_idx.clear()
        self.step_result_hit_counts.clear()
        self.step_post_action_latched.clear()
        self.step_wiring_monitor.configure([])
        self.active_step_wiring_idx = None
        self.step_wiring_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.step_result_status_by_idx.clear()
        self.step_result_hit_counts.clear()
        self.step_post_action_latched.clear()

        # AOI 特征提取器懒加载：有任一工序启用 AOI 时才初始化
        has_aoi = any(s.get('aoi_feature_check', {}).get('enabled') for s in self.process_steps)
        if has_aoi and self.aoi_extractor is None:
            try:
                from aoi_extractor import AOIFeatureExtractor
                device = get_safe_torch_device()
                self.aoi_extractor = AOIFeatureExtractor(backbone='resnet18', device=device)
            except Exception as e:
                print(f"[AOI] 特征提取器初始化失败: {e}")
        forbidden_str = config_data.get("forbidden_items", "")
        self.forbidden_targets = self.engine.parse_step_text(forbidden_str)
        self.state_alarm_rules = self._parse_state_alarm_rules(
            config_data.get("state_alarm_rules", "")
        )
        try:
            self.state_alarm_confirm_frames = max(1, int(config_data.get(
                "state_alarm_confirm_frames", process_editor.DEFAULT_STATE_ALARM_CONFIRM_FRAMES
            )))
        except (TypeError, ValueError):
            self.state_alarm_confirm_frames = process_editor.DEFAULT_STATE_ALARM_CONFIRM_FRAMES
        try:
            self.state_alarm_release_frames = max(1, int(config_data.get(
                "state_alarm_release_frames", process_editor.DEFAULT_STATE_ALARM_RELEASE_FRAMES
            )))
        except (TypeError, ValueError):
            self.state_alarm_release_frames = process_editor.DEFAULT_STATE_ALARM_RELEASE_FRAMES
        try:
            self.state_alarm_padding_ratio = max(0.0, float(config_data.get(
                "state_alarm_padding_ratio", process_editor.DEFAULT_STATE_ALARM_PADDING_RATIO
            )))
        except (TypeError, ValueError):
            self.state_alarm_padding_ratio = process_editor.DEFAULT_STATE_ALARM_PADDING_RATIO
        self.state_alarm_hit_counts.clear()
        self.state_alarm_active_idx = None
        self.state_alarm_clear_frames = 0
        self.state_alarm_message = ""
        # 切换配置只清掉进行中的产品；真正开始计件放到视频流启动后处理
        if old_steps != self.process_steps:
            self.ng_tracker.current_product = None
        self._reset_workflow_runtime()

    def _finish_and_restart_cycle(self, preserve_jump_alarm=False):
        """正常完成当前产品并自动开始下一个产品循环"""
        if self.ng_tracker.current_product:
            self.ng_tracker._finalize_product()
        if self.process_steps:
            self.ng_tracker.start_product(self.process_steps)
        self._reset_workflow_runtime(
            clear_group_final_checks=False,
            preserve_jump_alarm=preserve_jump_alarm,
        )
        self.slot_monitor.reset(preserve_calibration=True)
        self.slot_monitor_statuses = []
        self.slot_expectation_status = {
            "configured": False, "satisfied": True, "settled": True, "mismatches": []
        }
        self.just_restarted_cycle = True
        self._activate_restart_guard(preserve_alarm=preserve_jump_alarm)

    def _step_order_group(self, step_dict):
        return str(step_dict.get("order_group", "")).strip()

    def _is_hand_touch_step(self, step_dict):
        return step_dict.get("action_type") == "hand_touch"

    def _is_detach_step(self, step_dict):
        return step_dict.get("action_type") == "detach"

    def _step_action_confirm_frames(self, step_dict):
        value = step_dict.get("action_confirm_frames", ProcessLogicEngine.ACTION_EVIDENCE_FRAMES)
        if value in (None, ""):
            value = ProcessLogicEngine.ACTION_EVIDENCE_FRAMES
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return ProcessLogicEngine.ACTION_EVIDENCE_FRAMES

    def _step_stable_frames(self, step_dict):
        value = step_dict.get("stable_frames")
        if value in (None, "") and self._is_detach_step(step_dict):
            value = step_dict.get("detach_stable_frames", 0)
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _step_padding_ratio(self, step_dict):
        value = step_dict.get("padding_ratio")
        if value in (None, "") and self._is_detach_step(step_dict):
            value = step_dict.get("detach_padding_ratio", -1)
        if value in (None, ""):
            value = -1
        try:
            return float(value)
        except (TypeError, ValueError):
            return -1

    def _detach_missing_enabled(self, step_dict):
        value = step_dict.get("detach_missing_enabled", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _detach_missing_frames(self, step_dict):
        try:
            return max(1, int(step_dict.get("detach_missing_frames", 30)))
        except (TypeError, ValueError):
            return 30

    def _detach_missing_padding(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("detach_missing_padding_ratio", 0.15)))
        except (TypeError, ValueError):
            return 0.15

    def _requires_hand_release(self, step_dict):
        return (
            step_dict.get("action_type", "spatial") in ("spatial", "hand_touch")
            and bool(step_dict.get("require_hand_release", False))
        )

    def _hand_release_padding(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("hand_release_padding", 0.15)))
        except (TypeError, ValueError):
            return 0.15

    def _hand_release_frames(self, step_dict):
        try:
            return max(1, int(step_dict.get("hand_release_frames", 12)))
        except (TypeError, ValueError):
            return 12

    def _targets_for_step(self, step_dict):
        if self._is_detach_step(step_dict):
            configured = []
            for key in ("detach_removed", "detach_base"):
                value = str(step_dict.get(key, "")).strip()
                if value:
                    configured.extend(self.engine.parse_step_text(value))
            if len(configured) >= 2:
                unique = []
                for target in configured:
                    if target not in unique:
                        unique.append(target)
                return unique[:2]
        return self.engine.parse_step_text(step_dict.get("text", ""))

    def _wrong_pair_confirm_frames(self, step_dict):
        try:
            return max(1, int(step_dict.get("wrong_pair_confirm_frames", 3)))
        except (TypeError, ValueError):
            return 3

    def _wrong_pair_padding(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("wrong_pair_padding_ratio", 0.10)))
        except (TypeError, ValueError):
            return 0.10

    def _wrong_pair_targets(self, step_dict):
        """Return configured wrong item targets and their assembly target targets."""
        wrong_text = str(step_dict.get("wrong_pair_item", "") or "").strip()
        if not wrong_text:
            return [], []
        wrong_targets = self.engine.parse_step_text(wrong_text)

        target_text = str(step_dict.get("wrong_pair_target", "") or "").strip()
        if target_text:
            assembly_targets = self.engine.parse_step_text(target_text)
        else:
            correct_targets = self._targets_for_step(step_dict)
            assembly_targets = correct_targets[1:2]
        return wrong_targets, assembly_targets

    def _wrong_pair_candidate_indices(self):
        if not self.process_steps or not (0 <= self.current_step_idx < len(self.process_steps)):
            return []
        group_indices = self._unordered_group_indices(self.current_step_idx)
        if not group_indices:
            return [self.current_step_idx]
        return [idx for idx in group_indices if self._step_record_status(idx) == "pending"]

    def _check_wrong_pair_alarm(self, detections):
        """Detect configured wrong spatial pairings without changing workflow or NG state."""
        candidate_indices = self._wrong_pair_candidate_indices()
        candidate_set = set(candidate_indices)
        for idx in list(self.wrong_pair_counters):
            if idx not in candidate_set:
                self.wrong_pair_counters.pop(idx, None)

        for idx in candidate_indices:
            step_dict = self.process_steps[idx]
            enabled = (
                step_dict.get("action_type", "spatial") == "spatial"
                and bool(step_dict.get("wrong_pair_enabled", False))
            )
            wrong_targets, assembly_targets = self._wrong_pair_targets(step_dict) if enabled else ([], [])
            if not wrong_targets or not assembly_targets:
                self.wrong_pair_counters.pop(idx, None)
                continue

            intersects = self.engine.target_groups_intersect(
                wrong_targets,
                assembly_targets,
                detections,
                padding_ratio=self._wrong_pair_padding(step_dict),
            )
            if not intersects:
                self.wrong_pair_counters.pop(idx, None)
                continue

            required_frames = self._wrong_pair_confirm_frames(step_dict)
            current_frames = min(required_frames, self.wrong_pair_counters.get(idx, 0) + 1)
            self.wrong_pair_counters[idx] = current_frames
            if current_frames >= required_frames:
                wrong_name = "/".join(
                    name for name in (self._target_display_name(target) for target in wrong_targets) if name
                ) or str(step_dict.get("wrong_pair_item", "")).strip()
                target_name = "/".join(
                    name for name in (self._target_display_name(target) for target in assembly_targets) if name
                ) or str(step_dict.get("wrong_pair_target", "")).strip()
                return (
                    True,
                    f"❌ 错误装配：检测到【{wrong_name}】与【{target_name}】发生装配，请立即检查！",
                    idx,
                )
        return False, "", -1

    def _group_final_check_config(self, group_indices):
        for idx in group_indices or []:
            if not (0 <= idx < len(self.process_steps)):
                continue
            config = self.process_steps[idx].get("group_final_check")
            if isinstance(config, dict) and bool(config.get("enabled", False)):
                return config
        return None

    def _target_boxes_for_group_final(self, detections, targets, padding_ratio=0.0):
        boxes = []
        for detection in detections or []:
            bbox = detection.get("bbox")
            class_name = detection.get("class", "")
            if not bbox:
                continue
            if any(self.engine.target_matches(class_name, target) for target in (targets or [])):
                boxes.append(self.engine.expand_bbox(bbox, padding_ratio))
        return boxes

    def _schedule_group_final_check(self, group_indices, detections=None):
        """Start a non-blocking final-state audit after every member of an unordered group finishes."""
        config = self._group_final_check_config(group_indices)
        if not config:
            return False

        anchor_text = str(config.get("anchor", "") or "").strip()
        expected_text = str(config.get("expected", "") or "").strip()
        wrong_text = str(config.get("wrong", "") or "").strip()
        anchor_targets = self.engine.parse_step_text(anchor_text)
        expected_targets = self.engine.parse_step_text(expected_text)
        wrong_targets = self.engine.parse_step_text(wrong_text)
        if not anchor_targets or not expected_targets or not wrong_targets:
            return False

        try:
            confirm_frames = max(1, int(config.get(
                "confirm_frames", process_editor.DEFAULT_GROUP_FINAL_CONFIRM_FRAMES
            )))
        except (TypeError, ValueError):
            confirm_frames = process_editor.DEFAULT_GROUP_FINAL_CONFIRM_FRAMES
        try:
            padding_ratio = max(0.0, float(config.get(
                "padding_ratio", process_editor.DEFAULT_GROUP_FINAL_PADDING_RATIO
            )))
        except (TypeError, ValueError):
            padding_ratio = process_editor.DEFAULT_GROUP_FINAL_PADDING_RATIO
        try:
            window_frames = max(confirm_frames, int(config.get(
                "window_frames", process_editor.DEFAULT_GROUP_FINAL_WINDOW_FRAMES
            )))
        except (TypeError, ValueError):
            window_frames = process_editor.DEFAULT_GROUP_FINAL_WINDOW_FRAMES

        first_idx = min(group_indices)
        group_name = self._step_order_group(self.process_steps[first_idx]) or str(first_idx + 1)
        anchor_boxes = self._target_boxes_for_group_final(
            detections, anchor_targets, padding_ratio
        )
        # A cycle should never keep two audits for the same logical group alive.
        self.pending_group_final_checks = [
            audit for audit in self.pending_group_final_checks
            if audit.get("group_start") != first_idx
        ]
        self.pending_group_final_checks.append({
            "group": group_name,
            "group_start": first_idx,
            "anchor_text": anchor_text,
            "expected_text": expected_text,
            "wrong_text": wrong_text,
            "anchor_targets": anchor_targets,
            "expected_targets": expected_targets,
            "wrong_targets": wrong_targets,
            "anchor_boxes": anchor_boxes,
            "confirm_frames": confirm_frames,
            "window_frames": window_frames,
            "padding_ratio": padding_ratio,
            "age": 0,
            "correct_hits": 0,
            "wrong_hits": 0,
        })
        return True

    def _check_pending_group_final_checks(self, detections):
        """Return (new_alarm, held_message, alarm_key) without changing workflow/NG state."""
        held_message = ""
        held_key = ""
        if self.group_final_alert_frames > 0:
            held_message = self.group_final_alert_message
            held_key = self.group_final_alert_key
            self.group_final_alert_frames -= 1
            if self.group_final_alert_frames <= 0:
                self.group_final_alert_message = ""
                self.group_final_alert_key = ""

        new_alarm = False
        for audit in list(self.pending_group_final_checks):
            audit["age"] += 1
            padding_ratio = audit["padding_ratio"]
            current_anchor_boxes = self._target_boxes_for_group_final(
                detections, audit["anchor_targets"], padding_ratio
            )
            if current_anchor_boxes:
                audit["anchor_boxes"] = current_anchor_boxes
            anchor_boxes = audit.get("anchor_boxes") or []
            if not anchor_boxes:
                audit["correct_hits"] = 0
                audit["wrong_hits"] = 0
            else:
                hand_boxes = [
                    self.engine.expand_bbox(detection["bbox"], 0.02)
                    for detection in (detections or [])
                    if detection.get("bbox")
                    and self.engine.is_hand_class(detection.get("class", ""))
                ]
                hand_in_anchor = any(
                    self.engine.check_intersection(hand_box, anchor_box)
                    for hand_box in hand_boxes
                    for anchor_box in anchor_boxes
                )
                if hand_in_anchor:
                    audit["correct_hits"] = 0
                    audit["wrong_hits"] = 0
                else:
                    expected_boxes = self._target_boxes_for_group_final(
                        detections, audit["expected_targets"], padding_ratio
                    )
                    wrong_boxes = self._target_boxes_for_group_final(
                        detections, audit["wrong_targets"], padding_ratio
                    )
                    expected_at_anchor = any(
                        self.engine.check_intersection(item_box, anchor_box)
                        for item_box in expected_boxes
                        for anchor_box in anchor_boxes
                    )
                    wrong_at_anchor = any(
                        self.engine.check_intersection(item_box, anchor_box)
                        for item_box in wrong_boxes
                        for anchor_box in anchor_boxes
                    )
                    if expected_at_anchor:
                        audit["correct_hits"] += 1
                        audit["wrong_hits"] = 0
                    elif wrong_at_anchor:
                        audit["wrong_hits"] += 1
                        audit["correct_hits"] = 0
                    else:
                        audit["correct_hits"] = 0
                        audit["wrong_hits"] = 0

            if audit["correct_hits"] >= audit["confirm_frames"]:
                self.pending_group_final_checks.remove(audit)
                continue
            if audit["wrong_hits"] >= audit["confirm_frames"]:
                message = (
                    f"❌ 乱序组【{audit['group']}】终态异常："
                    f"【{audit['anchor_text']}】位置应看到【{audit['expected_text']}】，"
                    f"当前却检测到【{audit['wrong_text']}】。已报警，后续工序继续。"
                )
                key = f"group_final_{audit['group_start']}_{int(time.time() * 1000)}"
                self.group_final_alert_message = message
                self.group_final_alert_frames = 90
                self.group_final_alert_key = key
                held_message = message
                held_key = key
                new_alarm = True
                self.pending_group_final_checks.remove(audit)
                break
            if audit["age"] >= audit["window_frames"]:
                self.pending_group_final_checks.remove(audit)

        return new_alarm, held_message, held_key

    def _target_options(self, target):
        return self.engine.target_options(target)

    def _target_display_name(self, target):
        return self.engine.target_display_name(target, self.engine.eng_to_zh)

    def _parse_state_alarm_rules(self, raw_rules):
        """Parse global state alarms: bad item targets intersecting anchor targets."""
        rules = []

        def parse_alarm_mode(value, buzzer=None):
            if buzzer is not None:
                return "red_buzzer" if bool(buzzer) else "red_only"
            return process_editor.normalize_state_alarm_mode(value)

        def add_rule(item_text, target_text, label="", padding_ratio=None,
                     alarm_mode=None, buzzer=None):
            item_targets = self.engine.parse_step_text(str(item_text or ""))
            anchor_targets = self.engine.parse_step_text(str(target_text or ""))
            if not item_targets or not anchor_targets:
                return
            if padding_ratio in (None, ""):
                padding_value = None
            else:
                try:
                    padding_value = max(0.0, float(padding_ratio))
                except (TypeError, ValueError):
                    padding_value = None
            mode = parse_alarm_mode(alarm_mode, buzzer=buzzer) or "red_buzzer"
            rules.append({
                "item_targets": item_targets,
                "anchor_targets": anchor_targets,
                "label": str(label or "").strip(),
                "padding_ratio": padding_value,
                "alarm_mode": mode,
                "buzzer": mode == "red_buzzer",
            })

        if isinstance(raw_rules, list):
            for entry in raw_rules:
                if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                    continue
                add_rule(
                    entry.get("item") or entry.get("bad_item") or entry.get("wrong_item"),
                    entry.get("target") or entry.get("anchor"),
                    entry.get("name") or entry.get("label") or "",
                    entry.get("padding_ratio"),
                    entry.get("alarm_mode") or entry.get("alarm"),
                    entry.get("buzzer") if "buzzer" in entry else None,
                )
            return rules

        for raw_line in str(raw_rules or "").splitlines():
            parsed_line = process_editor.parse_state_alarm_rule_line(raw_line)
            if parsed_line.get("ignored") or parsed_line.get("error"):
                continue
            alarm_mode = parsed_line.get("alarm_mode") or "red_buzzer"
            label = parsed_line.get("label", str(raw_line).strip())
            if parsed_line.get("item_text") is not None:
                add_rule(
                    parsed_line.get("item_text"),
                    parsed_line.get("target_text"),
                    label,
                    alarm_mode=alarm_mode,
                )
                continue

            detected_targets = self.engine.parse_step_text(parsed_line.get("freeform_text", ""))
            if len(detected_targets) >= 2:
                mode = alarm_mode or "red_buzzer"
                rules.append({
                    "item_targets": detected_targets[:1],
                    "anchor_targets": detected_targets[1:2],
                    "label": label,
                    "padding_ratio": None,
                    "alarm_mode": mode,
                    "buzzer": mode == "red_buzzer",
                })

        return rules

    def _state_alarm_uses_buzzer(self, rule_idx):
        if 0 <= int(rule_idx) < len(self.state_alarm_rules or []):
            return bool(self.state_alarm_rules[int(rule_idx)].get("buzzer", True))
        return True

    def _check_state_alarm(self, detections):
        hit_idx = -1
        hit_message = ""
        for idx, rule in enumerate(self.state_alarm_rules or []):
            item_targets = rule.get("item_targets") or []
            anchor_targets = rule.get("anchor_targets") or []
            if not item_targets or not anchor_targets:
                continue
            padding_ratio = rule.get("padding_ratio")
            if padding_ratio is None:
                padding_ratio = self.state_alarm_padding_ratio
            if not self.engine.target_groups_intersect(
                    item_targets,
                    anchor_targets,
                    detections,
                    padding_ratio=padding_ratio):
                continue
            item_name = "/".join(
                name for name in (self._target_display_name(target) for target in item_targets) if name
            ) or "错误物品"
            anchor_name = "/".join(
                name for name in (self._target_display_name(target) for target in anchor_targets) if name
            ) or "参照目标"
            hit_idx = idx
            hit_message = f"持续状态报警：检测到【{item_name}】与【{anchor_name}】处于禁止关系！"
            break

        if hit_idx >= 0:
            self.state_alarm_hit_counts[hit_idx] = min(
                self.state_alarm_confirm_frames,
                self.state_alarm_hit_counts.get(hit_idx, 0) + 1,
            )
            for idx in list(self.state_alarm_hit_counts):
                if idx != hit_idx:
                    self.state_alarm_hit_counts[idx] = 0
            self.state_alarm_clear_frames = 0
            if (self.state_alarm_active_idx == hit_idx
                    or self.state_alarm_hit_counts[hit_idx] >= self.state_alarm_confirm_frames):
                self.state_alarm_active_idx = hit_idx
                self.state_alarm_message = hit_message
                return True, hit_message, hit_idx
            return False, "", -1

        for idx in list(self.state_alarm_hit_counts):
            self.state_alarm_hit_counts[idx] = 0
        if self.state_alarm_active_idx is not None:
            self.state_alarm_clear_frames += 1
            if self.state_alarm_clear_frames < self.state_alarm_release_frames:
                return True, self.state_alarm_message, self.state_alarm_active_idx
            self.state_alarm_active_idx = None
            self.state_alarm_clear_frames = 0
            self.state_alarm_message = ""
        return False, "", -1

    def _target_option_set(self, targets):
        options = set()
        for target in targets or []:
            options.update(self._target_options(target))
        return options

    def _prerequisite_indices(self, step_dict):
        raw = step_dict.get("prerequisite_steps", "")
        if isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            values = re.findall(r"\d+", str(raw or ""))
        indices = []
        for value in values:
            try:
                idx = int(value) - 1
            except (TypeError, ValueError):
                continue
            if idx >= 0 and idx not in indices:
                indices.append(idx)
        return indices

    def _is_assembly_step(self, step_dict, targets=None):
        if step_dict.get("action_type", "spatial") != "spatial":
            return False
        if targets is None:
            targets = self._targets_for_step(step_dict)
        return self.engine.is_assembly_like_step(targets, step_dict.get("text", ""))

    def _completed_step_indices(self):
        cp = self.ng_tracker.current_product
        if not cp:
            return set()
        completed = set()
        for idx, rec in enumerate(cp.get('step_records', [])):
            if rec.get('status') in ('completed', 'remedied'):
                completed.add(idx)
        return completed

    def _explicit_prereq_satisfied(self, step_idx, step_dict):
        prereqs = [idx for idx in self._prerequisite_indices(step_dict) if idx != step_idx]
        if not prereqs:
            self.step_prereq_status_by_idx.pop(step_idx, None)
            return True
        completed = self._completed_step_indices()
        missing = [idx for idx in prereqs if idx not in completed]
        if missing:
            self.step_prereq_status_by_idx[step_idx] = {
                "missing": missing,
                "required": prereqs,
            }
            return False
        self.step_prereq_status_by_idx.pop(step_idx, None)
        return True

    def _missing_prerequisite_indices(self, step_dict):
        prereqs = self._prerequisite_indices(step_dict)
        if not prereqs:
            return []
        completed = self._completed_step_indices()
        return [idx for idx in prereqs if idx not in completed]

    def _prerequisite_mode(self, step_dict):
        mode = str(step_dict.get("prerequisite_mode", "alarm_only") or "alarm_only")
        if mode not in ("block_and_alarm", "alarm_only", "block_only"):
            return "alarm_only"
        return mode

    def _prerequisite_alarm_blocks_advancement(self, step_idx):
        if step_idx < 0 or step_idx >= len(self.process_steps):
            return False
        step_dict = self.process_steps[step_idx]
        return bool(
            self._missing_prerequisite_indices(step_dict)
            and self._prerequisite_mode(step_dict) == "block_and_alarm"
        )

    def _detach_prereq_satisfied(self, step_idx):
        if step_idx < 0 or step_idx >= len(self.process_steps):
            return True
        detach_targets = self._targets_for_step(self.process_steps[step_idx])
        detach_options = self._target_option_set(detach_targets)
        if len(detach_options) < 2:
            return True

        matched_prior_assembly = None
        for prior_idx in range(step_idx - 1, -1, -1):
            prior_step = self.process_steps[prior_idx]
            prior_targets = self._targets_for_step(prior_step)
            if not self._is_assembly_step(prior_step, prior_targets):
                continue
            if detach_options.issubset(self._target_option_set(prior_targets)):
                matched_prior_assembly = prior_idx
                break

        if matched_prior_assembly is None:
            return True
        return matched_prior_assembly in self._completed_step_indices()

    def _unordered_group_indices(self, step_idx):
        if not self.process_steps or step_idx >= len(self.process_steps):
            return []
        group = self._step_order_group(self.process_steps[step_idx])
        if not group:
            return []

        start = step_idx
        while start > 0 and self._step_order_group(self.process_steps[start - 1]) == group:
            start -= 1

        end = step_idx
        while end + 1 < len(self.process_steps) and self._step_order_group(self.process_steps[end + 1]) == group:
            end += 1

        return list(range(start, end + 1))

    def _step_record_status(self, step_idx):
        cp = self.ng_tracker.current_product
        if not cp or step_idx >= len(cp.get('step_records', [])):
            return 'pending'
        return cp['step_records'][step_idx].get('status', 'pending')

    def _step_strategy(self, step_dict):
        return str(step_dict.get("multi_strategy", "lock") or "lock").lower()

    def _is_time_multi_step(self, step_dict):
        return int(step_dict.get("count", 1) or 1) > 1 and "time" in self._step_strategy(step_dict)

    def _cooldown_seconds(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("cooldown", 1.5) or 0.0))
        except (TypeError, ValueError):
            return 1.5

    def _step_completion_hold_seconds(self, step_dict):
        """读取单步的结果确认停留时长。"""
        try:
            return max(
                0.0,
                float(
                    step_dict.get(
                        "completion_hold_seconds",
                        process_editor.DEFAULT_COMPLETION_HOLD_SECONDS,
                    )
                    or 0.0
                ),
            )
        except (AttributeError, TypeError, ValueError):
            return process_editor.DEFAULT_COMPLETION_HOLD_SECONDS

    def _clear_step_cooldown(self, step_idx):
        self.step_cooldown_until_by_idx.pop(step_idx, None)
        self.step_cooldown_status_by_idx.pop(step_idx, None)

    def _clear_step_runtime_flags(self, step_idx):
        self._clear_step_cooldown(step_idx)
        self.step_prereq_status_by_idx.pop(step_idx, None)
        self.step_release_status_by_idx.pop(step_idx, None)
        self.wrong_pair_counters.pop(step_idx, None)
        self.step_result_status_by_idx.pop(step_idx, None)
        self.step_result_hit_counts.pop(step_idx, None)
        self.step_post_action_latched.discard(step_idx)

    def _set_step_cooldown(self, step_idx, step_dict):
        if not self._is_time_multi_step(step_dict):
            self._clear_step_cooldown(step_idx)
            return
        cooldown = self._cooldown_seconds(step_dict)
        until = time.time() + cooldown
        self.step_cooldown_until_by_idx[step_idx] = until
        self.step_cooldown_status_by_idx[step_idx] = {
            "active": True,
            "remaining": cooldown,
            "total": cooldown,
        }

    def _cooldown_remaining(self, step_idx):
        until = self.step_cooldown_until_by_idx.get(step_idx)
        if not until:
            self._clear_step_cooldown(step_idx)
            return 0.0
        remaining = max(0.0, until - time.time())
        if remaining <= 0:
            self._clear_step_cooldown(step_idx)
            return 0.0
        status = self.step_cooldown_status_by_idx.get(step_idx, {})
        total = float(status.get("total", remaining) or remaining)
        self.step_cooldown_status_by_idx[step_idx] = {
            "active": True,
            "remaining": remaining,
            "total": total,
        }
        return remaining

    def _is_step_cooling_down(self, step_idx, step_dict):
        if not self._is_time_multi_step(step_dict):
            self._clear_step_cooldown(step_idx)
            return False
        return self._cooldown_remaining(step_idx) > 0

    @staticmethod
    def _step_result_check(step_dict):
        raw = step_dict.get("result_check", {}) if isinstance(step_dict, dict) else {}
        raw = dict(raw) if isinstance(raw, dict) else {}
        mode = str(raw.get("mode") or "present")
        if mode not in {
            "present", "absent", "targets_intersect", "targets_separate"
        }:
            mode = "present"
        try:
            stable_frames = max(1, min(300, int(raw.get("stable_frames", 5))))
        except (TypeError, ValueError):
            stable_frames = 5
        timing = str(
            raw.get("timing") or process_editor.DEFAULT_RESULT_CHECK_TIMING
        ).strip()
        if timing not in {"before_completion", "after_completion"}:
            timing = process_editor.DEFAULT_RESULT_CHECK_TIMING
        return {
            "enabled": bool(raw.get("enabled", False)),
            "target": str(raw.get("target") or "").strip(),
            "mode": mode,
            "stable_frames": stable_frames,
            "timing": timing,
        }

    def _post_completion_required(self, step_idx, step_dict=None):
        if not (0 <= step_idx < len(self.process_steps)):
            return False
        step_dict = step_dict or self.process_steps[step_idx]
        result_check = self._step_result_check(step_dict)
        if result_check.get("enabled") and result_check.get("timing") == "after_completion":
            return True
        wiring = self._step_wiring_config(step_idx, step_dict)
        return bool(
            wiring.get("enabled")
            and wiring.get("step_mode") == "after_completion"
        )

    def _activate_post_completion_check(self, step_idx, group_indices=None):
        if not (0 <= step_idx < len(self.process_steps)):
            return False
        step_dict = self.process_steps[step_idx]
        wiring = self._step_wiring_config(step_idx, step_dict)
        visual = self._step_result_check(step_dict)
        if wiring.get("enabled") and wiring.get("step_mode") == "after_completion":
            check_type = "wiring"
            name = wiring.get("stage_name", f"步骤{step_idx + 1}接线终态验收")
            self.step_wiring_monitor.configure([wiring])
            self.active_step_wiring_idx = step_idx
        elif visual.get("enabled") and visual.get("timing") == "after_completion":
            check_type = "visual"
            name = f"步骤{step_idx + 1}最终结果验收"
            self.step_result_hit_counts.pop(step_idx, None)
            self.step_result_status_by_idx.pop(step_idx, None)
        else:
            return False
        self.post_completion_check = {
            "step_idx": step_idx,
            "group_indices": list(group_indices or []),
            "type": check_type,
            "name": name,
            "started_at": time.time(),
            "progress": 0,
            "satisfied": False,
            "message": "工序动作已完成100%，正在进行独立附加条件验收",
        }
        self.current_step_idx = min(group_indices) if group_indices else step_idx
        self.step_start_time = time.time()
        return True

    def _evaluate_post_completion_check(self, detections):
        runtime = self.post_completion_check
        if not runtime:
            return False, ""
        step_idx = int(runtime["step_idx"])
        step_dict = self.process_steps[step_idx]
        if runtime.get("type") == "wiring":
            status = dict(self.step_wiring_status or {})
            satisfied = bool(
                status.get("configured")
                and status.get("step_idx") == step_idx
                and status.get("satisfied")
            )
            check_progress = int(status.get("progress", 0) or 0)
            expected = " → ".join(status.get("expected_order", []) or []) or "已配置终态"
            actual = " → ".join(
                status.get("actual_order", []) or status.get("visible_order", []) or []
            ) or "等待线缆完整出现"
            message = f"独立接线验收：目标 {expected}；当前 {actual}"
        else:
            satisfied, check_progress = self._apply_visual_result_gate(
                step_idx, step_dict, detections, (True, 100)
            )
            status = dict(self.step_result_status_by_idx.get(step_idx, {}) or {})
            message = status.get("message", "正在检查最终视觉状态")
        runtime.update({
            "progress": min(100, int(check_progress or 0)),
            "satisfied": bool(satisfied),
            "message": message,
        })
        if not satisfied:
            return False, f"🧪 步骤 {step_idx + 1} 已完成100%，{message}"

        cp = self.ng_tracker.current_product
        if cp and step_idx < len(cp.get("step_records", [])):
            cp["step_records"][step_idx]["post_completion_check"] = {
                "passed": True,
                "type": runtime.get("type"),
                "passed_at": datetime.now().strftime("%H:%M:%S"),
            }
        group_indices = list(runtime.get("group_indices") or [])
        self.post_completion_check = None
        self.step_result_hit_counts.pop(step_idx, None)
        self.step_result_status_by_idx.pop(step_idx, None)
        self.step_post_action_latched.discard(step_idx)
        self.step_wiring_monitor.configure([])
        self.active_step_wiring_idx = None
        if group_indices:
            self._schedule_group_final_check(group_indices, detections)
            self.current_step_idx = max(group_indices) + 1
        else:
            self.current_step_idx = step_idx + 1
        self.step_start_time = time.time()
        if self.current_step_idx >= len(self.process_steps):
            self._finish_and_restart_cycle()
            return True, "✅ 独立附加条件验收通过，当前产品工序全部完成"
        return True, f"✅ 步骤 {step_idx + 1} 独立附加条件验收通过，进入下一工序"

    def _apply_visual_result_gate(self, step_idx, step_dict, detections, result):
        """Use the stable final visual state as the authoritative completion result.

        The action detector is only supporting evidence.  This lets a completed
        assembly pass after an occluded action, while preventing a detected hand
        action from passing when the visible final state is still wrong.
        """
        config = self._step_result_check(step_dict)
        if not config["enabled"]:
            self.step_result_status_by_idx.pop(step_idx, None)
            self.step_result_hit_counts.pop(step_idx, None)
            return result

        action_done, action_progress = result
        target_text = config["target"]
        relation_mode = config["mode"] in {
            "targets_intersect", "targets_separate"
        }
        if relation_mode:
            targets = [
                target for target in self._targets_for_step(step_dict)
                if not self.engine.target_is_hand(target)
            ][:2]
            target_text = " 与 ".join(
                self.engine.target_display_name(target, self.engine.eng_to_zh)
                for target in targets
            ) or "本步骤两个目标"
            present = bool(
                len(targets) == 2
                and self.engine.check_presence(targets, detections)
            )
            intersects = bool(
                present and self.engine.target_groups_intersect(
                    [targets[0]], [targets[1]], detections, 0.0
                )
            )
            condition_met = bool(
                present and (
                    intersects
                    if config["mode"] == "targets_intersect"
                    else not intersects
                )
            )
        else:
            targets = self.engine.parse_step_text(target_text) or (
                [target_text] if target_text else []
            )
            present = bool(targets and self.engine.check_presence(targets, detections))
            intersects = False
            condition_met = present if config["mode"] == "present" else not present
        hand_in_zone = False
        require_release = bool(self._requires_hand_release(step_dict))
        if require_release or config["mode"] in {"absent", "targets_separate"}:
            action_targets = self._targets_for_step(step_dict)
            if action_targets:
                hand_in_zone = self.engine.hand_in_operation_zone(
                    action_targets, detections, 0.10
                )
                if hand_in_zone:
                    condition_met = False
        hits = int(self.step_result_hit_counts.get(step_idx, 0))
        if condition_met and targets:
            hits += 1
        else:
            hits = 0
        self.step_result_hit_counts[step_idx] = hits
        required_hits = int(config["stable_frames"])
        if require_release:
            required_hits = max(
                required_hits, int(self._hand_release_frames(step_dict))
            )
        satisfied = bool(targets and hits >= required_hits)
        if config["mode"] == "targets_intersect":
            state_text = "已装配/相交" if intersects else "尚未装配/相交"
            expected_text = "应已装配/相交"
        elif config["mode"] == "targets_separate":
            state_text = "仍相交" if intersects else "已分离"
            expected_text = "应已分离"
        else:
            state_text = "已识别到" if present else "未识别到"
            expected_text = "应识别到" if config["mode"] == "present" else "应识别不到"
        self.step_result_status_by_idx[step_idx] = {
            "configured": True,
            "target": target_text,
            "mode": config["mode"],
            "present": present,
            "intersects": intersects,
            "condition_met": condition_met,
            "hand_in_zone": hand_in_zone,
            "action_done": bool(action_done),
            "hits": hits,
            "required_hits": required_hits,
            "satisfied": satisfied,
            "message": f"最终结果：{state_text}【{target_text}】，{expected_text}",
        }
        if satisfied:
            return True, 100
        gate_progress = int(round(hits / required_hits * 99))
        return False, max(
            min(99, int(action_progress or 0)), min(99, gate_progress)
        )

    def _step_wiring_config(self, step_idx, step_dict=None):
        if step_idx in self.step_wiring_configs:
            config = self.step_wiring_configs[step_idx]
        else:
            raw = (step_dict or {}).get("step_wiring_check", {})
            config = normalize_step_wiring_check(raw, step_idx)
        if step_dict and self._requires_hand_release(step_dict):
            config = dict(config)
            config["release_frames"] = max(
                int(config.get("release_frames", 1) or 1),
                int(self._hand_release_frames(step_dict)),
            )
        return config

    def _unordered_group_wiring_config(self, group_indices):
        """Return the one post-action wiring check shared by an unordered group."""
        for index in group_indices or []:
            if not (0 <= index < len(self.process_steps)):
                continue
            config = self._step_wiring_config(index, self.process_steps[index])
            if config.get("enabled") and config.get("step_mode") in {
                "after_action", "after_completion"
            }:
                return index, config
        return None, None

    def _evaluate_step_wiring_gate(self, step_idx, step_dict, result):
        config = self._step_wiring_config(step_idx, step_dict)
        if not config.get("enabled"):
            return result
        action_done, action_progress = result
        after_action = config.get("step_mode") == "after_action"
        if after_action:
            if action_done:
                self.step_post_action_latched.add(step_idx)
            if step_idx not in self.step_post_action_latched:
                # 附加验收不能替代本工序原本的动作判定。动作尚未完成时，
                # 进度完全沿用原动作，接线布局即使已经部分/全部可见也不参与计算。
                return result
        status = dict(self.step_wiring_status or {})
        satisfied = bool(
            status.get("configured")
            and status.get("step_idx") == step_idx
            and status.get("satisfied")
        )
        if satisfied:
            return True, 100
        if after_action:
            # 原动作已经完成，固定显示为“等待最终验收”，避免接线布局的
            # 1/2、2/2 可见度把原动作进度错误地改成 45% 或 90%。
            return False, 99
        return False, min(99, max(
            min(99, int(action_progress or 0)), int(status.get("progress", 0))
        ))

    def _apply_step_completion_gates(
        self, step_idx, step_dict, detections, result, suppress_wiring=False
    ):
        result_check_enabled = bool(
            self._step_result_check(step_dict).get("enabled")
            and self._step_result_check(step_dict).get("timing") != "after_completion"
        )
        wiring_config = self._step_wiring_config(step_idx, step_dict)
        wiring_enabled = bool(
            wiring_config.get("enabled")
            and wiring_config.get("step_mode") != "after_completion"
            and not suppress_wiring
        )
        legacy_slot_gate = bool(
            str(step_dict.get("slot_expectation", "") or "").strip()
            and step_dict.get("slot_expectation_block", True)
        )
        slot_result = self._apply_slot_expectation_gate(step_dict, result)
        final_result_gates = []
        if legacy_slot_gate:
            final_result_gates.append(slot_result)
        if result_check_enabled:
            final_result_gates.append(
                self._apply_visual_result_gate(
                    step_idx, step_dict, detections, result
                )
            )
        else:
            self.step_result_status_by_idx.pop(step_idx, None)
            self.step_result_hit_counts.pop(step_idx, None)
        if wiring_enabled:
            final_result_gates.append(
                self._evaluate_step_wiring_gate(step_idx, step_dict, result)
            )
        if not final_result_gates:
            return slot_result
        completed = all(done for done, _progress in final_result_gates)
        if completed:
            return True, 100
        return False, min(99, min(
            int(progress or 0) for _done, progress in final_result_gates
        ))

    def _evaluate_step_by_config(
        self, step_idx, step_dict, targets, detections, engine,
        suppress_wiring=False,
    ):
        if not self._explicit_prereq_satisfied(step_idx, step_dict):
            return False, 0

        if self._is_step_cooling_down(step_idx, step_dict):
            return False, 0

        expectation = str(step_dict.get("slot_expectation", "") or "").strip()
        if expectation and bool(step_dict.get("slot_expectation_only", False)):
            status = self.slot_monitor.evaluate_expectation(expectation)
            self.slot_expectation_status = status
            if status.get("satisfied", False):
                return True, 100
            if not status.get("settled", False):
                return False, 0
            expected = status.get("expected", {}) or {}
            actual = status.get("actual", {}) or {}
            matched = sum(
                1 for slot_name, target in expected.items()
                if actual.get(slot_name) == target
            )
            progress = int(round(matched / max(1, len(expected)) * 100))
            return False, min(99, progress)

        wiring_config = self._step_wiring_config(step_idx, step_dict)
        if (not suppress_wiring and wiring_config.get("enabled")
                and wiring_config.get("step_mode") == "result_only"):
            return self._evaluate_step_wiring_gate(step_idx, step_dict, (True, 100))

        difficulty = step_dict.get("difficulty", "中等")
        if self._is_hand_touch_step(step_dict):
            self.step_action_status_by_idx.pop(step_idx, None)
            result = engine.evaluate_hand_touch_step(
                targets, detections, difficulty,
                stable_frames_override=self._step_stable_frames(step_dict),
                padding_ratio_override=self._step_padding_ratio(step_dict),
                require_hand_release=self._requires_hand_release(step_dict),
                hand_release_padding=self._hand_release_padding(step_dict),
                hand_release_frames=self._hand_release_frames(step_dict),
            )
            release_status = dict(getattr(engine, "release_status", {}) or {})
            if release_status:
                self.step_release_status_by_idx[step_idx] = release_status
            else:
                self.step_release_status_by_idx.pop(step_idx, None)
            return self._apply_step_completion_gates(
                step_idx, step_dict, detections, result,
                suppress_wiring=suppress_wiring,
            )
        if self._is_detach_step(step_dict):
            self.step_action_status_by_idx.pop(step_idx, None)
            self.step_release_status_by_idx.pop(step_idx, None)
            result = engine.evaluate_detach_step(
                targets, detections, difficulty,
                assembly_ready=self._detach_prereq_satisfied(step_idx),
                stable_frames_override=self._step_stable_frames(step_dict),
                padding_ratio_override=self._step_padding_ratio(step_dict),
                missing_enabled=self._detach_missing_enabled(step_dict),
                missing_frames=self._detach_missing_frames(step_dict),
                missing_padding_ratio=self._detach_missing_padding(step_dict),
            )
            detach_status = dict(getattr(engine, "detach_status", {}) or {})
            if detach_status:
                removed_target = detach_status.get("removed_target")
                base_target = detach_status.get("base_target")
                detach_status["removed_name"] = self.engine.target_display_name(removed_target, self.engine.eng_to_zh)
                detach_status["base_name"] = self.engine.target_display_name(base_target, self.engine.eng_to_zh)
                self.step_detach_status_by_idx[step_idx] = detach_status
            return self._apply_step_completion_gates(
                step_idx, step_dict, detections, result,
                suppress_wiring=suppress_wiring,
            )
        action_confirm_frames = self._step_action_confirm_frames(step_dict)
        result = engine.evaluate_step(
            targets,
            detections,
            difficulty,
            step_dict.get("count", 1),
            step_dict.get("multi_strategy", "lock"),
            action_gate_enabled=action_confirm_frames > 0,
            action_touch_frames=action_confirm_frames,
            stable_frames_override=self._step_stable_frames(step_dict),
            padding_ratio_override=self._step_padding_ratio(step_dict),
            require_hand_release=self._requires_hand_release(step_dict),
            hand_release_padding=self._hand_release_padding(step_dict),
            hand_release_frames=self._hand_release_frames(step_dict),
        )
        action_status = dict(getattr(engine, "action_status", {}) or {})
        if action_status:
            self.step_action_status_by_idx[step_idx] = action_status
        else:
            self.step_action_status_by_idx.pop(step_idx, None)
        release_status = dict(getattr(engine, "release_status", {}) or {})
        if release_status:
            self.step_release_status_by_idx[step_idx] = release_status
        else:
            self.step_release_status_by_idx.pop(step_idx, None)
        return self._apply_step_completion_gates(
            step_idx, step_dict, detections, result,
            suppress_wiring=suppress_wiring,
        )

    def _apply_slot_expectation_gate(self, step_dict, result):
        """Optionally turn a normal action step into a final-layout acceptance step."""
        expectation = str(step_dict.get("slot_expectation", "") or "").strip()
        if not expectation:
            return result
        status = self.slot_monitor.evaluate_expectation(expectation)
        self.slot_expectation_status = status
        if not bool(step_dict.get("slot_expectation_block", True)):
            return result
        done, progress = result
        if done and not status.get("satisfied", False):
            return False, min(99, int(progress or 0))
        return result

    def _active_slot_expectation_step(self):
        if self.remediation_mode and self.remediation_step_idx is not None:
            index = self.remediation_step_idx
        else:
            index = self.current_step_idx
        if 0 <= index < len(self.process_steps):
            return self.process_steps[index]
        return None

    def _active_runtime_step_index(self):
        if self.remediation_mode and self.remediation_step_idx is not None:
            return self.remediation_step_idx
        return self.current_step_idx

    def _update_active_step_wiring(self, detections):
        runtime_step_idx = self._active_runtime_step_index()
        step_idx = runtime_step_idx
        config = None
        group_indices = []
        group_wiring = False
        if not self.remediation_mode:
            group_indices = self._unordered_group_indices(runtime_step_idx)
            group_config_idx, group_config = self._unordered_group_wiring_config(
                group_indices
            )
            if group_config is not None:
                step_idx = group_config_idx
                config = group_config
                group_wiring = True
        if config is None and 0 <= step_idx < len(self.process_steps):
            candidate = self._step_wiring_config(step_idx, self.process_steps[step_idx])
            if candidate.get("enabled"):
                config = candidate

        if config is None:
            if self.active_step_wiring_idx is not None:
                self.step_wiring_monitor.configure([])
            self.active_step_wiring_idx = None
            self.step_wiring_status = {
                "configured": False, "satisfied": True, "settled": True,
                "mismatches": [], "progress": 100,
            }
            return {"status": None, "alarms": []}

        if self.active_step_wiring_idx != step_idx:
            self.step_wiring_monitor.configure([config])
            self.active_step_wiring_idx = step_idx

        group_actions_done = bool(
            group_wiring
            and group_indices
            and all(self._step_record_status(index) != "pending" for index in group_indices)
        )
        waiting_phase = None
        if group_wiring and not group_actions_done:
            waiting_phase = "waiting_group_actions"
        elif not group_wiring and config.get("step_mode") == "after_action":
            if step_idx not in self.step_post_action_latched:
                waiting_phase = "waiting_action"
        elif not group_wiring and config.get("step_mode") == "after_completion":
            active_post_check = bool(
                self.post_completion_check
                and self.post_completion_check.get("step_idx") == step_idx
            )
            if not active_post_check:
                waiting_phase = "waiting_completion"

        if waiting_phase:
            # “附加到本工序”是严格的后置检查：先让原动作状态机独立完成，
            # 乱序组则要等组内所有动作完成，之后才采集接线顺序和错误报警。
            if (self.step_wiring_status.get("step_idx") != step_idx
                    or self.step_wiring_status.get("phase") != waiting_phase):
                self.step_wiring_monitor.configure([config])
            status = {
                "configured": True,
                "step_idx": step_idx,
                "group_indices": list(group_indices) if group_wiring else [],
                "name": (
                    f"乱序组接线验收（步骤{group_indices[0] + 1}-{group_indices[-1] + 1}）"
                    if group_wiring else
                    config.get("stage_name", f"步骤{step_idx + 1}接线验收")
                ),
                "phase": waiting_phase,
                "expected_order": list(config.get("expected_targets", []) or []),
                "actual_order": [],
                "settled": False,
                "satisfied": False,
                "mismatches": [],
                "progress": 0,
            }
            self.step_wiring_status = status
            return {"status": status, "alarms": []}

        result = self.step_wiring_monitor.update(detections, self.engine)
        statuses = list(result.get("statuses", []))
        raw_status = statuses[0] if statuses else {}
        expected = list(config.get("expected_targets", []) or [])
        if config.get("monitor_type") == "relative_order":
            actual = list(raw_status.get("actual_order", []) or [])
            visible = list(raw_status.get("live_order", []) or [])
        else:
            slots = raw_status.get("slots", {}) or {}
            actual = [slots.get(name, "未知") for name in config.get("slot_names", [])]
            visible = [value for value in actual if value not in ("", "未知", "未知槽位")]
        settled = raw_status.get("phase") == "monitoring"
        satisfied = bool(settled and len(actual) == len(expected) and actual == expected)
        matched = sum(
            1 for actual_target, expected_target in zip(actual, expected)
            if actual_target == expected_target
        )
        if satisfied:
            progress = 100
        elif settled and actual:
            progress = min(99, int(round(matched / max(1, len(expected)) * 99)))
        else:
            progress = min(90, int(round(len(visible) / max(1, len(expected)) * 90)))
        mismatches = [] if satisfied or not settled else [
            f"当前为{' → '.join(actual) or '未知'}，正确应为{' → '.join(expected)}"
        ]
        status = dict(raw_status)
        status.update({
            "configured": True,
            "step_idx": step_idx,
            "group_indices": list(group_indices) if group_wiring else [],
            "name": (
                f"乱序组接线验收（步骤{group_indices[0] + 1}-{group_indices[-1] + 1}）"
                if group_wiring else
                config.get("stage_name", f"步骤{step_idx + 1}接线验收")
            ),
            "expected_order": expected,
            "actual_order": actual,
            "settled": settled,
            "satisfied": satisfied,
            "mismatches": mismatches,
            "progress": progress,
        })
        self.step_wiring_status = status

        explicit_wrong = {
            tuple(combination) for combination in config.get("wrong_combinations", [])
        }
        alarm_armed = True
        wrong = bool(
            alarm_armed
            and
            settled and actual != expected and (
                tuple(actual) in explicit_wrong
                or config.get("alarm_all_mismatches", False)
            )
        )
        alarms = []
        if wrong:
            mode = config.get("alarm_mode", "red_buzzer")
            alarms.append({
                "key": f"step_wiring_{step_idx}",
                "message": (
                    f"步骤{step_idx + 1}接线结果错误：当前"
                    f"{' → '.join(actual) or '未识别完整'}，正确应为"
                    f"{' → '.join(expected)}"
                ),
                "buzzer": mode != "red_only",
                "alarm_mode": mode,
            })
        return {"status": status, "alarms": alarms}

    def _update_continuous_monitors(self, detections):
        state_result = self.toggle_state_monitor.update(detections, self.engine)
        self.toggle_state_statuses = list(state_result.get("statuses", []))
        events = list(state_result.get("events", []))
        if events:
            self.monitor_event_log.extend(events)
            self.monitor_event_log = self.monitor_event_log[-200:]
            current_product = self.ng_tracker.current_product
            if current_product is not None:
                current_product.setdefault("runtime_state_events", []).extend(events)

        slot_result = self.slot_monitor.update(detections, self.engine)
        self.slot_monitor_statuses = list(slot_result.get("statuses", []))
        step_wiring_result = self._update_active_step_wiring(detections)
        if step_wiring_result.get("status"):
            self.slot_monitor_statuses.append(step_wiring_result["status"])
        sequence_result = self.result_sequence_monitor.update(detections, self.engine)
        self.result_sequence_statuses = list(sequence_result.get("statuses", []))
        sequence_events = list(sequence_result.get("events", []))
        if sequence_events:
            self.monitor_event_log.extend(sequence_events)
            self.monitor_event_log = self.monitor_event_log[-200:]
        active_step = self._active_slot_expectation_step()
        expectation = active_step.get("slot_expectation", "") if active_step else ""
        self.slot_expectation_status = self.slot_monitor.evaluate_expectation(expectation)
        combined_result = dict(state_result)
        combined_result["alarms"] = (
            list(state_result.get("alarms", []))
            + list(slot_result.get("alarms", []))
            + list(step_wiring_result.get("alarms", []))
            + list(sequence_result.get("alarms", []))
        )
        return combined_result

    def _observed_slot_mismatch(self):
        """Alarm only for an observed wrong connector, not for a still-empty slot."""
        status = self.slot_expectation_status or {}
        if not status.get("configured") or not status.get("settled") or status.get("satisfied"):
            return None
        expected = status.get("expected", {}) or {}
        actual = status.get("actual", {}) or {}
        mismatches = []
        for slot_name, expected_target in expected.items():
            actual_target = actual.get(slot_name, "未知槽位")
            if actual_target not in ("空", "未知", "未知槽位", "") and actual_target != expected_target:
                mismatches.append(f"{slot_name}：当前{actual_target}，应为{expected_target}")
        if not mismatches:
            return None
        monitor_name = status.get("monitor_name") or status.get("monitor_id") or "槽位"
        return f"接线布局错误【{monitor_name}】：" + "；".join(mismatches)

    def _record_aoi_status(self, step_idx, state=None, similarity=None, threshold=None, best_angle=None):
        cp = self.ng_tracker.current_product
        if not cp or step_idx is None or step_idx >= len(cp.get('step_records', [])):
            return
        rec = cp['step_records'][step_idx]
        if state is not None:
            rec['aoi_state'] = state
        if similarity is not None:
            rec['aoi_similarity'] = float(similarity)
        if threshold is not None:
            rec['aoi_threshold'] = float(threshold)
        if best_angle is not None:
            rec['aoi_best_angle'] = float(best_angle)

    def _start_aoi_check(self, step_idx, aoi_cfg, context="normal"):
        self.aoi_state = 'finding_anchor'
        self.aoi_step_idx = step_idx
        self.aoi_context = context
        self.aoi_check_start_time = time.time()
        self.aoi_stable_count = 0
        self.aoi_similarity = 0.0
        self.aoi_anchor_class = aoi_cfg['anchor_class']
        self.aoi_threshold = aoi_cfg.get('threshold', 0.85)
        self.aoi_timeout = aoi_cfg.get('timeout', 5.0)
        self.aoi_standard_vector = np.array(aoi_cfg['standard_vector'], dtype=np.float32)
        self._record_aoi_status(step_idx, 'finding_anchor', 0.0, self.aoi_threshold)

    def _remember_unordered_completion(self, step_idx):
        if self._step_order_group(self.process_steps[step_idx]) and step_idx not in self.unordered_completion_order:
            self.unordered_completion_seq += 1
            self.unordered_completion_order[step_idx] = self.unordered_completion_seq

    def _advance_completed_step_or_start_post_check(self, step_idx, detections=None):
        """Advance a completed action, or expose its configured post-completion audit."""
        step_dict = self.process_steps[step_idx]
        self.step_progress_by_idx[step_idx] = 100
        if self._step_order_group(step_dict):
            self._remember_unordered_completion(step_idx)
            return self._sync_current_idx_for_unordered_group(
                self._unordered_group_indices(step_idx), detections
            )
        if self._post_completion_required(step_idx, step_dict):
            self._activate_post_completion_check(step_idx)
            return False
        self.current_step_idx = step_idx + 1
        return self.current_step_idx >= len(self.process_steps)

    def _sync_current_idx_for_unordered_group(self, group_indices, detections=None):
        pending = [i for i in group_indices if self._step_record_status(i) == 'pending']
        if pending:
            self.current_step_idx = pending[0]
            self.step_start_time = time.time()
            return False
        wiring_step_idx, wiring_config = self._unordered_group_wiring_config(group_indices)
        if wiring_config is not None:
            if (wiring_config.get("step_mode") == "after_completion"
                    and not self.post_completion_check):
                self._activate_post_completion_check(
                    wiring_step_idx, group_indices=group_indices
                )
            status = dict(self.step_wiring_status or {})
            wiring_satisfied = bool(
                status.get("configured")
                and status.get("step_idx") == wiring_step_idx
                and status.get("satisfied")
            )
            if not wiring_satisfied:
                # 完成后验收模式中，组内动作已经分别记为100%；这里只保留
                # 一个独立验收阶段，不再把某个动作伪装成99%。
                self.current_step_idx = min(group_indices)
                self.step_start_time = time.time()
                return False
        self._schedule_group_final_check(group_indices, detections)
        self.current_step_idx = max(group_indices) + 1
        if self.current_step_idx >= len(self.process_steps):
            self._finish_and_restart_cycle()
            return True
        self.step_start_time = time.time()
        return False

    def _evaluate_unordered_group(self, detections):
        group_indices = self._unordered_group_indices(self.current_step_idx)
        if not group_indices:
            return False, 0, ""
        group_wiring_idx, group_wiring_config = self._unordered_group_wiring_config(
            group_indices
        )

        best_progress = 0
        best_progress_idx = None
        completed_any = False
        evaluation_indices = list(group_indices)
        if (self.unordered_active_idx in evaluation_indices
                and self._step_record_status(self.unordered_active_idx) == 'pending'):
            evaluation_indices.remove(self.unordered_active_idx)
            evaluation_indices.insert(0, self.unordered_active_idx)

        for idx in evaluation_indices:
            if self._step_record_status(idx) != 'pending':
                continue
            step_dict = self.process_steps[idx]
            targets = self._targets_for_step(step_dict)
            wiring_config = self._step_wiring_config(idx, step_dict)
            wiring_only = bool(
                wiring_config.get("enabled")
                and wiring_config.get("step_mode") == "result_only"
            )
            final_result_enabled = bool(
                self._step_result_check(step_dict).get("enabled")
                or wiring_config.get("enabled")
            )
            if not targets and not wiring_only and not final_result_enabled:
                continue
            if idx not in self.unordered_step_engines:
                eng = ProcessLogicEngine()
                eng.lookup_dict = self.engine.lookup_dict
                eng.eng_to_zh = self.engine.eng_to_zh
                eng.regex_pattern = self.engine.regex_pattern
                self.unordered_step_engines[idx] = eng
            is_done, step_progress = self._evaluate_step_by_config(
                idx, step_dict, targets, detections, self.unordered_step_engines[idx],
                suppress_wiring=group_wiring_config is not None,
            )
            self.step_progress_by_idx[idx] = step_progress
            action_state = self.step_action_status_by_idx.get(idx, {}).get("state")
            if step_progress > best_progress:
                best_progress = step_progress
                best_progress_idx = idx
            elif best_progress_idx is None and action_state in ("arming", "armed"):
                best_progress_idx = idx
            if is_done:
                self.step_progress_by_idx[idx] = 100
                self.unordered_step_engines[idx].reset()
                self.step_post_action_latched.discard(idx)
                self.step_result_hit_counts.pop(idx, None)
                # 一次实际动作只能完成一个乱序成员。其他成员的影子状态必须清零，
                # 防止多个成员同时到 99% 后，一次离手把整组一起判成完成。
                for other_idx in group_indices:
                    if other_idx == idx or self._step_record_status(other_idx) != 'pending':
                        continue
                    other_engine = self.unordered_step_engines.get(other_idx)
                    if other_engine is not None:
                        other_engine.reset()
                    self.step_progress_by_idx[other_idx] = 0
                    self.step_action_status_by_idx.pop(other_idx, None)
                    self.step_release_status_by_idx.pop(other_idx, None)
                    self.step_detach_status_by_idx.pop(other_idx, None)
                aoi_cfg = step_dict.get('aoi_feature_check', {})
                if aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
                    self._start_aoi_check(idx, aoi_cfg)
                    return False, 100, f"步骤 {idx + 1} 动作完成，正在进入 AOI 比对"

                self.ng_tracker.mark_step_completed(idx)
                self.ng_tracker._check_and_restore_ok()
                self._remember_unordered_completion(idx)
                self.workflow_monitor.reset_runtime(clear_restart_guard=True)
                self.unordered_active_idx = None
                completed_any = True
                break

        self.unordered_active_idx = None if completed_any else best_progress_idx
        finished_group = False
        if completed_any:
            finished_group = self._sync_current_idx_for_unordered_group(group_indices, detections)
        if finished_group:
            return True, 100, "乱序组已完成"

        pending_count = sum(1 for i in group_indices if self._step_record_status(i) == 'pending')
        if pending_count == 0 and group_wiring_config is not None:
            status = dict(self.step_wiring_status or {})
            wiring_satisfied = bool(
                status.get("configured")
                and status.get("step_idx") == group_wiring_idx
                and status.get("satisfied")
            )
            if not wiring_satisfied:
                self.current_step_idx = min(group_indices)
                self.unordered_active_idx = None
                if group_wiring_config.get("step_mode") == "after_completion":
                    if not self.post_completion_check:
                        self._activate_post_completion_check(
                            group_wiring_idx, group_indices=group_indices
                        )
                    return False, 100, "乱序组动作已全部完成100%，正在进行独立接线验收"
                return False, 99, "乱序组动作已全部完成，正在检查最终接线状态"
            if not completed_any:
                finished_group = self._sync_current_idx_for_unordered_group(
                    group_indices, detections
                )
            return finished_group, 100, "乱序组动作及最终接线验收已完成"
        alert = ""
        if pending_count > 0:
            alert = f"乱序组待完成: {pending_count}/{len(group_indices)}"
        return finished_group, best_progress, alert

    def _clone_parser_to_engine(self, engine):
        engine.lookup_dict = self.engine.lookup_dict
        engine.eng_to_zh = self.engine.eng_to_zh
        engine.regex_pattern = self.engine.regex_pattern

    def _reset_jump_monitors(self, clear_restart_guard=False, preserve_alarm=False):
        self.final_step_engine.reset()
        self.final_sub_count = 0
        self.final_is_pausing = False
        self.final_pause_start_time = 0
        self.final_last_action_time = 0
        self.workflow_monitor.reset_runtime(
            clear_restart_guard=clear_restart_guard,
            preserve_alarm=preserve_alarm,
        )

    def _activate_restart_guard(self, preserve_alarm=False):
        self.workflow_monitor.arm_restart_guard(
            frames=75,
            preserve_alarm=preserve_alarm,
        )

    def _clear_restart_guard(self):
        self.workflow_monitor.restart_guard_active = False
        self.workflow_monitor.restart_guard_frames = 0
        self.workflow_monitor.restart_guard_idle_frames = 0

    def _enter_remediation_mode(self, step_idx):
        if not self.ng_tracker.current_product or step_idx not in self.ng_tracker.get_skipped_indices():
            self.remediation_status_msg = "没有可补救的跳过步骤。"
            return False
        if step_idx < 0 or step_idx >= len(self.process_steps):
            self.remediation_status_msg = "补救步骤不存在。"
            return False
        self.remediation_mode = True
        self.remediation_step_idx = step_idx
        self.remediation_resume_idx = min(self.current_step_idx, len(self.process_steps))
        self.remediation_status_msg = f"补救中：请完成步骤 {step_idx + 1}，完成前不会回到正常工序。"
        if step_idx not in self.remediation_engines:
            self.remediation_engines[step_idx] = ProcessLogicEngine()
        self._clone_parser_to_engine(self.remediation_engines[step_idx])
        self.remediation_engines[step_idx].reset()
        self.engine.reset()
        self.is_pausing = False
        self.current_sub_count = 0
        self._clear_step_cooldown(step_idx)
        self._reset_jump_monitors()
        self.step_start_time = time.time()
        return True

    def _exit_remediation_mode(self, message=""):
        if self.remediation_step_idx in self.remediation_engines:
            self.remediation_engines[self.remediation_step_idx].reset()
        self.remediation_mode = False
        self.remediation_step_idx = None
        self.remediation_status_msg = message
        self.current_step_idx = min(self.remediation_resume_idx, len(self.process_steps))
        self.remediation_resume_idx = self.current_step_idx
        self.engine.reset()
        self.is_pausing = False
        self.current_sub_count = 0
        self.step_start_time = time.time()
        self._reset_aoi_runtime()
        self._reset_jump_monitors()
        self.aoi_update_signal.emit(0.0, '', False)

    def _finish_remediation_step(self, step_idx, message_prefix="补救完成"):
        self.ng_tracker.mark_step_remedied(step_idx)
        self.ng_tracker._check_and_restore_ok()
        self.step_progress_by_idx[step_idx] = 100
        self._clear_step_runtime_flags(step_idx)
        reset_steps = self._reset_prereq_violation_dependents(step_idx)
        if reset_steps:
            first_reset = min(reset_steps)
            self.remediation_resume_idx = first_reset
            message = (
                f"{message_prefix}：步骤 {step_idx + 1} 已补救；"
                f"步骤 {first_reset + 1} 需重新执行，已回到该步骤。"
            )
        else:
            message = f"{message_prefix}：步骤 {step_idx + 1} 已补救，已回到正常工序。"
        self._exit_remediation_mode(message)

    def _reset_prereq_violation_dependents(self, remedied_idx):
        cp = self.ng_tracker.current_product
        if not cp:
            return []
        reset_indices = []
        for idx, step_dict in enumerate(self.process_steps):
            if idx <= remedied_idx:
                continue
            if remedied_idx not in self._prerequisite_indices(step_dict):
                continue
            if idx >= len(cp.get('step_records', [])):
                continue
            rec = cp['step_records'][idx]
            if not rec.get('prereq_violation_completed'):
                continue
            if rec.get('status') not in ('completed', 'remedied'):
                continue
            rec['status'] = 'pending'
            rec['reason'] = '前置补救完成后需要重新执行'
            rec['reset_after_prereq_remediation'] = True
            rec['reset_at'] = time.strftime("%H:%M:%S")
            rec.pop('prereq_violation_completed', None)
            self.step_progress_by_idx[idx] = 0
            self._clear_step_runtime_flags(idx)
            reset_indices.append(idx)
        return reset_indices

    def _advance_after_jump(self, jumped_to_idx, jump_msg):
        if not self.process_steps:
            return False
        if jumped_to_idx < self.current_step_idx or jumped_to_idx >= len(self.process_steps):
            return False

        jumped_step = self.process_steps[jumped_to_idx]
        missing_prereqs = self._missing_prerequisite_indices(jumped_step)
        is_prereq_violation = bool(missing_prereqs)
        # 相同索引只允许处理“当前步骤前置未满足却被实际执行”的违规场景。
        if jumped_to_idx == self.current_step_idx and not is_prereq_violation:
            return False

        jumped_group = self._step_order_group(jumped_step)
        skipped_indices = []
        for idx in range(self.current_step_idx, jumped_to_idx):
            # 乱序组成员彼此不是“中间步骤”；完成组内某一道不能把同组其他成员跳过。
            if jumped_group and self._step_order_group(self.process_steps[idx]) == jumped_group:
                continue
            if self._step_record_status(idx) == 'pending':
                self.ng_tracker.mark_step_skipped(
                    idx,
                    f"前置条件未完成却执行步骤 {jumped_to_idx + 1}，本步骤被跳过"
                    if is_prereq_violation else
                    f"检测到先完成步骤 {jumped_to_idx + 1}，本步骤被跳过"
                )
                self.step_progress_by_idx[idx] = 0
                self._clear_step_runtime_flags(idx)
                skipped_indices.append(idx)

        if is_prereq_violation:
            for idx in missing_prereqs:
                if idx == jumped_to_idx or idx in skipped_indices:
                    continue
                if 0 <= idx < len(self.process_steps) and self._step_record_status(idx) == 'pending':
                    self.ng_tracker.mark_step_skipped(
                        idx,
                        f"前置条件未完成却执行步骤 {jumped_to_idx + 1}，本步骤被跳过"
                    )
                    self.step_progress_by_idx[idx] = 0
                    self._clear_step_runtime_flags(idx)
                    skipped_indices.append(idx)

        self.ng_tracker.add_jump_alarm(
            self.current_step_idx,
            jump_msg,
            jumped_to_idx=jumped_to_idx,
            skipped_indices=skipped_indices,
        )

        self._clear_step_runtime_flags(jumped_to_idx)
        self.current_sub_count = 0
        self.is_pausing = False
        self.engine.reset()
        # 工序状态可以立即重置，但刚触发的跳步告警必须继续锁存到 2 秒结束。
        self._reset_jump_monitors(preserve_alarm=True)

        self.step_progress_by_idx[jumped_to_idx] = 100

        aoi_cfg = jumped_step.get('aoi_feature_check', {})
        if not is_prereq_violation and aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
            self.current_step_idx = jumped_to_idx
            self._start_aoi_check(jumped_to_idx, aoi_cfg)
            self.step_start_time = time.time()
            return True

        self.ng_tracker.mark_step_completed(jumped_to_idx)
        if is_prereq_violation:
            cp = self.ng_tracker.current_product
            if cp and jumped_to_idx < len(cp.get('step_records', [])):
                rec = cp['step_records'][jumped_to_idx]
                rec['prereq_violation_completed'] = True
                rec['missing_prerequisite_steps'] = [idx + 1 for idx in missing_prereqs]
                rec['reason'] = f"前置条件未完成时提前执行：{jump_msg}"
        self._advance_completed_step_or_start_post_check(jumped_to_idx)

        self.step_start_time = time.time()
        if not self.post_completion_check and self.current_step_idx >= len(self.process_steps):
            self._finish_and_restart_cycle(preserve_jump_alarm=True)
        return True

    def _display_process_steps(self):
        cp = self.ng_tracker.current_product
        if not cp:
            return self.process_steps
        records = cp.get('step_records', [])

        def make_step(idx):
            step = self.process_steps[idx]
            step_copy = dict(step)
            step_copy['_display_step_num'] = idx + 1
            runtime_progress = int(self.step_progress_by_idx.get(idx, 0))
            group = self._step_order_group(step_copy)
            suppress_unordered_runtime = False
            if group:
                step_copy['_unordered_group'] = group
                if idx == self.unordered_active_idx:
                    step_copy['_unordered_active'] = True
                elif self._step_record_status(idx) == 'pending':
                    runtime_progress = 0
                    suppress_unordered_runtime = self.unordered_active_idx is not None
                if idx in self.unordered_completion_order:
                    step_copy['_unordered_done_order'] = self.unordered_completion_order[idx]
            if suppress_unordered_runtime:
                step_copy['_suppress_unordered_runtime'] = True
            step_copy['_runtime_progress'] = runtime_progress
            jump_shadow_progress = int(
                self.workflow_monitor.shadow_progress_by_idx.get(idx, 0) or 0
            )
            if (
                self.show_jump_progress
                and idx != self.current_step_idx
                and self._step_record_status(idx) == "pending"
                and jump_shadow_progress > 0
            ):
                step_copy['_jump_shadow_progress'] = jump_shadow_progress
            if idx < len(records):
                step_copy['_runtime_status'] = records[idx].get('status', 'pending')
                for key in ('aoi_similarity', 'aoi_threshold', 'aoi_state', 'aoi_best_angle'):
                    if key in records[idx]:
                        step_copy[f'_{key}'] = records[idx].get(key)
            if self.aoi_step_idx == idx and self.aoi_state:
                step_copy['_aoi_state'] = self.aoi_state
                step_copy['_aoi_similarity'] = float(self.aoi_similarity)
                step_copy['_aoi_threshold'] = float(self.aoi_threshold)
            if self.remediation_mode and self.remediation_step_idx == idx:
                step_copy['_remediation_active'] = True
            if not suppress_unordered_runtime and idx in self.step_detach_status_by_idx:
                step_copy['_detach_status'] = dict(self.step_detach_status_by_idx[idx])
            if not suppress_unordered_runtime and idx in self.step_action_status_by_idx:
                step_copy['_action_status'] = dict(self.step_action_status_by_idx[idx])
            if not suppress_unordered_runtime and idx in self.step_release_status_by_idx:
                step_copy['_release_status'] = dict(self.step_release_status_by_idx[idx])
            if not suppress_unordered_runtime and idx in self.step_cooldown_status_by_idx:
                step_copy['_cooldown_status'] = dict(self.step_cooldown_status_by_idx[idx])
            if not suppress_unordered_runtime and idx in self.step_prereq_status_by_idx:
                step_copy['_prereq_status'] = dict(self.step_prereq_status_by_idx[idx])
            return step_copy

        display_steps = []
        idx = 0
        while idx < len(self.process_steps):
            group = self._step_order_group(self.process_steps[idx])
            if group:
                group_indices = self._unordered_group_indices(idx)
                completed = [
                    i for i in group_indices
                    if self._step_record_status(i) != 'pending' or i in self.unordered_completion_order
                ]
                pending = [i for i in group_indices if i not in completed]
                completed.sort(key=lambda i: self.unordered_completion_order.get(i, 10**6))
                ordered = completed + pending
                for pos, original_idx in enumerate(ordered):
                    step_copy = make_step(original_idx)
                    step_copy['_group_open'] = pos == 0
                    step_copy['_group_close'] = pos == len(ordered) - 1
                    step_copy['_group_size'] = len(ordered)
                    display_steps.append(step_copy)
                idx = group_indices[-1] + 1
            else:
                display_steps.append(make_step(idx))
                idx += 1
        return display_steps

    def run(self):
        self.running = True
        self._reset_transient_requests()
        self.step_start_time = time.time()
        pipeline = None
        cap = None
        # 🌟 第一重异常拦截：初始化摄像头防崩溃
        try:
            if self.source_type == "realsense":
                if not HAS_REALSENSE:
                    raise RuntimeError("当前环境没有安装 pyrealsense2，无法打开 RealSense")
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color)
                pipeline.start(config)
            elif self.source_type in ("4k_cam", "webcam"):
                cap = cv2.VideoCapture(self.source)
                #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
                cap.set(cv2.CAP_PROP_FPS, 60)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开摄像头 (ID: {self.source})")
            else:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开视频源/摄像头 (ID或路径: {self.source})")
        except Exception as e:
            self._safe_stop_pipeline(pipeline)
            self._safe_release_capture(cap)
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.update_ui_signal.emit(empty_frame, [], 0, False, 0, f"❌ 摄像头开启失败: {str(e)}", 0, 0)
            self.running = False
            return
        frame_idx = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        # 🌟 NG 追踪：启动第一个产品
        if self.process_steps and not self.ng_tracker.current_product:
            self.ng_tracker.start_product(self.process_steps)
        while self.running:
            self.just_restarted_cycle = False
            # --- 接收 UI 层的干预指令 ---
            if self.reset_signal:
                if self.ng_tracker.current_product:
                    self.ng_tracker.finalize_as_ng('手动重新开始，当前产品未完成')
                self._reset_workflow_runtime()
                self.slot_monitor.reset(preserve_calibration=True)
                self.result_sequence_monitor.reset()
                self.slot_monitor_statuses = []
                self.result_sequence_statuses = []
                self.slot_expectation_status = {
                    "configured": False, "satisfied": True,
                    "settled": True, "mismatches": [],
                }
                self.reset_signal = False
                # 启动新产品追踪
                if self.process_steps:
                    self.ng_tracker.start_product(self.process_steps)
                self._activate_restart_guard()
            if self.force_skip_signal:
                if self.process_steps and self.current_step_idx < len(self.process_steps):
                    old_idx = self.current_step_idx
                    self.current_step_idx += 1
                    self.step_progress_by_idx[old_idx] = 0
                    self._clear_step_runtime_flags(old_idx)
                    # 🌟 NG 追踪：手动跳过 → 记录 NG
                    self.ng_tracker.on_step_advance(old_idx, '手动跳过')
                    self.current_sub_count = 0
                    self.engine.reset()
                    self._reset_jump_monitors()
                    self.is_pausing = False
                    self.step_start_time = time.time()
                    if self.current_step_idx >= len(self.process_steps):
                        self._finish_and_restart_cycle()
                self.force_skip_signal = False
            # --- 2. 读取视频帧 (🌟 第二重异常拦截：防中途拔出断连) ---
            frame = None
            try:
                if self.source_type == "realsense":
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        raise RuntimeError("RealSense 彩色画面获取失败！")
                    frame = np.asanyarray(color_frame.get_data())
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    if not cap.isOpened(): raise RuntimeError("摄像头对象已关闭")
                    ret, frame = cap.read()
                    if not ret or frame is None: raise RuntimeError("视频流异常中断，画面获取失败！")

                native_frame = frame
                frame = apply_frame_transform(native_frame, self.frame_transform)
                # 💡 性能优化建议：如果 4K 画面导致你的 PySide 界面卡顿或 MediaPipe 变慢，
                # 可以在这里加一行把处理分辨率降下来，比如：
                # if self.source_type == "4k_cam":
                #     frame = cv2.resize(frame, (1920, 1080))
                raw_frame = frame.copy()
            except Exception as e:
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.update_ui_signal.emit(empty_frame, [], 0, False, 0, f"❌ 画面中断: {str(e)}", 0, 0)
                break
            # --- 3. 视频加速逻辑 ---
            frame_idx += 1
            # 🌟 新增：计算真实处理帧率
            fps_frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now
            if self.source_type == "video" and self.speed_multiplier > 1:
                if frame_idx % self.speed_multiplier != 0:
                    continue
            # --- 确定 YOLO 需要检测的类别 ---
            active_class_ids = self.selected_class_ids.copy()
            name_to_id = self.name_to_id_cache  # 使用缓存，避免 ONNX 触发 CUDA 设备检测
            for hand_name in ['hand', 'glove', '手', '手套']:
                if hand_name in name_to_id and name_to_id[hand_name] not in active_class_ids:
                    active_class_ids.append(name_to_id[hand_name])
            if self.forbidden_targets:
                for f_target in self.forbidden_targets:
                    for option in self._target_options(f_target):
                        if option in name_to_id and name_to_id[option] not in active_class_ids:
                            active_class_ids.append(name_to_id[option])
            # 🌟 修复：遍历收集【所有步骤】的目标，防止后续步骤的物品隐身！
            all_required_targets = set()
            required_targets = []  # 保存当前步骤目标给后续变绿用
            if self.process_steps:
                # 1. 收集全局目标，全部交给 YOLO 检测
                for step_dict in self.process_steps:
                    targets = self._targets_for_step(step_dict)
                    all_required_targets.update(targets)
                    result_check = self._step_result_check(step_dict)
                    if result_check.get("enabled") and result_check.get("target"):
                        all_required_targets.update(
                            self.engine.parse_step_text(result_check["target"])
                            or [result_check["target"]]
                        )
                    group_final_config = step_dict.get("group_final_check")
                    if isinstance(group_final_config, dict) and group_final_config.get("enabled"):
                        for key in ("anchor", "expected", "wrong"):
                            all_required_targets.update(self.engine.parse_step_text(
                                str(group_final_config.get(key, "") or "")
                            ))
                # 2. 提取当前目标
                if self.current_step_idx < len(self.process_steps):
                    current_step_dict = self.process_steps[self.current_step_idx]
                    required_targets = self._targets_for_step(current_step_dict)

            # 独立状态/槽位监测不依赖线性工序，即使工序为空也必须激活相关类别。
            monitor_expressions = (
                self.toggle_state_monitor.required_target_expressions()
                + self.slot_monitor.required_target_expressions()
                + self.result_sequence_monitor.required_target_expressions()
            )
            for config in self.step_wiring_configs.values():
                if not config.get("enabled"):
                    continue
                monitor_expressions.append(config.get("anchor_target", ""))
                monitor_expressions.extend(config.get("connector_targets", []))
            for expression in monitor_expressions:
                all_required_targets.update(self.engine.parse_step_text(expression) or [])

            # 将所有工序与持续监测涉及的 ID 添加进 YOLO 白名单。
            for target in all_required_targets:
                for option in self._target_options(target):
                    class_id = name_to_id.get(option)
                    if class_id is not None and class_id not in active_class_ids:
                        active_class_ids.append(class_id)

            # 派生关系的输出是虚拟标签，不能作为 YOLO class ID；真正需要送进
            # YOLO 的是整体框和两个内部物体类别。
            if self.derived_obb_tracker.enabled:
                virtual_ids = {
                    name_to_id[name]
                    for name in self.derived_obb_tracker.virtual_outputs()
                    if name in name_to_id
                }
                active_class_ids = [
                    class_id for class_id in active_class_ids
                    if class_id not in virtual_ids
                ]
                for class_name in self.derived_obb_tracker.required_yolo_classes():
                    class_id = name_to_id.get(class_name)
                    if class_id is not None and class_id not in active_class_ids:
                        active_class_ids.append(class_id)
            # --- YOLO 推理 ---
            detections = []
            annotated_frame = frame.copy()
            if self.model is not None and len(active_class_ids) > 0:
                infer_kwargs = self._build_infer_kwargs(active_class_ids)
                try:
                    results = self.model(native_frame, **infer_kwargs)
                except AssertionError as e:
                    if "Invalid device id" in str(e) or "CUDA" in str(e):
                        self.infer_device = "cpu"
                        infer_kwargs['device'] = "cpu"
                        log_runtime_device(f"Inference CUDA failed, fallback to CPU: {e}")
                        results = self.model(native_frame, **infer_kwargs)
                    else:
                        raise
                # 读取 Ultralytics 实际 letterbox 后送入模型的尺寸，而不是原视频尺寸。
                self._update_yolo_input_size(native_frame)
                if results and len(results) > 0:
                    names = self.model_names
                    native_h, native_w = native_frame.shape[:2]
                    # 1. 获取 UI 面板上真实勾选的 ID（真正要画出来的）
                    ui_checked_names = [self.model_names[cid] for cid in self.selected_class_ids if
                                         cid in self.model_names]
                    # 2. 干净利落地提取检测结果
                    if getattr(results[0], 'boxes', None) is not None and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            draw_bbox = transform_bbox_to_display(
                                [x1, y1, x2, y2], self.frame_transform, native_w, native_h
                            )
                            conf = float(box.conf[0])
                            cls_name = names.get(cls_id, str(cls_id))
                            # 收集给后台引擎用的完整数据（无论有没有勾选）
                            detections.append({
                                'class': cls_name,
                                'bbox': draw_bbox,
                                'confidence': conf,
                                '_class_id': cls_id,
                            })
                    # 如果用了 OBB 旋转框模型
                    elif getattr(results[0], 'obb', None) is not None and len(results[0].obb) > 0:
                        for obb in results[0].obb:
                            cls_id = int(obb.cls[0])
                            x1, y1, x2, y2 = obb.xyxy[0].tolist()
                            points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                            draw_bbox = transform_bbox_to_display(
                                [x1, y1, x2, y2], self.frame_transform, native_w, native_h
                            )
                            draw_points = transform_points_to_display(
                                points, self.frame_transform, native_w, native_h
                            ).astype(int)
                            conf = float(obb.conf[0])
                            cls_name = names.get(cls_id, str(cls_id))
                            detections.append({
                                'class': cls_name,
                                'bbox': draw_bbox,
                                'points': draw_points.tolist(),
                                'confidence': conf,
                                '_class_id': cls_id,
                            })

                    detections = self._postprocess_model_detections(detections)
                    boxes_to_draw = []
                    for detection in detections:
                        cls_name = detection["class"]
                        if cls_name not in ui_checked_names or cls_name in ('glove', '手套'):
                            continue
                        cls_id = int(detection.get("_class_id", self.name_to_id_cache.get(cls_name, 0)))
                        conf = float(detection.get("confidence", 0.0))
                        if detection.get("points") is not None:
                            box_data = np.asarray(detection["points"], dtype=np.int32).reshape(-1, 2)
                            box_type = 'obb'
                        else:
                            box_data = detection["bbox"]
                            box_type = 'aabb'
                        boxes_to_draw.append((box_data, cls_name, conf, box_type, cls_id))

                    derived_detections = self.derived_obb_tracker.update(detections)
                    if derived_detections:
                        detections.extend(derived_detections)
                        if self.derived_obb_tracker.config.get("show_result", True):
                            for derived in derived_detections:
                                derived_points = np.asarray(
                                    derived.get("points", []), dtype=np.int32
                                ).reshape(-1, 2)
                                if len(derived_points) >= 4:
                                    box_data = derived_points
                                    box_type = 'obb'
                                else:
                                    box_data = derived["bbox"]
                                    box_type = 'aabb'
                                derived_name = derived["class"]
                                derived_color_id = (
                                    7 if derived.get("relation_state") == "matched" else 0
                                )
                                boxes_to_draw.append((
                                    box_data,
                                    derived_name,
                                    float(derived.get("confidence", 1.0)),
                                    box_type,
                                    derived_color_id,
                                ))

                    # 夺回 UI 控制权：彻底抛弃 YOLO 自带的 .plot()
                    annotated_frame = frame.copy()

                    if not self.use_chinese_labels:
                        # 【路线 A：英文原版】用 OpenCV 画干净的框
                        for box_data, eng_name, conf, box_type, cls_id in boxes_to_draw:
                            rgb_color = detection_class_color_rgb(cls_id, eng_name)
                            cv_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                            if box_type == 'aabb':
                                x1, y1, x2, y2 = map(int, box_data)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cv_color, 2)
                                text_x, text_y = x1, max(0, y1 - 10)
                            else:
                                # 🌟 画斜框 (多边形)
                                cv2.polylines(annotated_frame, [box_data], isClosed=True, color=cv_color,
                                              thickness=2)
                                # 找最上面的一个点作为文字的基准点
                                top_pt = min(box_data, key=lambda p: p[1])
                                text_x, text_y = top_pt[0], max(0, top_pt[1] - 10)

                            cv2.putText(annotated_frame, f"{eng_name} {conf:.2f}", (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cv_color, 2)
                    else:
                        # 【路线 B：中文优化版】OpenCV 画框，只用 Pillow 缓存小块中文字。
                        for box_data, eng_name, conf, box_type, cls_id in boxes_to_draw:
                            zh_name = self.engine.eng_to_zh.get(eng_name, eng_name)
                            display_text = f"{zh_name} {conf:.2f}"
                            rgb_color = detection_class_color_rgb(cls_id, eng_name)
                            cv_color = (rgb_color[2], rgb_color[1], rgb_color[0])

                            if box_type == 'aabb':
                                x1, y1, x2, y2 = map(int, box_data)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cv_color, 3)
                                text_x, box_top = x1, y1
                            else:
                                cv2.polylines(
                                    annotated_frame, [box_data], isClosed=True,
                                    color=cv_color, thickness=3,
                                )
                                top_pt = min(box_data, key=lambda point: point[1])
                                text_x, box_top = int(top_pt[0]), int(top_pt[1])

                            text_patch = self._get_text_patch(
                                display_text, rgb_color
                            )
                            text_y = max(0, box_top - text_patch[0].shape[0])
                            self._paste_text_patch(
                                annotated_frame, text_patch, text_x, text_y
                            )
            # 🌟 3. 画盲区锁定框 (LOCKED)
            for zone in self.engine.blind_zones:
                zx1, zy1, zx2, zy2 = map(int, zone)
                cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (255, 144, 30), 4)
                cv2.putText(annotated_frame, "LOCKED", (zx1, zy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 144, 30), 2)

            # --- 状态机与警告判断 ---
            # AOI 通过后保持提示信息 3 秒，不立即清空
            if self.aoi_pass_msg_frames > 0 and self.aoi_state is None:
                self.aoi_pass_msg_frames -= 1
                alert_msg = f"✅ AOI 特征比对通过!  (相似度: {self.aoi_similarity:.2%})"
            else:
                alert_msg = ""
            progress = 0

            # --- AOI 强制放行信号处理 ---
            if self.aoi_force_signal:
                if self.aoi_state == 'blocked':
                    old_idx = self.aoi_step_idx if self.aoi_step_idx is not None else self.current_step_idx
                    was_remediation_aoi = self.aoi_context == "remediation"
                    if was_remediation_aoi:
                        self.ng_tracker.mark_step_remedied(old_idx)
                    else:
                        self.ng_tracker.mark_step_completed(old_idx)
                    cp = self.ng_tracker.current_product
                    if cp and old_idx < len(cp['step_records']):
                        cp['step_records'][old_idx]['aoi_forced'] = True
                        cp['step_records'][old_idx]['aoi_note'] = f'AOI特征比对失败_人工强制放行 (相似度: {self.aoi_similarity:.3f})'
                        cp['step_records'][old_idx]['aoi_similarity'] = float(self.aoi_similarity)
                        cp['step_records'][old_idx]['aoi_threshold'] = float(self.aoi_threshold)
                        cp['step_records'][old_idx]['aoi_state'] = 'forced'
                        cp['ng_reason'] = f'工序{old_idx+1} AOI特征比对未通过(人工放行)'
                    if was_remediation_aoi:
                        self._exit_remediation_mode(f"补救步骤 {old_idx + 1} 已人工放行，已回到正常工序。")
                    else:
                        self._advance_completed_step_or_start_post_check(old_idx, detections)
                        self.current_sub_count = 0
                        self.aoi_state = None
                        self.aoi_step_idx = None
                        self.aoi_context = "normal"
                        self._alarm_aoi_blocked_active = False
                        self.aoi_stable_count = 0
                        self.is_pausing = False
                        self.engine.reset()
                        self._reset_jump_monitors()
                        self._clear_restart_guard()
                        self.step_start_time = time.time()
                    alert_msg = "⚠️ AOI 特征比对人工强制放行！"
                    self.aoi_update_signal.emit(0.0, '', False)
                    # 最后一步放行后自动进入下一轮
                    if (not was_remediation_aoi
                            and not self.post_completion_check
                            and self.current_step_idx >= len(self.process_steps)):
                        self._finish_and_restart_cycle()
                        alert_msg = "⚠️ AOI 强制放行，当前产品已完成，进入下一轮！"
                self.aoi_force_signal = False

            # 0. 手动补救状态：只有用户点击“补救”后，才允许把跳过步骤改为已补救。
            if self.remediation_cancel_signal:
                self._exit_remediation_mode("已手动退出补救状态，回到正常工序。")
                self.remediation_cancel_signal = False
            if self.remediation_request_idx is not None:
                requested_idx = self.remediation_request_idx
                self.remediation_request_idx = None
                if self._enter_remediation_mode(requested_idx):
                    alert_msg = self.remediation_status_msg
                else:
                    alert_msg = self.remediation_status_msg or "无法进入补救状态。"

            # 1. 全局持续报警检查：违禁物、状态规则或槽位终态。
            continuous_result = self._update_continuous_monitors(detections)
            conditional_alarms = list(continuous_result.get("alarms", []))
            conditional_alarm = conditional_alarms[0] if conditional_alarms else None
            slot_alarm_msg = self._observed_slot_mismatch()
            active_slot_step = self._active_slot_expectation_step()
            slot_alarm_blocks = bool(
                slot_alarm_msg
                and (not active_slot_step
                     or active_slot_step.get("slot_expectation_block", True))
            )
            has_forbidden, fb_class = self.engine.check_forbidden(detections, self.forbidden_targets)
            has_state_alarm, state_alarm_msg, state_alarm_idx = self._check_state_alarm(detections)
            has_any_alarm = bool(
                has_forbidden or has_state_alarm or conditional_alarm or slot_alarm_msg
            )
            has_blocking_alarm = bool(
                has_forbidden or has_state_alarm or conditional_alarm or slot_alarm_blocks
            )
            if has_any_alarm:
                alarm_buzzer = bool(
                    has_forbidden
                    or (has_state_alarm and self._state_alarm_uses_buzzer(state_alarm_idx))
                    or (conditional_alarm and conditional_alarm.get("buzzer", True))
                    or slot_alarm_msg
                )
                self._set_forbidden_alarm(True, buzzer=alarm_buzzer)
                if has_forbidden:
                    zh_name = self.engine.eng_to_zh.get(fb_class, fb_class)
                    alert_msg = f"画面中检测到违禁项：【{zh_name}】！"
                elif conditional_alarm:
                    alert_msg = conditional_alarm.get("message", "状态条件报警！")
                elif slot_alarm_msg:
                    alert_msg = slot_alarm_msg
                else:
                    alert_msg = state_alarm_msg
            else:
                self._set_forbidden_alarm(False)
                # 2. 超时监控 (🌟 动态读取 step_timeout)
                if (not self.remediation_mode and self.process_steps
                        and self.current_step_idx < len(self.process_steps) and not self.is_pausing):
                    if time.time() - self.step_start_time > self.step_timeout:
                        alert_msg = f"⏱️ 当前步骤耗时过长 (超{self.step_timeout}秒)，请检查或强制跳过！"
                        self.ng_tracker.mark_step_timeout(self.current_step_idx, self.step_timeout)

            # 3. 工序流转与循环
            force_end_product = False
            # 最后一步跳步/收卷统一交给 WorkflowMonitor，避免专属末步引擎和主状态机抢状态。

            # ==============================================================
            # 3. 工序流转与循环
            # ==============================================================
            if not has_blocking_alarm:
                if self.remediation_mode and self.aoi_state is None:
                    rem_idx = self.remediation_step_idx
                    if rem_idx is None or rem_idx >= len(self.process_steps):
                        self._exit_remediation_mode("补救步骤不存在，已退出补救状态。")
                        alert_msg = self.remediation_status_msg
                    else:
                        rem_step = self.process_steps[rem_idx]
                        rem_targets = self._targets_for_step(rem_step)
                        if rem_idx not in self.remediation_engines:
                            self.remediation_engines[rem_idx] = ProcessLogicEngine()
                            self._clone_parser_to_engine(self.remediation_engines[rem_idx])
                        is_remedied, rem_progress = self._evaluate_step_by_config(
                            rem_idx, rem_step, rem_targets, detections, self.remediation_engines[rem_idx]
                        )
                        self.step_progress_by_idx[rem_idx] = rem_progress
                        progress = rem_progress
                        alert_msg = f"🛠️ 补救中：请完成步骤 {rem_idx + 1} [{rem_progress}%]，完成前不会回到正常工序。"
                        if is_remedied:
                            self.step_progress_by_idx[rem_idx] = 100
                            aoi_cfg = rem_step.get('aoi_feature_check', {})
                            if aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
                                self._start_aoi_check(rem_idx, aoi_cfg, context="remediation")
                                alert_msg = f"🛠️ 步骤 {rem_idx + 1} 动作已完成，正在进行 AOI 补救比对。"
                            else:
                                self._finish_remediation_step(rem_idx)
                                alert_msg = self.remediation_status_msg
                elif force_end_product:
                    # 💥 触发强行交卷！
                    self._flash_red_alarm("force_end")
                    self._advance_after_jump(
                        len(self.process_steps) - 1,
                        "强制收卷：检测到工人直接完成了最后一步"
                    )
                    alert_msg = "⚠️ 强行结算：检测到最后一步已完成，当前产品作废 (NG)，进入下个循环！"
                elif self.aoi_state is not None:
                    # ── AOI 特征比对状态机 ──
                    progress = 0
                    active_aoi_idx = self.aoi_step_idx if self.aoi_step_idx is not None else self.current_step_idx
                    if 0 <= active_aoi_idx < len(self.process_steps):
                        anchor_class = self.aoi_anchor_class
                        threshold = self.aoi_threshold
                        timeout = self.aoi_timeout
                        standard_vec = self.aoi_standard_vector
                        aoi_finished_this_frame = False

                        # 寻找锚定物
                        anchor_det = None
                        for d in detections:
                            if d['class'] == anchor_class:
                                anchor_det = d
                                break

                        if anchor_det is not None and self.aoi_extractor is not None:
                            # 多角度比对：容忍工件 ±10° 旋转偏差（blocked 状态下也继续比对）
                            sim, best_angle = self.aoi_extractor.compare_multi_angle(
                                raw_frame, anchor_det['bbox'], standard_vec
                            )
                            self.aoi_similarity = sim
                            self._record_aoi_status(active_aoi_idx, self.aoi_state or 'checking', sim, threshold, best_angle)

                            # 核心：无论 blocked 还是 checking，只要相似度达标就累加，不达标就扣
                            if sim >= threshold:
                                self.aoi_stable_count += 1
                                if self.aoi_stable_count >= 3:
                                    # 放行！包括从 blocked 恢复的情况
                                    was_blocked = (self.aoi_state == 'blocked')
                                    passed_idx = active_aoi_idx
                                    was_remediation_aoi = self.aoi_context == "remediation"
                                    self._record_aoi_status(passed_idx, 'passed', sim, threshold, best_angle)
                                    self.step_progress_by_idx[passed_idx] = 100
                                    self._clear_step_runtime_flags(passed_idx)
                                    if was_remediation_aoi:
                                        self.ng_tracker.mark_step_remedied(passed_idx)
                                        finished_group = False
                                    else:
                                        self.ng_tracker.mark_step_completed(passed_idx)
                                        finished_group = self._advance_completed_step_or_start_post_check(
                                            passed_idx, detections
                                        )
                                    self.current_sub_count = 0
                                    self.aoi_state = None
                                    self.aoi_step_idx = None
                                    self.aoi_context = "normal"
                                    self._alarm_aoi_blocked_active = False
                                    self.aoi_stable_count = 0
                                    self.aoi_pass_flash = 30   # 绿色边框闪烁 1 秒
                                    self.aoi_pass_msg_frames = 90  # 消息保持 3 秒
                                    self.is_pausing = False
                                    self.engine.reset()
                                    self._reset_jump_monitors()
                                    self._clear_restart_guard()
                                    self.step_start_time = time.time()
                                    self.aoi_update_signal.emit(sim, '', False)
                                    if was_remediation_aoi:
                                        self.ng_tracker._check_and_restore_ok()
                                        self._exit_remediation_mode(
                                            f"补救完成：步骤 {passed_idx + 1} 已通过 AOI，比对完成后回到正常工序。"
                                        )
                                        alert_msg = self.remediation_status_msg
                                    elif was_blocked:
                                        alert_msg = f"✅ AOI 特征比对恢复通过! (相似度: {sim:.2%}，阻塞已解除)"
                                        # 更新步骤记录：清除阻塞标记，记为恢复通过
                                        cp = self.ng_tracker.current_product
                                        if cp and passed_idx < len(cp['step_records']):
                                            rec = cp['step_records'][passed_idx]
                                            rec['aoi_recovered'] = True
                                            rec['was_aoi_blocked'] = True
                                            rec['aoi_blocked'] = False
                                            rec['aoi_similarity'] = float(sim)
                                            rec['aoi_threshold'] = float(threshold)
                                            rec['aoi_state'] = 'passed'
                                        # 尝试恢复产品状态为 OK（如果所有步骤都已完成）
                                        self.ng_tracker._check_and_restore_ok()
                                    else:
                                        self.ng_tracker._check_and_restore_ok()
                                        alert_msg = f"✅ AOI 特征比对通过! (相似度: {sim:.2%})"
                                    # 最后一步完成 → 延迟到闪烁结束后再重启，让用户看到通过提示
                                    if (not was_remediation_aoi
                                            and not self.post_completion_check
                                            and self.current_step_idx >= len(self.process_steps)):
                                        self.aoi_pending_restart = True
                                        alert_msg = f"✅ AOI 特征比对通过! 所有步骤已完成，即将进入下一轮！"
                                    elif not was_remediation_aoi and finished_group:
                                        alert_msg = f"✅ AOI 特征比对通过! 乱序组已完成，即将进入下一轮！"
                                    aoi_finished_this_frame = True
                            else:
                                self.aoi_stable_count = max(0, self.aoi_stable_count - 1)
                                # blocked 状态下相似度又掉下去了，复位计数器
                                if self.aoi_state == 'blocked' and self.aoi_stable_count == 0:
                                    pass  # 继续 blocked，等待恢复
                            # 更新当前状态（blocked 不会被 anchor detection 改变）
                            if not aoi_finished_this_frame and self.aoi_state != 'blocked':
                                self.aoi_state = 'checking'
                                self._alarm_aoi_blocked_active = False
                            if not aoi_finished_this_frame:
                                self._record_aoi_status(active_aoi_idx, self.aoi_state, self.aoi_similarity, threshold)
                        elif not aoi_finished_this_frame:
                            # 锚定物丢失
                            if self.aoi_state != 'blocked':
                                if self.aoi_state == 'checking':
                                    self.aoi_stable_count = max(0, self.aoi_stable_count - 1)
                                self.aoi_state = 'finding_anchor'
                            self._record_aoi_status(active_aoi_idx, self.aoi_state, self.aoi_similarity, threshold)

                        # 超时阻塞（只在未 blocked 时触发一次）
                        elapsed = time.time() - self.aoi_check_start_time
                        if not aoi_finished_this_frame and elapsed > timeout and self.aoi_state != 'blocked':
                            self.aoi_state = 'blocked'
                            if not self._alarm_aoi_blocked_active:
                                self._flash_red_alarm(f"aoi_blocked_{active_aoi_idx}")
                                self._alarm_aoi_blocked_active = True
                            self.aoi_stable_count = 0  # 进入阻塞时重置计数器，等待恢复
                            # 记录 AOI 阻塞原因到 NG 追踪器
                            cp = self.ng_tracker.current_product
                            if cp:
                                cp['ng_reason'] = f'工序{active_aoi_idx+1} AOI特征比对未通过(相似度{self.aoi_similarity:.2%}<阈值{threshold:.0%})'
                                cp['status'] = 'NG'
                                step_rec = cp['step_records'][active_aoi_idx] if active_aoi_idx < len(cp['step_records']) else None
                                if step_rec:
                                    step_rec['aoi_blocked'] = True
                                    step_rec['aoi_similarity'] = float(self.aoi_similarity)
                                    step_rec['aoi_threshold'] = float(threshold)
                                    step_rec['aoi_state'] = 'blocked'

                        # 状态文案
                        if aoi_finished_this_frame:
                            pass
                        elif self.aoi_state == 'finding_anchor':
                            zh_name = self.engine.eng_to_zh.get(anchor_class, anchor_class)
                            alert_msg = f"🔍 AOI特征比对: 正在寻找锚定物 [{zh_name}]... ({elapsed:.1f}s/{timeout:.0f}s)"
                        elif self.aoi_state == 'checking':
                            alert_msg = f"🔬 AOI特征比对: 相似度={self.aoi_similarity:.2%} 阈值={threshold:.0%} 稳定帧={self.aoi_stable_count}/3"
                        elif self.aoi_state == 'blocked':
                            if self.aoi_similarity >= threshold:
                                alert_msg = f"🔄 AOI 阻塞恢复中: 相似度={self.aoi_similarity:.2%} >= 阈值{threshold:.0%} 稳定帧={self.aoi_stable_count}/3"
                            else:
                                alert_msg = f"🚫 AOI 特征比对失败! 当前相似度 {self.aoi_similarity:.2%} < 阈值 {threshold:.0%}，疑似来料异常!"

                        # AOI 信号给 UI
                        if not aoi_finished_this_frame:
                            self.aoi_update_signal.emit(self.aoi_similarity, self.aoi_state,
                                                        self.aoi_state == 'blocked')

                        # 画 AOI 锚定框
                        if not aoi_finished_this_frame and anchor_det:
                            x1, y1, x2, y2 = map(int, anchor_det['bbox'])
                            if self.aoi_state == 'checking' and self.aoi_similarity >= threshold:
                                color = (0, 255, 0)
                            elif self.aoi_state == 'blocked':
                                color = (0, 0, 255)
                            else:
                                color = (0, 165, 255)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(annotated_frame, f"AOI sim={self.aoi_similarity:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # 阻塞状态红框闪烁
                        if not aoi_finished_this_frame and self.aoi_state == 'blocked':
                            img_h, img_w = annotated_frame.shape[:2]
                            if int(time.time() * 3) % 2 == 0:
                                cv2.rectangle(annotated_frame, (10, 10), (img_w - 10, img_h - 10), (0, 0, 255), 20)
                                cv2.putText(annotated_frame, "AOI BLOCKED - CHECK PART!", (img_w // 8, img_h // 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5, cv2.LINE_AA)

                else:
                    # 👇 原来正常的流转逻辑，整体往右缩进一格，放在 else 里面
                    if self.process_steps and self.current_step_idx < len(self.process_steps):
                        handled_post_check = False
                        if self.post_completion_check:
                            handled_post_check = True
                            _, post_check_msg = self._evaluate_post_completion_check(detections)
                            progress = 100
                            if post_check_msg and not alert_msg:
                                alert_msg = post_check_msg
                        group_indices = (
                            [] if handled_post_check
                            else self._unordered_group_indices(self.current_step_idx)
                        )
                        if group_indices:
                            finished_group, group_progress, group_alert = self._evaluate_unordered_group(detections)
                            progress = group_progress
                            if group_alert and not alert_msg:
                                alert_msg = group_alert
                            if finished_group:
                                self.current_sub_count = 0
                                self.engine.reset()
                                self.step_start_time = time.time()
                        elif not handled_post_check:
                            current_step_dict = self.process_steps[self.current_step_idx]
                            req_count = current_step_dict.get("count", 1)

                            # 1. 核心修复：重新调用引擎，计算当前的进度！
                            is_step_done, step_progress = self._evaluate_step_by_config(
                                self.current_step_idx, current_step_dict, required_targets, detections, self.engine
                            )
                            progress = step_progress  # 把算出来的进度赋值给 UI 变量
                            self.step_progress_by_idx[self.current_step_idx] = step_progress

                            # 2. 状态机：如果这一步做完了，开启短暂的“确认暂停”状态
                            if is_step_done and not self.is_pausing:
                                self.is_pausing = True
                                self.pause_start_time = time.time()

                            # 3. 状态机：按本步骤配置的确认时长停留后，正式进入下一步
                            if self.is_pausing:
                                completion_hold_seconds = self._step_completion_hold_seconds(
                                    current_step_dict
                                )
                                if time.time() - self.pause_start_time >= completion_hold_seconds:
                                    self.is_pausing = False
                                    self.current_sub_count += 1
                                    self.engine.reset()  # 清空动作引擎记忆
                                    self.step_post_action_latched.discard(self.current_step_idx)
                                    self.step_result_hit_counts.pop(self.current_step_idx, None)
                                    self._reset_jump_monitors()

                                    # 检查子次数是否全部达标
                                    if self.current_sub_count >= req_count:
                                        step_dict = self.process_steps[self.current_step_idx]
                                        self._clear_step_runtime_flags(self.current_step_idx)
                                        aoi_cfg = step_dict.get('aoi_feature_check', {})
                                        if aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
                                            # 进入 AOI 特征比对状态，不立即推进步骤
                                            self._start_aoi_check(self.current_step_idx, aoi_cfg)
                                        else:
                                            completed_idx = self.current_step_idx
                                            self.ng_tracker.on_step_advance(completed_idx, 'completed')
                                            self.step_progress_by_idx[completed_idx] = 100
                                            self._clear_step_runtime_flags(completed_idx)
                                            self._clear_restart_guard()
                                            product_complete = (
                                                self._advance_completed_step_or_start_post_check(
                                                    completed_idx, detections
                                                )
                                            )
                                            # 最后一步完成后自动进入下一轮
                                            if product_complete and not self.post_completion_check:
                                                self._finish_and_restart_cycle()
                                        self.current_sub_count = 0
                                    else:
                                        self._set_step_cooldown(self.current_step_idx, current_step_dict)

                                    self.step_start_time = time.time()

            # AOI 通过后的绿色闪烁提示（持续 ~3 秒，独立于 AOI 状态机）
            if self.aoi_pass_flash > 0:
                self.aoi_pass_flash -= 1
                img_h, img_w = annotated_frame.shape[:2]
                if int(time.time() * 4) % 2 == 0:
                    cv2.rectangle(annotated_frame, (10, 10), (img_w - 10, img_h - 10), (0, 255, 0), 20)
                    cv2.putText(annotated_frame, "AOI PASS!", (img_w // 4, img_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5, cv2.LINE_AA)
            # 消息保持结束后执行延迟的循环重启。这里必须放在 aoi_pass_flash 外面：
            # 否则闪烁帧数先归零后，pending restart 会永远没有机会触发。
            if self.aoi_pending_restart and self.aoi_pass_msg_frames <= 0 and self.aoi_state is None:
                self.aoi_pending_restart = False
                self.aoi_pass_flash = 0
                self._finish_and_restart_cycle()

            # ==============================================================
            # 🌟🌟🌟 新增：意图感知、跳步报警与 UI 渲染 🌟🌟🌟
            # ==============================================================
            current_step_text = ""
            req_count = 1
            if self.process_steps and self.current_step_idx < len(self.process_steps):
                step_info = self.process_steps[self.current_step_idx]
                req_count = step_info.get("count", 1)

                # 🌟 你要的显示第几次：如果步骤配置了多次，自动拼接在文本后面
                sub_count_str = f"({self.current_sub_count + 1}/{req_count})" if req_count > 1 else ""
                current_step_text = step_info.get("text", "") + sub_count_str

            # 1. 交给引擎去算手里拿了啥 (支持画面里出现多只手/手套同时判定！)
            # ⚠️ 注意这里删掉了 frame 参数，传入了 self.is_pausing 状态
            hand_results, held_objs, hand_renders = self.intent_engine.process_intent(
                frame,detections, required_targets, progress, self.is_pausing, current_step_text,
                self.engine.eng_to_zh
            )
            # 2. 组终态、错误装配、接近预警与正式跳步分开计算。
            # 组终态/错误装配/预警只提示，不推进工序、不记 NG。
            new_group_final_alarm, group_final_msg, group_final_key = (
                self._check_pending_group_final_checks(detections)
            )
            if new_group_final_alarm:
                self._flash_red_alarm(group_final_key)
            if self.just_restarted_cycle or self.remediation_mode or self.aoi_state is not None:
                self.workflow_monitor.clear_prewarning_runtime()
                self.wrong_pair_counters.clear()
                is_wrong_pair, wrong_pair_msg, wrong_pair_step_idx = False, "", -1
                is_prewarning, prewarning_msg, prewarning_step_idx = False, "", -1
                is_jump, jump_msg, jumped_to_idx = False, "", -1
            else:
                is_wrong_pair, wrong_pair_msg, wrong_pair_step_idx = self._check_wrong_pair_alarm(detections)
                is_prewarning, prewarning_msg, prewarning_step_idx = self.workflow_monitor.check_prewarning(
                    detections, self.process_steps, self.current_step_idx,
                    completed_step_indices=self._completed_step_indices()
                )
                is_jump, jump_msg, jumped_to_idx = self.workflow_monitor.check_jump_by_completion(
                    detections, self.process_steps, self.current_step_idx, held_objs,
                    completed_step_indices=self._completed_step_indices()
                )

            jump_suspicions = []
            if self.show_jump_progress and not is_jump:
                for suspected_idx, suspected_progress in sorted(
                    self.workflow_monitor.shadow_progress_by_idx.items()
                ):
                    if (
                        suspected_idx != self.current_step_idx
                        and 0 <= suspected_idx < len(self.process_steps)
                        and self._step_record_status(suspected_idx) == "pending"
                        and int(suspected_progress or 0) > 0
                    ):
                        jump_suspicions.append(
                            (suspected_idx, int(suspected_progress))
                        )

            if is_wrong_pair:
                self._flash_red_alarm(f"wrong_pair_{wrong_pair_step_idx}")
                alert_msg = wrong_pair_msg
                img_h, img_w = annotated_frame.shape[:2]
                border_color = (0, 0, 255) if fps_frame_count % 4 < 2 else (0, 128, 255)
                cv2.rectangle(annotated_frame, (15, 15), (img_w - 15, img_h - 15), border_color, 25)
                cv2.putText(annotated_frame, "WARNING: WRONG ASSEMBLY!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 6, cv2.LINE_AA)
            elif is_jump:
                self._flash_red_alarm(f"jump_{jumped_to_idx if jumped_to_idx >= 0 else self.current_step_idx}")
                alert_msg = jump_msg
                if (jumped_to_idx >= 0
                        and not self._prerequisite_alarm_blocks_advancement(jumped_to_idx)):
                    self._advance_after_jump(jumped_to_idx, jump_msg)
                img_h, img_w = annotated_frame.shape[:2]

                # 1. 边框闪烁效果：只画一圈粗边框，绝不全屏覆盖遮挡视线
                border_color = (0, 0, 255) if fps_frame_count % 4 < 2 else (0, 128, 255)  # 红/橙交替闪烁
                cv2.rectangle(annotated_frame, (15, 15), (img_w - 15, img_h - 15), border_color, 25)

                # 2. 告警文字：因为 cv2.putText 不支持中文会变 ???，工业界一般直接用醒目的英文或拼音
                cv2.putText(annotated_frame, "WARNING: STEP JUMP!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6, cv2.LINE_AA)
            elif is_prewarning:
                self._flash_red_alarm(f"prewarning_{prewarning_step_idx}")
                alert_msg = prewarning_msg
                img_h, img_w = annotated_frame.shape[:2]
                border_color = (0, 165, 255) if fps_frame_count % 4 < 2 else (0, 215, 255)
                cv2.rectangle(annotated_frame, (15, 15), (img_w - 15, img_h - 15), border_color, 20)
                cv2.putText(annotated_frame, "CAUTION: EARLY ACTION!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, border_color, 5, cv2.LINE_AA)
            elif group_final_msg:
                alert_msg = group_final_msg
                img_h, img_w = annotated_frame.shape[:2]
                border_color = (0, 0, 255) if fps_frame_count % 4 < 2 else (0, 128, 255)
                cv2.rectangle(annotated_frame, (15, 15), (img_w - 15, img_h - 15), border_color, 20)
                cv2.putText(annotated_frame, "WARNING: GROUP FINAL STATE!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, border_color, 5, cv2.LINE_AA)
            elif jump_suspicions and not alert_msg:
                suspicion_text = "、".join(
                    f"步骤 {idx + 1}「{self.process_steps[idx].get('text', '')}」"
                    f"[{suspected_progress}%]"
                    for idx, suspected_progress in jump_suspicions[:3]
                )
                if len(jump_suspicions) > 3:
                    suspicion_text += f"等 {len(jump_suspicions)} 步"
                alert_msg = (
                    f"🔎 疑似正在执行：{suspicion_text}，请确认是否发生跳步"
                )

            # 🌟🌟🌟 修复：将缩进向左退一格，使其与 if is_jump 平级！
            # 3. 循环画出所有的手/手套和悬浮文字
            if hand_renders:
                for render_box, hand_color, hand_text, box_type, landmarks, hand_id in hand_renders:
                    cv_color = (hand_color[2], hand_color[1], hand_color[0])
                    if box_type == 'aabb':
                        hx1, hy1, hx2, hy2 = map(int, render_box)
                        cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), cv_color, 4)
                        text_x, box_top = hx1, hy1
                    else:
                        cv2.polylines(
                            annotated_frame, [render_box], isClosed=True,
                            color=cv_color, thickness=4,
                        )
                        top_pt = min(render_box, key=lambda point: point[1])
                        text_x, box_top = int(top_pt[0]), int(top_pt[1])

                    text_patch = self._get_text_patch(
                        hand_text, hand_color, font=self.font_hand
                    )
                    text_y = max(0, box_top - text_patch[0].shape[0])
                    self._paste_text_patch(
                        annotated_frame, text_patch, text_x, text_y
                    )

                # 🌟 重新遍历一次，专门画骨骼
                for _, _, _, _, landmarks, _ in hand_renders:
                    if landmarks is not None:
                        mp_drawing.draw_landmarks(
                            annotated_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
            # ==============================================================
            # 🌟🌟🌟 新增代码结束 🌟🌟🌟
            # 🌟 在左上角绘制当前帧率
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 50 if not is_jump else 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            frame_h, frame_w = raw_frame.shape[:2]
            cv2.putText(annotated_frame, f"RES: {frame_w}x{frame_h}", (20, 100 if not is_jump else 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            if self.yolo_input_size:
                yolo_input_w, yolo_input_h = self.yolo_input_size
                cv2.putText(
                    annotated_frame,
                    f"YOLO IN: {yolo_input_w}x{yolo_input_h}",
                    (20, 140 if not is_jump else 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
            # ==============================================================
            # 🌟🌟🌟 新增：2K 拍照与录像模块 (在缩放前处理，保证原画质！) 🌟🌟🌟
            # ==============================================================
            os.makedirs("video-photo", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 拍照逻辑
            if self.req_take_photo:
                photo_path = f"video-photo/photo_{timestamp}.jpg"
                # 🌟 关键修改 2：保存干净的原图
                cv2.imwrite(photo_path, raw_frame)
                self.req_take_photo = False

            # AOI 抓拍标准样件
            if self.req_aoi_capture and self._aoi_capture_anchor:
                self._aoi_capture_ttl -= 1
                anchor_class = self._aoi_capture_anchor
                anchor_det = next((d for d in detections if d['class'] == anchor_class), None)
                if anchor_det is not None:
                    # 懒初始化：如果还没加载过 AOI 提取器，现场创建
                    if self.aoi_extractor is None:
                        try:
                            from aoi_extractor import AOIFeatureExtractor
                            device = get_safe_torch_device()
                            self.aoi_extractor = AOIFeatureExtractor(backbone='resnet18', device=device)
                        except Exception as e:
                            print(f"[AOI] 特征提取器初始化失败: {e}")
                    if self.aoi_extractor is not None:
                        feat = self.aoi_extractor.extract(raw_frame, anchor_det['bbox'])
                        crop = self.aoi_extractor._crop(raw_frame, anchor_det['bbox'])
                        if crop is None:
                            crop = raw_frame.copy()
                        else:
                            crop = crop.copy()
                        frame_h, frame_w = raw_frame.shape[:2]
                        crop_h, crop_w = crop.shape[:2]
                        self.aoi_capture_done_signal.emit(feat, crop, (frame_w, frame_h, crop_w, crop_h))
                        self._aoi_capture_ttl = 0
                    self.req_aoi_capture = False
                    self._aoi_capture_anchor = None
                elif self._aoi_capture_ttl <= 0:
                # 超时：锚定物在 ~3 秒内未检测到，取消抓拍
                    zh_name = self.engine.eng_to_zh.get(anchor_class, anchor_class)

                    # 💡 核心修复：不但要在 banner 报警，还要弹出一个极其明显的阻断式警告
                    alert_msg = f"⚠️ AOI 抓拍失败：未在画面中找到【{zh_name}】！"
                    self.aoi_capture_failed_signal.emit(alert_msg)

                    # 借用 update_ui_signal 把这个严重的错误发给前端弹窗
                    # (由于这里是子线程，千万不要直接在这里写 QMessageBox)
                    self.req_aoi_capture = False
                    self._aoi_capture_anchor = None

            # 报警追溯录像独立于手动录像：视频内叠加工序/进度/产品状态，
            # 同目录 JSON 保存可检索的上下文时间线。
            try:
                alarm_context = self._build_alarm_record_context(alert_msg, progress)
                self._update_alarm_clip(
                    time.time(), annotated_frame, alarm_context, alert_msg, current_fps
                )
            except Exception as exc:
                print(f"[AlarmRecording] 帧处理失败: {exc}")
                self._finish_alarm_clip("recording_error")

            # 2. 录像状态机控制
            if self.req_record_action == 'start':
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0 if current_fps <= 0 else current_fps
                self.video_writer = cv2.VideoWriter(
                    f"video-photo/video_{timestamp}.mp4", fourcc, fps,
                    (self.record_width, self.record_height))

                if self.video_writer.isOpened():
                    self.is_recording = True
                    self.is_record_paused = False
                    self.record_start_time = time.time()
                    self.total_paused_time = 0
                else:
                    self.video_writer = None
                    self.is_recording = False
                    self.is_record_paused = False
                    alert_msg = "❌ 录像启动失败：无法创建视频文件"
                self.req_record_action = None

            elif self.req_record_action == 'pause':
                if self.is_recording and not self.is_record_paused:
                    self.is_record_paused = True
                    self.pause_start_tick = time.time()
                self.req_record_action = None

            elif self.req_record_action == 'resume':
                if self.is_recording and self.is_record_paused:
                    self.is_record_paused = False
                    self.total_paused_time += (time.time() - self.pause_start_tick)
                self.req_record_action = None

            elif self.req_record_action == 'stop':
                self._finish_recording()
                self.req_record_action = None

            # 3. 写入视频帧与更新时长
            if self.is_recording:
                if not self.is_record_paused:
                    if self.video_writer:
                        # 按录制分辨率缩放后写入
                        rec_frame = cv2.resize(raw_frame, (self.record_width, self.record_height))
                        self.video_writer.write(rec_frame)

                    active_time = time.time() - self.record_start_time - self.total_paused_time
                    mins, secs = divmod(int(active_time), 60)
                    self.recording_time_signal.emit(f"{mins:02d}:{secs:02d}")
            # ==============================================================

            # ==============================================================

            # 🌟 性能救星 (修复版)：等比例缩小画面发给 UI！
            raw_h, raw_w = annotated_frame.shape[:2]
            # 🌟 性能救星：发给 UI 前，强制把预览画面缩小！
            # 虽然你后台是用 4k 或 1080p 检测的，但 UI 显示根本不需要这么大
            # 把它缩到 800x600 左右，UI 线程的压力会瞬间骤降，帧率直接起飞
            # 🌟 性能救星 (修复版)：等比例缩小画面发给 UI！
            # 动态计算原始画面的宽高比，绝不拉伸变形
            raw_h, raw_w = annotated_frame.shape[:2]
            max_display_width = 1080  # 设定 UI 显示的最大安全宽度

            if raw_w > max_display_width:
                scale_ratio = max_display_width / raw_w
                new_w = max_display_width
                new_h = int(raw_h * scale_ratio)
                display_frame = cv2.resize(annotated_frame, (new_w, new_h))
            else:
                # 如果原视频本来就不大，直接原图送过去
                display_frame = annotated_frame.copy()

            # 🌟 注意：这里把 annotated_frame 换成 display_frame 发送
            self.update_ui_signal.emit(
                display_frame, self._display_process_steps(), self.current_step_idx,
                self.is_pausing, progress, alert_msg,
                self.ng_tracker.ok_count, self.current_sub_count  # 👈 修改这里：用 ok_count 代替原来的 completed_cycles
            )
        # 释放资源
        self._safe_stop_pipeline(pipeline)
        self._safe_release_capture(cap)
        self.running = False
        self._finish_recording()
        self._finish_alarm_clip("stream_stopped")
        self._alarm_frame_buffer.clear()
        self._alarm_last_buffered_ts = 0.0
        self._reset_transient_requests()
        self.aoi_update_signal.emit(0.0, '', False)
    def stop(self):
        self.running = False
        if self.isRunning() and QThread.currentThread() != self:
            self.wait()
        self._finish_recording()
        self._reset_transient_requests()


class RecordingDialog(QDialog):
    """录制与拍照子窗口 —— 独立管理录制状态，关闭时安全释放资源"""

    def __init__(self, vision_thread, parent=None):
        super().__init__(parent)
        self.vision_thread = vision_thread
        self.setWindowTitle("📸 录制与拍照")
        self.setMinimumWidth(380)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._user_stopped = False  # 用户主动停止录制，关闭时不需要再问
        self.setup_ui()
        self.vision_thread.recording_time_signal.connect(self._on_rec_time)
        self.vision_thread.alarm_clip_saved_signal.connect(self._on_alarm_clip_saved)

    def setup_ui(self):
        layout = QVBoxLayout()

        # 录制分辨率
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("录制分辨率:"))
        self.combo_record_res = QComboBox()
        self.combo_record_res.addItems(["1080p (1920×1080)", "2K (2560×1440)", "4K (3840×2160)"])
        self.combo_record_res.setCurrentIndex(0)
        res_row.addWidget(self.combo_record_res)
        layout.addLayout(res_row)

        # 抓拍按钮
        self.btn_take_photo = QPushButton("📷 抓拍当前画面")
        self.btn_take_photo.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 6px;")
        self.btn_take_photo.clicked.connect(self._take_photo)
        layout.addWidget(self.btn_take_photo)

        # 录制控制按钮
        rec_btn_row = QHBoxLayout()
        self.btn_record_start = QPushButton("⏺️ 开始录像")
        self.btn_record_pause = QPushButton("⏸️ 暂停")
        self.btn_record_stop = QPushButton("⏹️ 结束")

        self.btn_record_start.setStyleSheet("color: #d93025; font-weight: bold; padding: 5px;")
        self.btn_record_pause.setEnabled(False)
        self.btn_record_stop.setEnabled(False)

        self.btn_record_start.clicked.connect(self._record_start)
        self.btn_record_pause.clicked.connect(self._record_pause_resume)
        self.btn_record_stop.clicked.connect(self._record_stop)

        rec_btn_row.addWidget(self.btn_record_start)
        rec_btn_row.addWidget(self.btn_record_pause)
        rec_btn_row.addWidget(self.btn_record_stop)
        layout.addLayout(rec_btn_row)

        # 录制时长
        self.lbl_record_time = QLabel("⏱️ 录制时长: 00:00")
        self.lbl_record_time.setStyleSheet("color: #d93025; font-weight: bold; font-size: 16px;")
        self.lbl_record_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_record_time)

        # 提示文字
        hint = QLabel("提示：录制使用当前主画面摄像头的视频流")
        hint.setStyleSheet("color: #6c757d; font-size: 12px;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        alarm_group = QGroupBox("🚨 报警追溯录像")
        alarm_layout = QVBoxLayout(alarm_group)
        self.chk_alarm_clip = QCheckBox(
            "报警时自动保存前 5 秒 + 后 5 秒（含工序信息）"
        )
        self.chk_alarm_clip.setChecked(bool(self.vision_thread.alarm_clip_enabled))
        self.chk_alarm_clip.stateChanged.connect(self._toggle_alarm_clip)
        alarm_layout.addWidget(self.chk_alarm_clip)
        alarm_note = QLabel(
            "录像左上角会显示方案、产品、当前工序、进度和报警原因；"
            "同时生成 JSON 工序时间线。"
        )
        alarm_note.setWordWrap(True)
        alarm_note.setStyleSheet("color:#5f6368; font-size:12px;")
        alarm_layout.addWidget(alarm_note)
        alarm_row = QHBoxLayout()
        self.lbl_alarm_clip = QLabel("尚未生成报警录像")
        self.lbl_alarm_clip.setStyleSheet("color:#6c757d; font-size:12px;")
        self.btn_open_alarm_folder = QPushButton("📂 打开报警录像目录")
        self.btn_open_alarm_folder.clicked.connect(self._open_alarm_folder)
        alarm_row.addWidget(self.lbl_alarm_clip, stretch=1)
        alarm_row.addWidget(self.btn_open_alarm_folder)
        alarm_layout.addLayout(alarm_row)
        layout.addWidget(alarm_group)

        self.setLayout(layout)

    def _toggle_alarm_clip(self, state):
        self.vision_thread.alarm_clip_enabled = bool(state)

    def _open_alarm_folder(self):
        folder = os.path.abspath(self.vision_thread.alarm_clip_root)
        os.makedirs(folder, exist_ok=True)
        try:
            os.startfile(folder)
        except Exception as exc:
            QMessageBox.warning(self, "无法打开目录", str(exc))

    def _on_alarm_clip_saved(self, video_path):
        self.lbl_alarm_clip.setText(f"已保存：{os.path.basename(video_path)}")
        self.lbl_alarm_clip.setToolTip(video_path)

    def _take_photo(self):
        if not self.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先在主界面开启摄像头！")
            return
        self.vision_thread.req_take_photo = True

    def _record_start(self):
        if not self.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先在主界面开启摄像头！")
            return
        if self.vision_thread.is_recording:
            return
        rec_res = self.combo_record_res.currentText()
        if "1080p" in rec_res:
            self.vision_thread.record_width, self.vision_thread.record_height = 1920, 1080
        elif "2K" in rec_res:
            self.vision_thread.record_width, self.vision_thread.record_height = 2560, 1440
        elif "4K" in rec_res:
            self.vision_thread.record_width, self.vision_thread.record_height = 3840, 2160

        self.vision_thread.req_record_action = 'start'
        self._user_stopped = False
        self.btn_record_start.setEnabled(False)
        self.btn_record_pause.setEnabled(True)
        self.btn_record_pause.setText("⏸️ 暂停")
        self.btn_record_stop.setEnabled(True)
        self.lbl_record_time.setText("⏱️ 录制时长: 00:00")

    def _record_pause_resume(self):
        if not self.vision_thread.is_recording:
            self.btn_record_start.setEnabled(True)
            self.btn_record_pause.setEnabled(False)
            self.btn_record_stop.setEnabled(False)
            self.btn_record_pause.setText("⏸️ 暂停")
            return
        if self.vision_thread.is_record_paused:
            self.vision_thread.req_record_action = 'resume'
            self.btn_record_pause.setText("⏸️ 暂停")
        else:
            self.vision_thread.req_record_action = 'pause'
            self.btn_record_pause.setText("▶️ 继续")

    def _record_stop(self):
        self.vision_thread.req_record_action = 'stop'
        self._user_stopped = True
        self.btn_record_start.setEnabled(True)
        self.btn_record_pause.setEnabled(False)
        self.btn_record_stop.setEnabled(False)

    def _on_rec_time(self, time_str):
        self.lbl_record_time.setText(f"⏱️ 录制时长: {time_str}")
        if time_str == "00:00" and not self.vision_thread.is_recording:
            self.btn_record_start.setEnabled(True)
            self.btn_record_pause.setEnabled(False)
            self.btn_record_pause.setText("⏸️ 暂停")
            self.btn_record_stop.setEnabled(False)
        if time_str != "00:00":
            if int(time.time() * 2) % 2 == 0:
                self.lbl_record_time.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
            else:
                self.lbl_record_time.setStyleSheet("color: #aa0000; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_record_time.setStyleSheet("color: #555; font-weight: bold; font-size: 16px;")

    def closeEvent(self, event):
        if self.vision_thread.is_recording and not self._user_stopped:
            reply = QMessageBox.question(
                self, '确认关闭',
                '正在录制中，关闭此窗口将停止录制并自动保存视频。\n确定要关闭吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.vision_thread.req_record_action = 'stop'
        # 断开信号防止泄漏（每个对话框实例只连接一次）
        try:
            self.vision_thread.recording_time_signal.disconnect(self._on_rec_time)
        except TypeError:
            pass  # 已经断开过了
        try:
            self.vision_thread.alarm_clip_saved_signal.disconnect(self._on_alarm_clip_saved)
        except TypeError:
            pass
        event.accept()


class AlarmSettingsDialog(QDialog):
    """三色灯和蜂鸣器设置窗口。"""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("三色灯与蜂鸣器设置")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout()

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("串口:"))
        self.combo_alarm_port = QComboBox()
        self.combo_alarm_port.setEditable(True)
        self.combo_alarm_port.setToolTip("选择三色灯 USB 串口；也可以手动输入 COM7 之类的端口号")
        port_row.addWidget(self.combo_alarm_port, stretch=1)
        self.btn_alarm_refresh = QPushButton("刷新")
        self.btn_alarm_refresh.clicked.connect(app.refresh_alarm_ports)
        port_row.addWidget(self.btn_alarm_refresh)
        self.btn_alarm_apply = QPushButton("应用")
        self.btn_alarm_apply.clicked.connect(app.apply_alarm_port)
        port_row.addWidget(self.btn_alarm_apply)
        layout.addLayout(port_row)

        buzzer_row = QHBoxLayout()
        self.chk_alarm_buzzer = process_editor.VisibleCheckBox("启用蜂鸣器")
        self.chk_alarm_buzzer.setChecked(app.vision_thread.alarm_light.buzzer_enabled)
        self.chk_alarm_buzzer.setToolTip("测试时可以关闭蜂鸣器；红灯报警仍然生效")
        self.chk_alarm_buzzer.stateChanged.connect(app.toggle_alarm_buzzer)
        buzzer_row.addWidget(self.chk_alarm_buzzer)
        self.lbl_alarm_status = QLabel("")
        buzzer_row.addWidget(self.lbl_alarm_status, stretch=1)
        layout.addLayout(buzzer_row)

        hint = QLabel("换电脑后如果自动识别不准，在这里手动选择或输入 COM7、COM28 这类端口。")
        hint.setStyleSheet("color:#6c757d; font-size:12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.setLayout(layout)

        app.combo_alarm_port = self.combo_alarm_port
        app.chk_alarm_buzzer = self.chk_alarm_buzzer
        app.lbl_alarm_status = self.lbl_alarm_status
        app.refresh_alarm_ports()


class AoiSettingsDialog(QDialog):
    """AOI 建档设置窗口。"""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("AOI 特征建档")
        self.resize(720, 620)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout()

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("目标步骤:"))
        self.combo_aoi_step = QComboBox()
        self.combo_aoi_step.setToolTip("选择要配置 AOI 特征比对的工序步骤")
        self.combo_aoi_step.currentIndexChanged.connect(app.on_aoi_step_selected)
        step_row.addWidget(self.combo_aoi_step)
        layout.addLayout(step_row)

        anchor_row = QHBoxLayout()
        anchor_row.addWidget(QLabel("锚定物类别:"))
        self.combo_aoi_anchor = QComboBox()
        self.combo_aoi_anchor.setToolTip("选择 AOI 比对的目标类别")
        anchor_row.addWidget(self.combo_aoi_anchor)
        layout.addLayout(anchor_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("相似度阈值:"))
        self.aoi_thresh_slider = QSlider(Qt.Horizontal)
        self.aoi_thresh_slider.setRange(50, 99)
        self.aoi_thresh_slider.setValue(85)
        self.aoi_thresh_slider.setTickInterval(5)
        self.aoi_thresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.aoi_thresh_label = QLabel("0.85")
        self.aoi_thresh_label.setFixedWidth(35)
        self.aoi_thresh_label.setStyleSheet("font-weight: bold;")
        self.aoi_thresh_slider.valueChanged.connect(lambda v: self.aoi_thresh_label.setText(f"{v/100:.2f}"))
        thresh_row.addWidget(self.aoi_thresh_slider)
        thresh_row.addWidget(self.aoi_thresh_label)
        layout.addLayout(thresh_row)

        btn_row = QHBoxLayout()
        self.btn_aoi_capture = QPushButton("📷 抓拍标准样件")
        self.btn_aoi_capture.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_aoi_capture.setEnabled(False)
        self.btn_aoi_capture.clicked.connect(app.on_aoi_capture)
        self.btn_aoi_save = QPushButton("💾 保存 AOI 特征")
        self.btn_aoi_save.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold;")
        self.btn_aoi_save.setEnabled(False)
        self.btn_aoi_save.clicked.connect(app.on_aoi_save_feature)
        self.btn_aoi_archive = QPushButton("🧪 独立建档")
        self.btn_aoi_archive.setStyleSheet("background-color: #ff8c00; color: white; font-weight: bold;")
        self.btn_aoi_archive.clicked.connect(app.open_aoi_archive_dialog)
        btn_row.addWidget(self.btn_aoi_capture)
        btn_row.addWidget(self.btn_aoi_save)
        btn_row.addWidget(self.btn_aoi_archive)
        layout.addLayout(btn_row)

        self.aoi_preview_label = QLabel("(抓拍后显示)")
        self.aoi_preview_label.setAlignment(Qt.AlignCenter)
        self.aoi_preview_label.setMinimumSize(360, 220)
        self.aoi_preview_label.setStyleSheet("border: 1px dashed #ccc; background-color: #f0f0f0;")
        self.aoi_preview_scroll = QScrollArea()
        self.aoi_preview_scroll.setWidget(self.aoi_preview_label)
        self.aoi_preview_scroll.setWidgetResizable(False)
        self.aoi_preview_scroll.setMinimumHeight(260)
        self.aoi_preview_scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; background: #f8f9fa; }")
        layout.addWidget(self.aoi_preview_scroll)

        self.aoi_status_label = QLabel("使用前请先加载模型和方案。")
        self.aoi_status_label.setStyleSheet("color:#666; font-size:12px;")
        self.aoi_status_label.setWordWrap(True)
        layout.addWidget(self.aoi_status_label)

        self.setLayout(layout)

        app.combo_aoi_step = self.combo_aoi_step
        app.combo_aoi_anchor = self.combo_aoi_anchor
        app.aoi_thresh_slider = self.aoi_thresh_slider
        app.aoi_thresh_label = self.aoi_thresh_label
        app.btn_aoi_capture = self.btn_aoi_capture
        app.btn_aoi_save = self.btn_aoi_save
        app.aoi_preview_label = self.aoi_preview_label
        app.aoi_preview_scroll = self.aoi_preview_scroll
        app.aoi_status_label = self.aoi_status_label
        app.refresh_aoi_step_combo()


class AoiArchiveDialog(QDialog):
    """Standalone AOI golden-sample capture without starting workflow supervision."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("AOI 独立特征建档")
        self.resize(900, 650)
        self.cap = None
        self.pipeline = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.last_native_frame = None
        self.last_anchor_bbox = None
        self.captured_vector = None
        self.captured_crop = None
        self.captured_resolution = None

        layout = QVBoxLayout()
        form = QGridLayout()
        form.addWidget(QLabel("目标步骤:"), 0, 0)
        self.combo_step = QComboBox()
        self.combo_step.currentIndexChanged.connect(self._refresh_anchor_combo)
        form.addWidget(self.combo_step, 0, 1)
        form.addWidget(QLabel("锚定物类别:"), 1, 0)
        self.combo_anchor = QComboBox()
        form.addWidget(self.combo_anchor, 1, 1)
        form.addWidget(QLabel("相似度阈值:"), 2, 0)
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(50, 99)
        self.slider_threshold.setValue(85)
        self.lbl_threshold = QLabel("0.85")
        self.slider_threshold.valueChanged.connect(lambda v: self.lbl_threshold.setText(f"{v/100:.2f}"))
        form.addWidget(self.slider_threshold, 2, 1)
        form.addWidget(self.lbl_threshold, 2, 2)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("打开建档预览")
        self.btn_start.clicked.connect(self.start_preview)
        self.btn_capture = QPushButton("抓拍标准样件")
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.capture_sample)
        self.btn_save = QPushButton("保存 AOI 特征")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_feature)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_capture)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        self.preview_label = QLabel("打开预览后显示画面")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("border: 1px solid #d8dee9; background: #111; color: white;")
        layout.addWidget(self.preview_label, stretch=1)

        self.status_label = QLabel("先加载模型和方案，再打开独立建档预览。")
        self.status_label.setStyleSheet("color:#555;")
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self._refresh_step_combo()

    def _refresh_step_combo(self):
        self.combo_step.clear()
        for i, step in enumerate(self.app.vision_thread.process_steps):
            text = step.get("text", "")[:50]
            self.combo_step.addItem(f"步骤{i + 1}: {text}", i)
        self._refresh_anchor_combo()

    def _refresh_anchor_combo(self):
        self.combo_anchor.clear()
        step_idx = self.combo_step.currentData()
        if step_idx is None or step_idx >= len(self.app.vision_thread.process_steps):
            return
        step = self.app.vision_thread.process_steps[step_idx]
        targets = self.app.vision_thread._targets_for_step(step)
        for target in targets:
            for option in self.app.vision_thread.engine.target_options(target):
                zh = self.app.vision_thread.engine.eng_to_zh.get(option, option)
                self.combo_anchor.addItem(zh, option)

    def _selected_capture_size(self):
        text = self.app.combo_capture_res.currentText().split("(")[0].strip()
        w, h = text.split("×")
        return int(w), int(h)

    def start_preview(self):
        if self.app.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先停止主界面的工序监督，再打开独立建档预览。")
            return
        if self.app.vision_thread.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型。")
            return
        if self.combo_step.count() == 0:
            QMessageBox.warning(self, "提示", "当前方案没有工序步骤，请先配置工序。")
            return
        if self.combo_anchor.count() == 0:
            QMessageBox.warning(self, "提示", "当前步骤没有匹配到可建档的目标类别。")
            return

        self.stop_preview()
        source = self.app.combo_source.currentData()
        width, height = self._selected_capture_size()
        try:
            if source == "realsense":
                if not HAS_REALSENSE:
                    raise RuntimeError("当前环境没有安装 pyrealsense2")
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color)
                self.pipeline.start(config)
            else:
                self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                if not self.cap.isOpened():
                    raise RuntimeError("摄像头打开失败")
            self.timer.start(33)
            self.btn_start.setText("重启建档预览")
            self.btn_capture.setEnabled(True)
            self.status_label.setText("建档预览已开启，不会进入工序监督。")
        except Exception as exc:
            self.stop_preview()
            QMessageBox.critical(self, "建档预览失败", str(exc))

    def stop_preview(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.last_native_frame = None
        self.last_anchor_bbox = None
        self.btn_capture.setEnabled(False)

    def _read_frame(self):
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            frame = np.asanyarray(color_frame.get_data())
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def _find_anchor_bbox(self, native_frame, anchor_class):
        class_id = self.app.vision_thread.name_to_id_cache.get(anchor_class)
        if class_id is None:
            return None, 0.0
        kwargs = dict(classes=[class_id], verbose=False, conf=self.app.vision_thread.current_conf,
                      device=self.app.vision_thread.infer_device)
        if self.app.vision_thread.yolo_imgsz is not None:
            kwargs["imgsz"] = self.app.vision_thread.yolo_imgsz
        results = self.app.vision_thread.model(native_frame, **kwargs)
        if not results:
            return None, 0.0

        best_bbox = None
        best_conf = 0.0
        result = results[0]
        if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = box.xyxy[0].tolist()
        elif getattr(result, "obb", None) is not None and len(result.obb) > 0:
            for obb in result.obb:
                conf = float(obb.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = obb.xyxy[0].tolist()
        return best_bbox, best_conf

    def _update_frame(self):
        native_frame = self._read_frame()
        if native_frame is None:
            self.status_label.setText("未读取到画面。")
            return
        self.last_native_frame = native_frame.copy()
        anchor_class = self.combo_anchor.currentData()
        self.last_anchor_bbox = None
        display = apply_frame_transform(native_frame, self.app.combo_frame_transform.currentData()).copy()
        bbox, conf = self._find_anchor_bbox(native_frame, anchor_class)
        if bbox is not None:
            self.last_anchor_bbox = bbox
            h, w = native_frame.shape[:2]
            draw_bbox = transform_bbox_to_display(bbox, self.app.combo_frame_transform.currentData(), w, h)
            x1, y1, x2, y2 = map(int, draw_bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{anchor_class} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.status_label.setText(f"已找到锚定物 {anchor_class}，可抓拍。")
        else:
            self.status_label.setText(f"正在寻找锚定物 {anchor_class}...")

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pixmap)

    def _ensure_extractor(self):
        if self.app.vision_thread.aoi_extractor is None:
            from aoi_extractor import AOIFeatureExtractor
            device = get_safe_torch_device()
            self.app.vision_thread.aoi_extractor = AOIFeatureExtractor(backbone="resnet18", device=device)
        return self.app.vision_thread.aoi_extractor

    def capture_sample(self):
        if self.last_native_frame is None or self.last_anchor_bbox is None:
            QMessageBox.warning(self, "提示", "还没有找到锚定物，请让目标出现在画面中。")
            return
        extractor = self._ensure_extractor()
        self.captured_vector = extractor.extract(self.last_native_frame, self.last_anchor_bbox)
        self.captured_crop = extractor._crop(self.last_native_frame, self.last_anchor_bbox)
        if self.captured_crop is None:
            QMessageBox.warning(self, "提示", "裁剪标准样件失败，请重新抓拍。")
            return
        frame_h, frame_w = self.last_native_frame.shape[:2]
        crop_h, crop_w = self.captured_crop.shape[:2]
        self.captured_resolution = (frame_w, frame_h, crop_w, crop_h)
        self.btn_save.setEnabled(True)
        rgb = cv2.cvtColor(self.captured_crop, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_img))
        self.status_label.setText(f"已抓拍标准样件 | 原图 {frame_w}×{frame_h} | 裁剪 {crop_w}×{crop_h}")

    def save_feature(self):
        step_idx = self.combo_step.currentData()
        anchor_class = self.combo_anchor.currentData()
        if self.captured_vector is None or self.captured_crop is None:
            QMessageBox.warning(self, "提示", "请先抓拍标准样件。")
            return
        self.app.save_aoi_feature_to_config(
            step_idx,
            anchor_class,
            self.captured_vector,
            self.captured_crop,
            self.captured_resolution,
            self.slider_threshold.value() / 100.0,
        )
        self.status_label.setText("AOI 特征已保存，可关闭窗口后开始工序监督。")
        self.btn_save.setEnabled(False)

    def closeEvent(self, event):
        self.stop_preview()
        event.accept()


class MainTesterApp(QMainWindow):
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        os.makedirs("configs", exist_ok=True)
        super().__init__()
        self.setWindowTitle("智能工序指引系统")
        self.resize(1150, 800)
        self.app_start_time = datetime.now()  # 🌟 记录软件开启时间
        self.vision_thread = VisionThread()
        # 🌟 启动时加载历史 NG 记录
        ng_path = os.path.join(base_path, "logs", "ng_records.json")
        self.vision_thread.ng_tracker.load(ng_path)
        self.vision_thread.update_ui_signal.connect(self.update_ui)
        self.vision_thread.aoi_update_signal.connect(self.on_aoi_status_update)
        self.vision_thread.aoi_capture_done_signal.connect(self.on_aoi_capture_done)
        self.vision_thread.aoi_capture_failed_signal.connect(self.on_aoi_capture_failed)
        self.vision_thread.finished.connect(self.on_vision_thread_finished)
        self.current_config_path = ""
        self.checkboxes = {}
        self.camera_is_open = False
        self.active_source_type = None
        self._aoi_captured_vector = None
        self._aoi_captured_crop = None
        self._aoi_captured_resolution = None
        self.session_baseline_counts = self._snapshot_profile_counts()
        self.setup_ui()
        self.apply_app_style()
        self.refresh_model_list()

    # --- 录制与拍照子窗口 ---
    def open_recording_dialog(self):
        if not self.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先在主界面开启摄像头！")
            return
        if hasattr(self, '_rec_dlg') and self._rec_dlg is not None and self._rec_dlg.isVisible():
            self._rec_dlg.raise_()
            self._rec_dlg.activateWindow()
            return
        self._rec_dlg = RecordingDialog(self.vision_thread, self)
        self._rec_dlg.show()

    def _snapshot_profile_counts(self):
        snapshot = {}
        for profile, data in self.vision_thread.ng_tracker.profiles_data.items():
            snapshot[profile] = {
                "ok_count": data.get("ok_count", 0),
                "ng_count": data.get("ng_count", 0),
            }
        return snapshot

    def _session_delta_for_profile(self, profile_name):
        db = self.vision_thread.ng_tracker._get_current_db()
        baseline = self.session_baseline_counts.get(profile_name, {"ok_count": 0, "ng_count": 0})
        return {
            "ok": max(0, db.get("ok_count", 0) - baseline.get("ok_count", 0)),
            "ng": max(0, db.get("ng_count", 0) - baseline.get("ng_count", 0)),
            "total_ok": db.get("ok_count", 0),
            "total_ng": db.get("ng_count", 0),
        }

    def apply_app_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f7fb; font-family: "Microsoft YaHei", "SimHei"; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #24324b; }
            QPushButton {
                background: #ffffff;
                border: 1px solid #c7d0df;
                border-radius: 5px;
                padding: 6px 10px;
                color: #24324b;
            }
            QPushButton:hover { background: #eef4ff; border-color: #7aa7ff; }
            QPushButton:disabled { color: #99a1ad; background: #eef0f4; border-color: #d7dbe3; }
            QComboBox, QLineEdit {
                background: #ffffff;
                border: 1px solid #c7d0df;
                border-radius: 5px;
                padding: 4px 6px;
            }
            QTextBrowser {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 6px;
            }
            QSlider::groove:horizontal { height: 6px; background: #dce3ef; border-radius: 3px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #1a73e8; }
        """)
    def setup_ui(self):
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        # 顶部信息与控制栏
        top_bar_layout = QHBoxLayout()
        self.lbl_cycle = QLabel("📦 累计完成: 0 件")
        self.lbl_cycle.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a73e8;")

        self.lbl_ng = QLabel("❌ NG: 0 件")
        self.lbl_ng.setStyleSheet("font-size: 18px; font-weight: bold; color: #dc3545;")

        self.btn_view_ng = QPushButton("📋 查看 NG 记录")
        self.btn_view_ng.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px;")
        self.btn_view_ng.clicked.connect(self.show_ng_records)

        self.btn_skip = QPushButton("⏭️ 强制跳过 (Check)")
        self.btn_skip.setStyleSheet("background-color: #ffc107; font-weight: bold; padding: 5px;")
        self.btn_skip.clicked.connect(self.trigger_skip)

        self.btn_remediate = QPushButton("🛠️ 补救")
        self.btn_remediate.setStyleSheet("background-color: #eef0f4; color: #6c757d; font-weight: bold; padding: 5px;")
        self.btn_remediate.setEnabled(False)
        self.btn_remediate.clicked.connect(self.trigger_remediation)

        self.btn_reset = QPushButton("🔄 重新开始 (Reset)")
        self.btn_reset.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 5px;")
        self.btn_reset.clicked.connect(self.trigger_reset)

        self.btn_aoi_force = QPushButton("🔓 确认无误，强制放行")
        self.btn_aoi_force.setStyleSheet("background-color: #6c757d; color: white; padding: 5px;")
        self.btn_aoi_force.setEnabled(False)
        self.btn_aoi_force.clicked.connect(self.trigger_aoi_force)

        top_bar_layout.addWidget(self.lbl_cycle)
        top_bar_layout.addWidget(self.lbl_ng)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.btn_view_ng)
        top_bar_layout.addWidget(self.btn_skip)
        top_bar_layout.addWidget(self.btn_remediate)
        top_bar_layout.addWidget(self.btn_reset)
        top_bar_layout.addWidget(self.btn_aoi_force)
        video_layout.addLayout(top_bar_layout)

        # 状态面板
        self.status_banner = QTextBrowser()
        self.status_banner.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ced4da; border-radius: 8px;")
        self.status_banner.setMinimumHeight(150)
        self.status_banner.setMaximumHeight(200)
        video_layout.addWidget(self.status_banner)

        self.video_label = QLabel("等待开启摄像头/视频...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        video_layout.addWidget(self.video_label, stretch=1)
        main_layout.addLayout(video_layout, stretch=6)

        # --- 右侧控制台 ---
        control_layout = QVBoxLayout()

        # 1. 模型配置
        model_group = QGroupBox("📦 1. 模型与方案管理")
        m_layout = QVBoxLayout()

        # 行1：导入/删除/训练
        model_btn_layout = QHBoxLayout()
        self.btn_import_model = QPushButton("➕ 导入")
        self.btn_del_model = QPushButton("🗑️ 删除")
        self.btn_del_model.setStyleSheet("color: red;")
        self.btn_import_model.clicked.connect(self.import_new_model)
        self.btn_del_model.clicked.connect(self.delete_current_model)
        self.btn_fast_train = QPushButton("🚀 快速训练")
        self.btn_fast_train.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold;")
        self.btn_fast_train.clicked.connect(self.open_fast_trainer)
        model_btn_layout.addWidget(self.btn_import_model)
        model_btn_layout.addWidget(self.btn_del_model)
        model_btn_layout.addWidget(self.btn_fast_train)
        m_layout.addLayout(model_btn_layout)

        # 行2：模型单独一行，避免长模型名挤压；引擎/方案横向并排
        combo_grid = QGridLayout()
        combo_grid.addWidget(QLabel("模型:"), 0, 0)
        self.combo_models = QComboBox()
        self.combo_models.setMinimumWidth(60)
        self.combo_models.currentIndexChanged.connect(self.load_selected_model)
        combo_grid.addWidget(self.combo_models, 0, 1, 1, 3)
        combo_grid.addWidget(QLabel("引擎:"), 1, 0)
        self.combo_engine = QComboBox()
        self.combo_engine.addItem("PyTorch (.pt)")
        self.combo_engine.setMinimumWidth(60)
        self.combo_engine.currentIndexChanged.connect(self.on_engine_changed)
        combo_grid.addWidget(self.combo_engine, 1, 1)
        combo_grid.addWidget(QLabel("方案:"), 1, 2)
        self.combo_profiles = QComboBox()
        self.combo_profiles.setMinimumWidth(60)
        self.combo_profiles.currentIndexChanged.connect(self.load_selected_profile)
        combo_grid.addWidget(self.combo_profiles, 1, 3)
        combo_grid.setColumnStretch(1, 1)
        combo_grid.setColumnStretch(3, 1)
        m_layout.addLayout(combo_grid)

        # 行3：三个配置按钮横向排列
        cfg_btn_row = QHBoxLayout()
        self.btn_edit_mapping = QPushButton("📝 中英文映射")
        self.btn_edit_process = QPushButton("🧠 工序及安全")
        self.btn_edit_process.setStyleSheet("background-color: #e8f0fe; color: #1a73e8; font-weight: bold;")
        self.btn_export_onnx = QPushButton("🚀 导出ONNX")
        self.btn_export_onnx.setToolTip("将当前 .pt 模型转为 ONNX 格式")
        self.btn_edit_mapping.clicked.connect(self.open_mapping_dialog)
        self.btn_edit_process.clicked.connect(self.open_process_dialog)
        self.btn_export_onnx.clicked.connect(self.export_model_to_onnx)
        cfg_btn_row.addWidget(self.btn_edit_mapping)
        cfg_btn_row.addWidget(self.btn_edit_process)
        cfg_btn_row.addWidget(self.btn_export_onnx)
        m_layout.addLayout(cfg_btn_row)

        model_group.setLayout(m_layout)
        control_layout.addWidget(model_group)

        # 2. 目标过滤
        self.filter_group = QGroupBox("🎯 2. 目标过滤 (展示中文)")
        filter_outer_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_sel_all = QPushButton("✅ 全选")
        btn_desel_all = QPushButton("❌ 全不选")
        btn_sel_all.clicked.connect(lambda: self.set_all_filters(True))
        btn_desel_all.clicked.connect(lambda: self.set_all_filters(False))
        btn_layout.addWidget(btn_sel_all)
        btn_layout.addWidget(btn_desel_all)
        self.chk_chinese_label = process_editor.VisibleCheckBox("🀄 启用中文标签")
        self.chk_chinese_label.setStyleSheet("color: #d93025; font-weight: bold;")
        self.chk_chinese_label.setChecked(True)  # 默认启用，匹配 VisionThread 默认值
        self.chk_chinese_label.stateChanged.connect(self.toggle_chinese_label)
        btn_layout.addWidget(self.chk_chinese_label)
        self.chk_mediapipe = process_editor.VisibleCheckBox("🖐️ MediaPipe 手势识别")
        self.chk_mediapipe.setChecked(False)
        self.chk_mediapipe.setToolTip("默认关闭，可手动开启以检测手部动作；关闭时只做 YOLO 目标检测")
        self.chk_mediapipe.stateChanged.connect(self.toggle_mediapipe)
        btn_layout.addWidget(self.chk_mediapipe)
        filter_outer_layout.addLayout(btn_layout)
        self.filter_layout = QGridLayout()
        filter_content = QWidget()
        filter_content.setLayout(self.filter_layout)
        filter_scroll = QScrollArea()
        filter_scroll.setWidgetResizable(True)
        filter_scroll.setMinimumHeight(120)
        filter_scroll.setMaximumHeight(260)
        filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        filter_scroll.setWidget(filter_content)
        filter_outer_layout.addWidget(filter_scroll)

        # 置信度滑块
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("🎚️ YOLO 置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        self.conf_label = QLabel("0.25")
        self.conf_label.setFixedWidth(40)
        self.conf_label.setStyleSheet("font-weight: bold;")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        filter_outer_layout.addLayout(conf_layout)

        detection_filter_layout = QHBoxLayout()
        self.btn_same_class_box_filter = QPushButton("同类框去重/稳定：关")
        self.btn_same_class_box_filter.setCheckable(True)
        self.btn_same_class_box_filter.setChecked(False)
        self.btn_same_class_box_filter.setToolTip(
            "默认关闭。开启后只处理高度重叠的同类别框：压掉低置信度重复框，"
            "并对连续帧位置做轻量平滑；彼此分开的同类物体不会合并。"
        )
        self.btn_same_class_box_filter.toggled.connect(self.toggle_same_class_box_filter)
        self.btn_cross_class_box_filter = QPushButton("异类框互斥：关")
        self.btn_cross_class_box_filter.setCheckable(True)
        self.btn_cross_class_box_filter.setChecked(False)
        self.btn_cross_class_box_filter.setToolTip(
            "默认关闭。开启后，同一位置高度重叠但类别不同的框只保留置信度最高者。"
            "它用于缓解一个物体同时显示多个类别，根治仍需补充并纠正训练数据。"
        )
        self.btn_cross_class_box_filter.toggled.connect(self.toggle_cross_class_box_filter)
        detection_filter_layout.addWidget(self.btn_same_class_box_filter)
        detection_filter_layout.addWidget(self.btn_cross_class_box_filter)
        filter_outer_layout.addLayout(detection_filter_layout)
        detection_filter_note = QLabel("两个功能默认关闭；关闭时推理参数和检测结果保持原样。")
        detection_filter_note.setStyleSheet("color:#5f6368;")
        detection_filter_note.setWordWrap(True)
        filter_outer_layout.addWidget(detection_filter_note)
        self._refresh_detection_filter_buttons()

        self.filter_group.setLayout(filter_outer_layout)
        control_layout.addWidget(self.filter_group)

        # 3. 推流控制
        cam_group = QGroupBox("🎥 3. 推流与播放控制")
        c_layout = QVBoxLayout()

        # 摄像头源 + 采集分辨率
        cam_row1 = QHBoxLayout()
        cam_row1.addWidget(QLabel("摄像头:"))
        self.combo_source = QComboBox()
        for cam_idx in range(4):
            self.combo_source.addItem(f"摄像头 {cam_idx}", cam_idx)
        if HAS_REALSENSE:
            self.combo_source.addItem("RealSense 彩色相机", "realsense")
        self.combo_source.setToolTip("OpenCV 摄像头编号会随设备、驱动、插拔顺序变化；哪个有画面就用哪个，依次尝试 0/1/2/3。")
        cam_row1.addWidget(self.combo_source, stretch=2)
        cam_row1.addWidget(QLabel("分辨率:"))
        self.combo_capture_res = QComboBox()
        self.combo_capture_res.addItems(["1280×720", "1920×1080", "2560×1440", "3840×2160 (4K)"])
        self.combo_capture_res.setCurrentIndex(1)  # 默认 1920×1080
        self.combo_capture_res.setToolTip(
            "采集分辨率在打开摄像头时应用；切换后请停止并重新打开摄像头。"
            "摄像头不支持该档位时，驱动可能回退到最接近的分辨率。"
        )
        cam_row1.addWidget(self.combo_capture_res, stretch=2)
        c_layout.addLayout(cam_row1)

        cam_row2 = QHBoxLayout()
        cam_row2.addWidget(QLabel("方向:"))
        self.combo_frame_transform = QComboBox()
        self.combo_frame_transform.addItem("正常", "none")
        self.combo_frame_transform.addItem("上下翻转", "flip_v")
        self.combo_frame_transform.addItem("左右翻转", "flip_h")
        self.combo_frame_transform.addItem("180°", "rotate_180")
        self.combo_frame_transform.currentIndexChanged.connect(self.on_frame_transform_changed)
        cam_row2.addWidget(self.combo_frame_transform)

        self.btn_cam = QPushButton("打开选定相机")
        self.btn_cam.clicked.connect(self.toggle_camera)
        cam_row2.addWidget(self.btn_cam)

        # YOLO 输入尺寸 + 视频导入
        cam_row2.addWidget(QLabel("YOLO:"))
        self.combo_yolo_imgsz = QComboBox()
        self.combo_yolo_imgsz.addItems(["默认 (通常640)", "640", "960", "1088", "1280"])
        self.combo_yolo_imgsz.setCurrentText("1280")
        self.combo_yolo_imgsz.setEditable(True)
        self.combo_yolo_imgsz.setToolTip(
            "YOLO 推理输入尺寸（长边约束并保持画面比例后补边）。"
            "“默认”表示使用 Ultralytics 框架默认值，通常是 640，并不等于训练图片或摄像头的原始分辨率。"
        )
        cam_row2.addWidget(self.combo_yolo_imgsz)

        self.btn_vid = QPushButton("导入视频文件")
        self.btn_vid.clicked.connect(lambda: self.start_vision(self.select_video_file(), is_video=True))
        cam_row2.addWidget(self.btn_vid)
        cam_row2.addWidget(QLabel("倍速:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["1x 正常", "2x 加速", "4x 加速", "8x 极速"])
        self.combo_speed.currentIndexChanged.connect(self.change_video_speed)
        cam_row2.addWidget(self.combo_speed)
        c_layout.addLayout(cam_row2)

        cam_group.setLayout(c_layout)
        control_layout.addWidget(cam_group)

        # 4. 录制与拍照入口
        rec_entry_row = QHBoxLayout()
        self.btn_open_recording = QPushButton("📸 录制与拍照")
        self.btn_open_recording.setStyleSheet("background-color: #d93025; color: white; font-weight: bold; padding: 8px; font-size: 14px;")
        self.btn_open_recording.clicked.connect(self.open_recording_dialog)
        rec_entry_row.addWidget(self.btn_open_recording)
        control_layout.addLayout(rec_entry_row)

        # 5. 工具入口
        tools_row = QHBoxLayout()
        self.btn_alarm_settings = QPushButton("🚨 报警设置")
        self.btn_alarm_settings.setStyleSheet("background-color: #fff3cd; font-weight: bold; padding: 8px;")
        self.btn_alarm_settings.clicked.connect(self.open_alarm_settings_dialog)
        self.btn_aoi_settings = QPushButton("🔬 AOI 建档")
        self.btn_aoi_settings.setStyleSheet("background-color: #e8f0fe; color: #1a73e8; font-weight: bold; padding: 8px;")
        self.btn_aoi_settings.clicked.connect(self.open_aoi_settings_dialog)
        tools_row.addWidget(self.btn_alarm_settings)
        tools_row.addWidget(self.btn_aoi_settings)
        control_layout.addLayout(tools_row)
        control_layout.addStretch()

        control_panel = QWidget()
        control_panel.setLayout(control_layout)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setWidget(control_panel)
        control_scroll.setMinimumWidth(520)
        main_layout.addWidget(control_scroll, stretch=4)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def toggle_chinese_label(self, state):
        # 🌟 修复：直接读取复选框的布尔状态，不和枚举对象做比较了
        self.vision_thread.use_chinese_labels = self.chk_chinese_label.isChecked()

    def toggle_mediapipe(self, state):
        enabled = self.chk_mediapipe.isChecked()
        self.vision_thread.intent_engine.enable_mediapipe = enabled
        label = "已启用" if enabled else "已关闭 (仅YOLO推理)"
        self.status_banner.append(f"<div style='color:#1a73e8;'>🖐️ MediaPipe 手势识别: {label}</div>")

    def on_conf_slider_changed(self, value):
        conf = value / 100.0
        self.conf_label.setText(f"{conf:.2f}")
        self.vision_thread.current_conf = conf

    @staticmethod
    def _style_detection_filter_button(button, enabled, enabled_text, disabled_text):
        button.setText(enabled_text if enabled else disabled_text)
        button.setStyleSheet(
            "background:#188038; color:white; font-weight:bold; padding:6px;"
            if enabled else
            "background:#f1f3f4; color:#3c4043; padding:6px;"
        )

    def _refresh_detection_filter_buttons(self):
        if hasattr(self, "btn_same_class_box_filter"):
            self._style_detection_filter_button(
                self.btn_same_class_box_filter,
                self.btn_same_class_box_filter.isChecked(),
                "同类框去重/稳定：开",
                "同类框去重/稳定：关",
            )
        if hasattr(self, "btn_cross_class_box_filter"):
            self._style_detection_filter_button(
                self.btn_cross_class_box_filter,
                self.btn_cross_class_box_filter.isChecked(),
                "异类框互斥：开",
                "异类框互斥：关",
            )

    def toggle_same_class_box_filter(self, enabled):
        enabled = bool(enabled)
        self.vision_thread.same_class_box_filter_enabled = enabled
        self.vision_thread.reset_detection_postprocessing()
        self._refresh_detection_filter_buttons()
        state = "已开启" if enabled else "已关闭（原始识别）"
        self.status_banner.append(f"<div style='color:#1a73e8;'>🎯 同类框去重/稳定: {state}</div>")

    def toggle_cross_class_box_filter(self, enabled):
        enabled = bool(enabled)
        self.vision_thread.cross_class_box_filter_enabled = enabled
        self.vision_thread.reset_detection_postprocessing()
        self._refresh_detection_filter_buttons()
        state = "已开启" if enabled else "已关闭（原始识别）"
        self.status_banner.append(f"<div style='color:#1a73e8;'>🎯 异类框互斥: {state}</div>")

    def on_frame_transform_changed(self):
        self.vision_thread.frame_transform = self.combo_frame_transform.currentData() or "none"

    def on_engine_changed(self):
        """推理引擎切换时重载模型"""
        self.refresh_engine_combo()
        if self.combo_models.count() > 0:
            self.load_selected_model()
            engine = self.combo_engine.currentText()
            self.status_banner.append(f"<div style='color:#1a73e8;'>🔄 切换推理引擎: {engine}</div>")

    def refresh_engine_combo(self):
        """根据当前模型是否存在 .onnx 文件来更新引擎选项"""
        model_name = self.combo_models.currentText()
        self.combo_engine.blockSignals(True)
        current = self.combo_engine.currentText()
        self.combo_engine.clear()
        self.combo_engine.addItem("PyTorch (.pt)")
        if model_name:
            onnx_path = f"models/{model_name}.onnx"
            if os.path.exists(onnx_path):
                self.combo_engine.addItem("ONNX (.onnx)")
        # 恢复之前的选择（如果还存在的话）
        idx = self.combo_engine.findText(current)
        if idx >= 0:
            self.combo_engine.setCurrentIndex(idx)
        self.combo_engine.blockSignals(False)

    def _set_stream_ui_state(self, active, source_type=None):
        self.active_source_type = source_type if active else None
        self.camera_is_open = active and source_type in ("webcam", "4k_cam", "realsense")

        if not active:
            self.btn_cam.setText("打开选定相机")
            self.combo_speed.setEnabled(True)
            if hasattr(self, "btn_aoi_capture"):
                self.btn_aoi_capture.setEnabled(False)
            self.btn_aoi_force.setEnabled(False)
            return

        if source_type == "video":
            self.btn_cam.setText("停止视频")
            self.combo_speed.setEnabled(True)
        else:
            self.btn_cam.setText("关闭相机")
            self.combo_speed.setEnabled(False)
        if hasattr(self, "btn_aoi_capture") and hasattr(self, "combo_aoi_step"):
            self.btn_aoi_capture.setEnabled(self.combo_aoi_step.count() > 0)

    def _apply_yolo_imgsz_setting(self):
        yolo_text = self.combo_yolo_imgsz.currentText().strip()
        try:
            self.vision_thread.yolo_imgsz = int(yolo_text)
        except ValueError:
            self.vision_thread.yolo_imgsz = None

    def on_vision_thread_finished(self):
        if self.vision_thread.isRunning():
            return
        self._set_stream_ui_state(False)

    # --- 控制逻辑 ---
    def toggle_camera(self):
        if self.vision_thread.isRunning() and self.active_source_type == "video":
            self.vision_thread.stop()
            self.video_label.clear()
            self.video_label.setText("视频已停止")
            self._set_stream_ui_state(False)
            return

        if not self.camera_is_open:
            try:
                if self.vision_thread.isRunning():
                    self.vision_thread.stop()
                    self._set_stream_ui_state(False)

                selected_source = self.combo_source.currentData()
                if selected_source == "realsense":
                    self.vision_thread.source = None
                    self.vision_thread.source_type = "realsense"
                else:
                    self.vision_thread.source = selected_source
                    self.vision_thread.source_type = "webcam"

                # 采集分辨率（如 "3840×2160 (4K)" → 3840, 2160）
                cap_res = self.combo_capture_res.currentText()
                cap_res = cap_res.split("(")[0].strip()  # 去掉可能的 "(4K)" 后缀
                w, h = cap_res.split("×")
                self.vision_thread.capture_width = int(w)
                self.vision_thread.capture_height = int(h)

                # YOLO 输入尺寸（"默认" → None，使用模型原生分辨率）
                self._apply_yolo_imgsz_setting()
                self.on_frame_transform_changed()

                self.vision_thread.speed_multiplier = 1
                self.vision_thread.prepare_for_new_stream()
                self.vision_thread.start()

                self._set_stream_ui_state(True, self.vision_thread.source_type)
            except Exception as e:
                self._set_stream_ui_state(False)
                QMessageBox.critical(self, "摄像头错误", f"无法打开摄像头: {str(e)}\n\n请检查:\n1. 摄像头是否已连接\n2. 摄像头是否被其他程序占用\n3. 分辨率是否支持")
                self.status_banner.append(f"<div style='color:#d93025;'>❌ 摄像头开启失败: {e}</div>")
        else:
            self.vision_thread.stop()
            self.video_label.clear()
            self.video_label.setText("相机已关闭")
            self._set_stream_ui_state(False)

    def select_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        return path

    def start_vision(self, source, is_video=False):
        if source is None or source == "": return
        try:
            if self.vision_thread.isRunning():
                self.vision_thread.stop()
                self._set_stream_ui_state(False)
            self.vision_thread.source = source
            if is_video:
                self.vision_thread.source_type = "video"
                self.change_video_speed()
            self._apply_yolo_imgsz_setting()
            self.on_frame_transform_changed()
            self.vision_thread.prepare_for_new_stream()
            self.vision_thread.start()
            self._set_stream_ui_state(True, self.vision_thread.source_type)
        except Exception as e:
            self._set_stream_ui_state(False)
            QMessageBox.critical(self, "视频错误", f"无法打开视频文件: {str(e)}\n\n请检查文件是否损坏或格式不支持")
            self.status_banner.append(f"<div style='color:#d93025;'>❌ 视频导入失败: {e}</div>")

    def change_video_speed(self):
        idx = self.combo_speed.currentIndex()
        speed = 2 ** idx
        self.vision_thread.speed_multiplier = speed

    def trigger_skip(self):
        self.vision_thread.force_skip_signal = True

    def trigger_remediation(self):
        if not self.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先开启工序监督，再进行补救。")
            return
        if self.vision_thread.remediation_mode:
            choice = QMessageBox.question(
                self,
                "退出补救",
                "当前补救步骤还没有完成。确定要手动退出补救状态，回到正常工序吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if choice == QMessageBox.StandardButton.Yes:
                self.vision_thread.remediation_cancel_signal = True
            return

        skipped = self.vision_thread.ng_tracker.get_skipped_indices()
        if not skipped:
            QMessageBox.information(self, "补救", "当前没有已跳过且未补救的步骤。")
            return

        if len(skipped) == 1:
            selected_idx = skipped[0]
        else:
            labels = []
            label_to_idx = {}
            for idx in skipped:
                text = self.vision_thread.process_steps[idx].get("text", "") if idx < len(self.vision_thread.process_steps) else ""
                label = f"步骤 {idx + 1}: {text[:40]}"
                labels.append(label)
                label_to_idx[label] = idx
            item, ok = QInputDialog.getItem(self, "选择补救步骤", "选择要补救的步骤:", labels, 0, False)
            if not ok:
                return
            selected_idx = label_to_idx[item]

        self.vision_thread.remediation_request_idx = selected_idx
        self.status_banner.append(f"<div style='color:#ff8c00;'>🛠️ 已进入补救准备：步骤 {selected_idx + 1}</div>")

    def trigger_reset(self):
        choice = QMessageBox.question(self, '确认重新开始', '当前产品会记录为“手动重新开始，未完成”。确定要重新开始下一件吗？',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if choice == QMessageBox.StandardButton.Yes:
            self.vision_thread.reset_signal = True

    def open_alarm_settings_dialog(self):
        if hasattr(self, "_alarm_settings_dlg") and self._alarm_settings_dlg is not None and self._alarm_settings_dlg.isVisible():
            self._alarm_settings_dlg.raise_()
            self._alarm_settings_dlg.activateWindow()
            return
        if not hasattr(self, "_alarm_settings_dlg") or self._alarm_settings_dlg is None:
            self._alarm_settings_dlg = AlarmSettingsDialog(self, self)
        else:
            self.refresh_alarm_ports()
        self._alarm_settings_dlg.show()

    def open_aoi_settings_dialog(self):
        if hasattr(self, "_aoi_settings_dlg") and self._aoi_settings_dlg is not None and self._aoi_settings_dlg.isVisible():
            self._aoi_settings_dlg.raise_()
            self._aoi_settings_dlg.activateWindow()
            return
        if not hasattr(self, "_aoi_settings_dlg") or self._aoi_settings_dlg is None:
            self._aoi_settings_dlg = AoiSettingsDialog(self, self)
        else:
            self.refresh_aoi_step_combo()
        self._aoi_settings_dlg.show()

    def refresh_alarm_ports(self):
        if not hasattr(self, "combo_alarm_port"):
            return
        current_data = self.combo_alarm_port.currentData()
        current = "" if current_data == "" else self.combo_alarm_port.currentText().strip()
        active_port = self.vision_thread.alarm_light.port or ""
        preferred = current or active_port

        self.combo_alarm_port.blockSignals(True)
        self.combo_alarm_port.clear()
        self.combo_alarm_port.addItem("自动识别", "")
        ports = list_serial_port_options()
        for port in ports:
            device = port.get("device", "")
            desc = port.get("description", "")
            label = f"{device} - {desc}" if desc else device
            self.combo_alarm_port.addItem(label, device)

        if preferred:
            idx = self.combo_alarm_port.findData(preferred)
            if idx >= 0:
                self.combo_alarm_port.setCurrentIndex(idx)
            else:
                self.combo_alarm_port.setEditText(preferred)
        elif active_port:
            self.combo_alarm_port.setEditText(active_port)
        self.combo_alarm_port.blockSignals(False)

        if active_port:
            self.lbl_alarm_status.setText(f"当前端口: {active_port}")
        elif ports:
            self.lbl_alarm_status.setText("请选择串口后点应用")
        else:
            self.lbl_alarm_status.setText("未发现串口")

    def apply_alarm_port(self):
        selected = self.combo_alarm_port.currentData()
        if selected is None:
            selected = self.combo_alarm_port.currentText().strip()
            if " - " in selected:
                selected = selected.split(" - ", 1)[0].strip()
        enabled, port = self.vision_thread.alarm_light.set_port(selected)
        if enabled and port:
            self.lbl_alarm_status.setText(f"已使用端口: {port}")
            self.status_banner.append(f"<div style='color:#1a73e8;'>🚨 三色灯串口已设置为 {port}</div>")
        else:
            self.lbl_alarm_status.setText("三色灯未启用")
            self.status_banner.append("<div style='color:#d93025;'>🚨 三色灯未启用：未选择串口或串口不可用</div>")

    def toggle_alarm_buzzer(self, state):
        enabled = self.chk_alarm_buzzer.isChecked()
        self.vision_thread.alarm_light.set_buzzer_enabled(enabled)
        label = "已启用" if enabled else "已关闭"
        self.lbl_alarm_status.setText(f"蜂鸣器{label}；端口: {self.vision_thread.alarm_light.port or '未设置'}")
        self.status_banner.append(f"<div style='color:#1a73e8;'>🔊 蜂鸣器{label}</div>")

    # --- AOI 特征建档回调 ---
    def open_aoi_archive_dialog(self):
        if hasattr(self, "_aoi_archive_dlg") and self._aoi_archive_dlg is not None and self._aoi_archive_dlg.isVisible():
            self._aoi_archive_dlg.raise_()
            self._aoi_archive_dlg.activateWindow()
            return
        self._aoi_archive_dlg = AoiArchiveDialog(self, self)
        self._aoi_archive_dlg.show()

    def refresh_aoi_step_combo(self):
        """刷新 AOI 步骤下拉框"""
        if not hasattr(self, "combo_aoi_step"):
            return
        self.combo_aoi_step.blockSignals(True)
        self.combo_aoi_step.clear()
        steps = self.vision_thread.process_steps
        if steps:
            for i, s in enumerate(steps):
                text = s.get('text', '')[:40]
                self.combo_aoi_step.addItem(f"步骤{i+1}: {text}", i)
        self.combo_aoi_step.blockSignals(False)
        if self.combo_aoi_step.count() > 0:
            self.on_aoi_step_selected(0)

    def on_aoi_step_selected(self, index):
        if not hasattr(self, "combo_aoi_anchor"):
            return
        self.combo_aoi_anchor.clear()
        self.btn_aoi_save.setEnabled(False)
        self.aoi_preview_label.clear()
        self.aoi_preview_label.setText("(抓拍后显示)")
        self._aoi_captured_vector = None
        self._aoi_captured_crop = None
        self._aoi_captured_resolution = None
        step_idx = self.combo_aoi_step.currentData()
        if step_idx is None or not self.vision_thread.process_steps:
            self.btn_aoi_capture.setEnabled(False)
            return
        step_dict = self.vision_thread.process_steps[step_idx]
        targets = self.vision_thread._targets_for_step(step_dict)
        for t in targets:
            for option in self.vision_thread.engine.target_options(t):
                zh = self.vision_thread.engine.eng_to_zh.get(option, option)
                self.combo_aoi_anchor.addItem(zh, option)
        aoi_cfg = step_dict.get('aoi_feature_check', {})
        if aoi_cfg.get('enabled'):
            threshold = aoi_cfg.get('threshold', 0.85)
            self.aoi_thresh_slider.setValue(int(threshold * 100))
            anchor_class = aoi_cfg.get('anchor_class', '')
            idx = self.combo_aoi_anchor.findData(anchor_class)
            if idx >= 0:
                self.combo_aoi_anchor.setCurrentIndex(idx)
            res = aoi_cfg.get('capture_resolution', {})
            if res:
                self.aoi_status_label.setText(
                    f"已加载现有 AOI 配置 | 原图 {res.get('frame_width', '?')}×{res.get('frame_height', '?')} | "
                    f"裁剪 {res.get('crop_width', '?')}×{res.get('crop_height', '?')}"
                )
            else:
                self.aoi_status_label.setText("已加载现有 AOI 配置")
            # 尝试恢复之前保存的特征图缩略图
            if self.current_config_path:
                model_name = os.path.splitext(os.path.basename(self.current_config_path))[0].replace('_map', '')
                profile_name = self.combo_profiles.currentText()
                thumb_path = f"aoi_captures/{model_name}_{profile_name}_step{step_idx}.png"
                if os.path.exists(thumb_path):
                    pixmap = QPixmap(thumb_path)
                    if not pixmap.isNull():
                        self.aoi_preview_label.setPixmap(pixmap)
                        self.aoi_preview_label.resize(pixmap.size())
                        self.btn_aoi_save.setEnabled(False)  # 已有配置，无需重新保存（可替换）
            # 将已有向量加载到内存，允许替换
            std_vec = aoi_cfg.get('standard_vector')
            if std_vec:
                self._aoi_captured_vector = np.array(std_vec, dtype=np.float32)
        else:
            self.aoi_thresh_slider.setValue(85)
            self.aoi_status_label.setText("")
        self.btn_aoi_capture.setEnabled(self.vision_thread.isRunning())

    def on_aoi_capture(self):
        if not self.vision_thread.isRunning():
            QMessageBox.warning(self, "提示", "请先开启摄像头！")
            return
        if not self.vision_thread.process_steps:
            QMessageBox.warning(self, "提示", "当前配置没有工序步骤！\n\n请先在「工序编辑器」中为当前方案添加步骤，并在步骤描述中写明目标类别名称。")
            return
        anchor_class = self.combo_aoi_anchor.currentData()
        if anchor_class is None:
            QMessageBox.warning(self, "提示", "请先选择一个锚定物类别！\n\n如果下拉为空，说明该步骤描述中未匹配到已知类别，请检查步骤描述和模型映射。")
            return
        self.btn_aoi_save.setEnabled(False)
        self._aoi_captured_vector = None
        self._aoi_captured_crop = None
        self._aoi_captured_resolution = None
        self.vision_thread.req_aoi_capture = True
        self.vision_thread._aoi_capture_anchor = anchor_class
        self.vision_thread._aoi_capture_ttl = 90  # ~3秒超时（30fps）
        self.aoi_status_label.setText(f"正在抓拍【{self.combo_aoi_anchor.currentText()}】，请保持锚定物在画面中...")

    def on_aoi_capture_done(self, feature_vector, crop_img, resolution_info):
        self._aoi_captured_vector = feature_vector
        self._aoi_captured_crop = crop_img.copy()  # 留一份用于后续保存
        self._aoi_captured_resolution = resolution_info
        if not hasattr(self, "btn_aoi_save"):
            return
        self.btn_aoi_save.setEnabled(True)
        rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.aoi_preview_label.setPixmap(pixmap)
        self.aoi_preview_label.resize(pixmap.size())
        frame_w, frame_h, crop_w, crop_h = resolution_info
        self.aoi_status_label.setText(
            f"特征已提取 | 原图 {frame_w}×{frame_h} | 裁剪 {crop_w}×{crop_h}，请调整阈值后保存"
        )

    def on_aoi_capture_failed(self, message):
        if hasattr(self, "btn_aoi_save"):
            self.btn_aoi_save.setEnabled(False)
        self._aoi_captured_vector = None
        self._aoi_captured_crop = None
        self._aoi_captured_resolution = None
        if hasattr(self, "aoi_status_label"):
            self.aoi_status_label.setText(message)
        QMessageBox.warning(self, "AOI 抓拍失败", message)

    def save_aoi_feature_to_config(self, step_idx, anchor_class, feature_vector, crop_img, resolution_info, threshold):
        if not self.current_config_path:
            QMessageBox.warning(self, "提示", "请先在主界面加载一个模型配置！")
            return False
        with open(self.current_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        process_editor.normalize_profile_config(data)
        profile_name = self.combo_profiles.currentText()
        if not profile_name or profile_name not in data.get("profiles", {}):
            profile_name = data.get("active_profile")
        steps = data['profiles'][profile_name]['process_steps'] if profile_name and profile_name in data.get('profiles', {}) else data['process_steps']
        if step_idx is None:
            QMessageBox.warning(self, "提示", "请先选择目标步骤。")
            return False
        if step_idx < len(steps):
            aoi_feature_check = {
                'enabled': True,
                'anchor_class': anchor_class,
                'standard_vector': feature_vector.tolist(),
                'threshold': threshold,
                'timeout': 5.0
            }
            if resolution_info:
                frame_w, frame_h, crop_w, crop_h = resolution_info
                aoi_feature_check['capture_resolution'] = {
                    'frame_width': int(frame_w),
                    'frame_height': int(frame_h),
                    'crop_width': int(crop_w),
                    'crop_height': int(crop_h),
                }
            steps[step_idx]['aoi_feature_check'] = aoi_feature_check
        else:
            QMessageBox.warning(self, "提示", "目标步骤不存在，请刷新方案后重试。")
            return False
        with open(self.current_config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # 同时保存特征图缩略图，重启后可恢复预览
        os.makedirs("aoi_captures", exist_ok=True)
        model_name = os.path.splitext(os.path.basename(self.current_config_path))[0].replace('_map', '')
        thumb_path = f"aoi_captures/{model_name}_{profile_name}_step{step_idx}.png"
        if crop_img is not None:
            cv2.imwrite(thumb_path, crop_img)
        self.load_selected_profile()
        self.refresh_aoi_step_combo()
        if hasattr(self, "aoi_status_label"):
            self.aoi_status_label.setText("AOI 配置已保存!")
        self.status_banner.append("<div style='color:#28a745;'>AOI 特征已建档保存!</div>")
        return True

    def on_aoi_save_feature(self):
        step_idx = self.combo_aoi_step.currentData()
        anchor_class = self.combo_aoi_anchor.currentData()
        threshold = self.aoi_thresh_slider.value() / 100.0
        if self._aoi_captured_vector is None or step_idx is None:
            QMessageBox.warning(self, "提示", "请先抓拍标准样件！\n\n使用步骤：\n1. 左侧选择目标步骤\n2. 选择锚定物类别\n3. 点击「抓拍标准样件」\n4. 调整阈值后点击「保存 AOI 特征」")
            return
        self.save_aoi_feature_to_config(
            step_idx,
            anchor_class,
            self._aoi_captured_vector,
            self._aoi_captured_crop,
            self._aoi_captured_resolution,
            threshold,
        )

    def on_aoi_status_update(self, similarity, state, is_blocked):
        self.btn_aoi_force.setEnabled(is_blocked)
        if is_blocked:
            self.btn_aoi_force.setStyleSheet(
                "background-color: #dc3545; color: white; font-weight: bold; padding: 5px; font-size: 14px;")
        elif state == 'checking':
            self.btn_aoi_force.setEnabled(False)
            self.btn_aoi_force.setStyleSheet("background-color: #6c757d; color: white; padding: 5px;")
        else:
            self.btn_aoi_force.setEnabled(False)
            self.btn_aoi_force.setStyleSheet("background-color: #6c757d; color: white; padding: 5px;")

    def trigger_aoi_force(self):
        self.vision_thread.aoi_force_signal = True

    # --- NG 记录查看 ---
    def show_ng_records(self):
        tracker = self.vision_thread.ng_tracker
        dates = tracker.get_available_dates()
        profile_name = tracker.active_profile  # 获取当前方案名

        dlg = QDialog(self)
        dlg.setWindowTitle(f"异常/NG 记录 - 【{profile_name}】")
        dlg.resize(800, 600)

        layout = QVBoxLayout()

        # 顶部统计 + 日期过滤
        top_row = QHBoxLayout()
        # 👇 直接读取纯净的统计数据，去掉了进行中的干扰
        summary = QLabel(f"📊 当前方案: <b>{profile_name}</b> | "
                         f"<span style='color:#28a745;'>✅ 累计完成: {tracker.ok_count} 件</span> | "
                         f"<span style='color:#dc3545;'>❌ NG: {tracker.ng_count} 件</span>")
        summary.setStyleSheet("font-size: 16px;")
        top_row.addWidget(summary)
        top_row.addStretch()

        top_row.addWidget(QLabel("筛选日期:"))
        combo_date = QComboBox()
        combo_date.addItem("全部日期", "")
        for d in dates:
            combo_date.addItem(d, d)
        combo_date.setMinimumWidth(110)
        top_row.addWidget(combo_date)
        layout.addLayout(top_row)

        # 结果区域
        ng_browser = QTextBrowser()
        ng_browser.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(ng_browser)

        def refresh_view(date_filter=''):
            ng_list = tracker.get_ng_products(date_filter if date_filter else None)
            if not ng_list:
                ng_browser.setHtml("<p style='color:#6c757d;'>暂无异常/NG 记录</p>")
                return
            html = "<div style='font-family: Microsoft YaHei, SimHei;'>"
            current_date = ''
            for p in reversed(ng_list):
                pdate = p.get('date', '')
                if pdate != current_date:
                    current_date = pdate
                    html += f"<h2 style='color:#1a73e8; margin-top:15px;'>📅 {current_date}</h2>"
                pid = p.get('id', '?')
                st = p.get('start_time', '')
                end = p.get('end_time', '')
                reason = p.get('ng_reason', '')

                # 👇 新增：如果 end_time 是空的，说明是实时拦截到的、工人还在做的新鲜 NG 产品
                status_label = "" if end else " <i>(流水线进行中)</i>"
                end_str = end if end else "未完成..."

                product_status = p.get('status', 'NG')
                if product_status == 'OK':
                    title_color = '#1a73e8'
                    title_text = f"🔄 产品 #{pid}{status_label} — 异常已处理，最终 OK"
                else:
                    title_color = '#dc3545'
                    title_text = f"🚫 产品 #{pid}{status_label} — NG ({reason})"
                html += f"<hr><h3 style='color:{title_color};'>{title_text}</h3>"
                html += f"<b>时间:</b> {st} ~ {end_str}<br>"
                # 👇 优化 UI 显示，干掉底部刷屏的告警，融合进表格状态中
                html += "<table style='width:100%; border-collapse:collapse;'>"
                html += "<tr style='background:#eee;'><th style='text-align:left;padding:4px;'>步骤</th><th style='text-align:left;padding:4px;'>内容</th><th style='text-align:left;padding:4px;'>状态</th></tr>"

                # 提前把当前产品的告警记录提取出来
                alarms = p.get('jump_alarms', [])

                for rec in p.get('step_records', []):
                    status = rec.get('status', 'pending')
                    step_num = rec['step']

                    # 检查当前步骤是否触发过跳步告警
                    has_alarm = any(a['current_step'] == step_num for a in alarms)

                    if status == 'completed':
                        aoi_note = rec.get('aoi_note', '')
                        aoi_forced = rec.get('aoi_forced', False)
                        aoi_recovered = rec.get('aoi_recovered', False)
                        aoi_blocked = rec.get('aoi_blocked', False)
                        if aoi_forced:
                            badge = f'<span style="color:#ff8c00;">⚠️ AOI未通过(人工放行)</span>'
                        elif aoi_blocked:
                            badge = f'<span style="color:#dc3545;">🚫 AOI 阻塞中 (相似度: {rec.get("aoi_similarity", 0):.2%})</span>'
                        elif aoi_recovered:
                            badge = '<span style="color:#28a745;">✅ AOI 恢复通过</span>'
                        elif rec.get('timeout_warning'):
                            badge = '<span style="color:#ff8c00;">⏱️ 已完成（曾超时提醒）</span>'
                        else:
                            badge = '<span style="color:#28a745;">✅ 已完成</span>'
                    elif status == 'skipped':
                        reason = rec.get("reason", "")
                        if rec.get('aoi_blocked'):
                            badge = f'<span style="color:#dc3545;">🚫 AOI 阻塞 (未放行)</span>'
                        else:
                            badge = f'<span style="color:#dc3545;">❌ 跳过 ({reason})</span>'
                    elif status == 'remedied':
                        badge = f'<span style="color:#1a73e8;">🔄 已补救 ({rec.get("reason", "")})</span>'
                    else:
                        # 如果还没做，但触发了跳步/AOI/超时事件
                        if rec.get('aoi_blocked'):
                            badge = f'<span style="color:#dc3545; font-weight:bold;">🚫 AOI 阻塞中 (相似度: {rec.get("aoi_similarity", 0):.2%})</span>'
                        elif has_alarm:
                            badge = '<span style="color:#d93025; font-weight:bold;">⚠️ 触发跳步 (未补救)</span>'
                        elif rec.get('timeout_warning'):
                            badge = '<span style="color:#ff8c00;">⏱️ 超时提醒</span>'
                        else:
                            badge = '<span style="color:#6c757d;">⚪ 未进行</span>'

                    html += f"<tr><td style='padding:4px;'>{step_num}</td><td style='padding:4px;'>{rec['text']}</td><td style='padding:4px;'>{badge}</td></tr>"
                html += "</table>"
                # 🌟 原来这里有一段 if alarms: 的代码，专门在底部打印告警列表，现在已经被彻底删除了！
            html += "</div>"
            ng_browser.setHtml(html)

        combo_date.currentTextChanged.connect(lambda: refresh_view(combo_date.currentData()))
        refresh_view()

        # 底部按钮行：刷新 + 关闭
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("🔄 刷新")
        btn_refresh.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold;")
        btn_refresh.clicked.connect(lambda: refresh_view(combo_date.currentData()))
        btn_row.addWidget(btn_refresh)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.setLayout(layout)
        dlg.exec()

    # --- 配置与模型管理 ---
    def import_new_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "",
                                                   "模型文件 (*.pt *.onnx);;PyTorch (*.pt);;ONNX (*.onnx)")
        if not file_path: return

        model_name, ok = QInputDialog.getText(self, "命名", "请输入模型名称:")
        if not ok or not model_name.strip(): return
        model_name = model_name.strip()

        ext = os.path.splitext(file_path)[1]  # .pt 或 .onnx
        new_model_path = f"models/{model_name}{ext}"
        config_path = f"configs/{model_name}_map.json"

        try:
            # 👇 核心修复：判断路径是否一致。如果选中的正是 models 目录下的文件，直接跳过复制！
            if os.path.abspath(file_path) != os.path.abspath(new_model_path):
                shutil.copy(file_path, new_model_path)

            self.current_config_path = config_path
            model_manager.ModelMappingDialog(new_model_path, config_path, self).exec()

            # 刷新列表
            self.refresh_model_list()

            # 💡 额外优化：导入成功后，下拉框自动帮你选中刚刚导入的模型
            idx = self.combo_models.findText(model_name)
            if idx >= 0:
                self.combo_models.setCurrentIndex(idx)

        except Exception as e:
            # 报错绝不能吞掉，一定要弹窗告诉你
            QMessageBox.critical(self, "导入失败", f"发生错误：{str(e)}")

    # 删除模型逻辑
    def delete_current_model(self):
        model_name = self.combo_models.currentText()
        if not model_name: return

        reply = QMessageBox.question(self, '确认删除',
                                     f"确定要彻底删除模型【{model_name}】及其所有工序配置吗？\n此操作不可恢复！",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.vision_thread.stop()
            self._set_stream_ui_state(False)
            self.video_label.setText("模型已删除，请重新选择")
            for ext in ['.pt', '.onnx']:
                path = f"models/{model_name}{ext}"
                if os.path.exists(path):
                    os.remove(path)
            json_path = f"configs/{model_name}_map.json"
            if os.path.exists(json_path): os.remove(json_path)
            self.refresh_model_list()
            QMessageBox.information(self, "成功", "模型及配置已删除！")
    def refresh_model_list(self):
        self.combo_models.blockSignals(True)
        self.combo_models.clear()
        if os.path.exists("configs"):
            for filename in os.listdir("configs"):
                if filename.endswith("_map.json"):
                    self.combo_models.addItem(filename.replace("_map.json", ""))
        self.combo_models.blockSignals(False)
        self.refresh_engine_combo()
        # 列表刷新后，自动加载第一个
        if self.combo_models.count() > 0:
            self.load_selected_model()

    def open_fast_trainer(self):
        if not hasattr(self, 'trainer_dlg') or self.trainer_dlg is None:
            self.trainer_dlg = fast_trainer.FastTrainerDialog(base_path, self)
        self.trainer_dlg.show()
        self.trainer_dlg.activateWindow()

    def load_selected_model(self):
        model_name = self.combo_models.currentText()
        if not model_name: return
        restart_source = None
        if self.vision_thread.isRunning():
            restart_source = (self.vision_thread.source, self.vision_thread.source_type)
            self.vision_thread.stop()
            self._set_stream_ui_state(False)

        self.current_config_path = f"configs/{model_name}_map.json"

        # 清理旧的复选框
        for i in reversed(range(self.filter_layout.count())):
            self.filter_layout.itemAt(i).widget().setParent(None)
        self.checkboxes.clear()

        if os.path.exists(self.current_config_path):
            with open(self.current_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                process_editor.normalize_profile_config(data)

                # 根据用户选择的推理引擎加载 .pt 或 .onnx
                torch_device = get_safe_torch_device()
                model_path = data["model_path"]
                pt_path = f"models/{model_name}.pt"
                if model_path.endswith('.pt') and os.path.exists(model_path):
                    pt_path = model_path
                onnx_path = f"models/{model_name}.onnx"
                if model_path.endswith('.onnx') and os.path.exists(model_path):
                    onnx_path = model_path

                engine = self.combo_engine.currentText()
                if 'ONNX' in engine and os.path.exists(onnx_path):
                    self.vision_thread.infer_device = get_onnxruntime_device_arg(torch_device)
                    self.vision_thread.model = YOLO(onnx_path, task='obb')
                    log_runtime_device(
                        f"Inference loaded: engine=ONNX, model={onnx_path}, "
                        f"device={describe_infer_device(engine, self.vision_thread.infer_device, torch_device)}"
                    )
                    self.status_banner.append(
                        f"<div style='color:#1a73e8;'>ONNX 推理设备: {self.vision_thread.infer_device}</div>"
                    )
                elif os.path.exists(pt_path):
                    self.vision_thread.infer_device = get_ultralytics_device_arg(torch_device)
                    self.vision_thread.model = YOLO(pt_path, task='obb').to(torch_device)
                    log_runtime_device(
                        f"Inference loaded: engine=PyTorch, model={pt_path}, "
                        f"device={describe_infer_device(engine, self.vision_thread.infer_device, torch_device)}"
                    )
                elif os.path.exists(onnx_path):
                    self.vision_thread.infer_device = get_onnxruntime_device_arg(torch_device)
                    self.vision_thread.model = YOLO(onnx_path, task='obb')
                    log_runtime_device(
                        f"Inference loaded: engine=ONNX fallback, model={onnx_path}, "
                        f"device={describe_infer_device('ONNX', self.vision_thread.infer_device, torch_device)}"
                    )
                    self.status_banner.append(
                        f"<div style='color:#1a73e8;'>仅找到 ONNX 模型，已使用 ONNX 推理设备: {self.vision_thread.infer_device}</div>"
                    )
                else:
                    raise FileNotFoundError(f"未找到模型文件: {pt_path} 或 {onnx_path}")

                # 更新下拉框
                self.combo_profiles.blockSignals(True)
                self.combo_profiles.clear()
                profiles = data.get("profiles", {})
                if profiles:
                    self.combo_profiles.addItems(profiles.keys())
                    active = data.get("active_profile", list(profiles.keys())[0])
                    self.combo_profiles.setCurrentText(active)
                self.combo_profiles.blockSignals(False)

                # 调用加载当前方案
                self.load_selected_profile(restart_stream=False)

                # 重建 UI 复选框
                row, col = 0, 0
                max_cols = 3
                for class_id_str, info in data["mapping"].items():
                    class_id = int(class_id_str)
                    display_text = info["zh_name"]
                    cb = process_editor.VisibleCheckBox(f"{display_text}")
                    cb.setStyleSheet("""
                        QCheckBox {
                            background: #ffffff;
                            border: 1px solid #c7d0df;
                            border-radius: 5px;
                            padding: 4px 6px;
                            color: #6c757d;
                        }
                        QCheckBox:checked {
                            background: #1a73e8;
                            border-color: #0b57d0;
                            color: white;
                            font-weight: bold;
                        }
                        QCheckBox::indicator {
                            width: 16px;
                            height: 16px;
                        }
                    """)
                    cb.setChecked(True)
                    cb.stateChanged.connect(self.update_vision_targets)
                    self.filter_layout.addWidget(cb, row, col)
                    self.checkboxes[class_id] = cb
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1

            self.update_vision_targets()

            if restart_source and restart_source[0] is not None:
                self.vision_thread.source, self.vision_thread.source_type = restart_source
                self.vision_thread.prepare_for_new_stream()
                self.vision_thread.start()
                self._set_stream_ui_state(True, self.vision_thread.source_type)

    # 🌟 新增：切换并加载选中方案
    def load_selected_profile(self, restart_stream=True):
        profile_name = self.combo_profiles.currentText()
        # 🌟 修复 1：删掉了对 profile_name 为空的拦截！
        # 即使刚导入新模型没有工序方案，也要往下走，把基础的中文字典加载进去！
        if not self.current_config_path: return

        restart_source = None
        if restart_stream and self.vision_thread.isRunning():
            restart_source = (self.vision_thread.source, self.vision_thread.source_type)
            self.vision_thread.stop()
            self._set_stream_ui_state(False)

        with open(self.current_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            process_editor.normalize_profile_config(data)
            if not profile_name or profile_name not in data.get("profiles", {}):
                profile_name = data["active_profile"]

            # 如果有选中的方案，则读取工序配置
            if profile_name and "profiles" in data and profile_name in data["profiles"]:
                profile_data = data["profiles"][profile_name]
                data["process_steps"] = profile_data.get("process_steps", [])
                data["forbidden_items"] = profile_data.get("forbidden_items", "")
                data["state_alarm_rules"] = profile_data.get(
                    "state_alarm_rules", process_editor.DEFAULT_STATE_ALARM_RULES
                )
                data["state_alarm_confirm_frames"] = profile_data.get(
                    "state_alarm_confirm_frames", process_editor.DEFAULT_STATE_ALARM_CONFIRM_FRAMES
                )
                data["state_alarm_release_frames"] = profile_data.get(
                    "state_alarm_release_frames", process_editor.DEFAULT_STATE_ALARM_RELEASE_FRAMES
                )
                data["state_alarm_padding_ratio"] = profile_data.get(
                    "state_alarm_padding_ratio", process_editor.DEFAULT_STATE_ALARM_PADDING_RATIO
                )
                data["toggle_state_monitors"] = profile_data.get(
                    "toggle_state_monitors", process_editor.DEFAULT_TOGGLE_STATE_MONITORS
                )
                data["state_conditional_rules"] = profile_data.get(
                    "state_conditional_rules", process_editor.DEFAULT_STATE_CONDITIONAL_RULES
                )
                data["slot_monitors"] = profile_data.get(
                    "slot_monitors", process_editor.DEFAULT_SLOT_MONITORS
                )
                data["result_monitor_stages"] = profile_data.get(
                    "result_monitor_stages", process_editor.DEFAULT_RESULT_MONITOR_STAGES
                )
                data["step_timeout"] = profile_data.get("step_timeout", process_editor.DEFAULT_STEP_TIMEOUT)
                data["jump_monitor_scope"] = profile_data.get(
                    "jump_monitor_scope", process_editor.DEFAULT_JUMP_MONITOR_SCOPE
                )
                data["jump_strong_action_enabled"] = profile_data.get(
                    "jump_strong_action_enabled", process_editor.DEFAULT_JUMP_STRONG_ACTION_ENABLED
                )
                data["jump_strong_action_frames"] = profile_data.get(
                    "jump_strong_action_frames", process_editor.DEFAULT_JUMP_STRONG_ACTION_FRAMES
                )
                data["jump_ignore_static_intersection"] = profile_data.get(
                    "jump_ignore_static_intersection", process_editor.DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
                )
                data["jump_progress_visible"] = profile_data.get(
                    "jump_progress_visible", process_editor.DEFAULT_JUMP_PROGRESS_VISIBLE
                )
            else:
                # 即使没有工序，也给个空壳，保证不出错
                data["process_steps"] = []
                data["forbidden_items"] = ""
                data["state_alarm_rules"] = process_editor.DEFAULT_STATE_ALARM_RULES
                data["state_alarm_confirm_frames"] = process_editor.DEFAULT_STATE_ALARM_CONFIRM_FRAMES
                data["state_alarm_release_frames"] = process_editor.DEFAULT_STATE_ALARM_RELEASE_FRAMES
                data["state_alarm_padding_ratio"] = process_editor.DEFAULT_STATE_ALARM_PADDING_RATIO
                data["toggle_state_monitors"] = process_editor.DEFAULT_TOGGLE_STATE_MONITORS
                data["state_conditional_rules"] = process_editor.DEFAULT_STATE_CONDITIONAL_RULES
                data["slot_monitors"] = process_editor.DEFAULT_SLOT_MONITORS
                data["result_monitor_stages"] = process_editor.DEFAULT_RESULT_MONITOR_STAGES
                data["step_timeout"] = process_editor.DEFAULT_STEP_TIMEOUT
                data["jump_monitor_scope"] = process_editor.DEFAULT_JUMP_MONITOR_SCOPE
                data["jump_strong_action_enabled"] = process_editor.DEFAULT_JUMP_STRONG_ACTION_ENABLED
                data["jump_strong_action_frames"] = process_editor.DEFAULT_JUMP_STRONG_ACTION_FRAMES
                data["jump_ignore_static_intersection"] = process_editor.DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
                data["jump_progress_visible"] = process_editor.DEFAULT_JUMP_PROGRESS_VISIBLE

            # 👇 先通知 NG 追踪器切换到当前方案的数据隔离区，避免新方案产品记到旧方案里
            if profile_name:
                if self.vision_thread.ng_tracker.current_product:
                    if self.vision_thread.ng_tracker.has_product_activity(min_elapsed_sec=5):
                        self.vision_thread.ng_tracker.finalize_as_ng('切换模型/方案时产品未完成')
                    else:
                        self.vision_thread.ng_tracker.current_product = None
                model_name = self.combo_models.currentText().strip()
                tracker_profile = f"{model_name} / {profile_name}" if model_name else profile_name
                self.vision_thread.ng_tracker.switch_profile(tracker_profile)
            # 核心：不管有没有工序，都把包含 mapping(中英文映射) 的 data 喂给底层线程
            self.vision_thread.load_config(data)
            self.refresh_aoi_step_combo()
        if restart_source and restart_source[0] is not None:
            self.vision_thread.source, self.vision_thread.source_type = restart_source
            self.vision_thread.prepare_for_new_stream()
            self.vision_thread.start()
            self._set_stream_ui_state(True, self.vision_thread.source_type)
    def set_all_filters(self, state):
        for cb in self.checkboxes.values(): cb.setChecked(state)
    def update_vision_targets(self):
        selected_ids = []
        for class_id, cb in self.checkboxes.items():
            if cb.isChecked(): selected_ids.append(class_id)
        self.vision_thread.selected_class_ids = selected_ids

    def export_model_to_onnx(self):
        """将当前 .pt 模型导出为 ONNX 格式"""
        model_name = self.combo_models.currentText()
        if not model_name:
            QMessageBox.warning(self, "提示", "请先选择一个模型！")
            return
        pt_path = f"models/{model_name}.pt"
        onnx_path = f"models/{model_name}.onnx"
        if not os.path.exists(pt_path):
            QMessageBox.warning(self, "提示", f"未找到 .pt 文件: {pt_path}")
            return
        if os.path.exists(onnx_path):
            reply = QMessageBox.question(self, "确认覆盖",
                f"ONNX 文件已存在，是否覆盖？\n{onnx_path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        restart_source = None
        if self.vision_thread.isRunning():
            restart_source = (self.vision_thread.source, self.vision_thread.source_type)
            self.vision_thread.stop()
            self._set_stream_ui_state(False)
            self.vision_thread.ng_tracker.current_product = None
        try:
            self.status_banner.append("<div style='color:#1a73e8;'>🔄 正在导出 ONNX 模型，请稍候...</div>")
            QApplication.processEvents()
            torch_device = get_safe_torch_device()
            export_device = get_ultralytics_device_arg(torch_device)
            model = YOLO(pt_path, task='obb')
            model.export(format='onnx', dynamic=True, simplify=True, opset=12, device=export_device)
            self.refresh_engine_combo()
            self.combo_engine.blockSignals(True)
            self.combo_engine.setCurrentText("ONNX (.onnx)")
            self.combo_engine.blockSignals(False)
            self.load_selected_model()
            if restart_source and restart_source[0] is not None:
                self.vision_thread.source, self.vision_thread.source_type = restart_source
                self.vision_thread.prepare_for_new_stream()
                self.vision_thread.start()
                self._set_stream_ui_state(True, self.vision_thread.source_type)
            self.status_banner.append(
                "<div style='color:#28a745; font-weight:bold;'>✅ ONNX 导出成功，已自动切换到 ONNX 引擎！</div>")
        except Exception as e:
            if restart_source and restart_source[0] is not None and not self.vision_thread.isRunning():
                self.vision_thread.source, self.vision_thread.source_type = restart_source
                self.vision_thread.prepare_for_new_stream()
                self.vision_thread.start()
                self._set_stream_ui_state(True, self.vision_thread.source_type)
            self.status_banner.append(f"<div style='color:#d93025;'>❌ ONNX 导出失败: {e}</div>")

    def open_mapping_dialog(self):
        if not self.current_config_path: return
        with open(self.current_config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if model_manager.ModelMappingDialog(data["model_path"], self.current_config_path, self).exec():
            self.load_selected_model()

    def open_process_dialog(self):
        if not self.current_config_path: return
        if process_editor.ProcessGuideDialog(self.current_config_path, self).exec():
            self.load_selected_model()

    # --- UI 渲染 ---
    def update_ui(self, cv_img, process_steps, current_idx, is_pausing, progress, alert_msg, cycles, sub_count):
        self.lbl_cycle.setText(f"📦 累计完成: {cycles} 件")
        self.lbl_ng.setText(f"❌ NG: {self.vision_thread.ng_tracker.ng_count} 件")
        if self.vision_thread.remediation_mode:
            self.btn_remediate.setText("🛠️ 退出补救")
            self.btn_remediate.setEnabled(True)
            self.btn_remediate.setStyleSheet("background-color: #ff8c00; color: white; font-weight: bold; padding: 5px;")
        else:
            skipped_count = len(self.vision_thread.ng_tracker.get_skipped_indices())
            self.btn_remediate.setText(f"🛠️ 补救 ({skipped_count})" if skipped_count else "🛠️ 补救")
            self.btn_remediate.setEnabled(skipped_count > 0)
            self.btn_remediate.setStyleSheet(
                "background-color: #ff8c00; color: white; font-weight: bold; padding: 5px;"
                if skipped_count else
                "background-color: #eef0f4; color: #6c757d; font-weight: bold; padding: 5px;"
            )
        html_content = "<div style='line-height: 1.6; font-size: 16px; padding: 5px;'>"

        def detach_config_html(step_dict):
            if step_dict.get("action_type") != "detach":
                return ""
            removed = str(step_dict.get("detach_removed", "")).strip()
            base = str(step_dict.get("detach_base", "")).strip()
            status = step_dict.get("_detach_status") or {}
            removed = removed or status.get("removed_name", "")
            base = base or status.get("base_name", "")
            if not removed and not base:
                return ""
            missing_text = ""
            if bool(step_dict.get("detach_missing_enabled", False)):
                missing_frames = int(step_dict.get("detach_missing_frames", 30) or 30)
                missing_padding = float(step_dict.get("detach_missing_padding_ratio", 0.15) or 0.0)
                missing_text = f" / 消失判定: {missing_frames}帧, 外扩{missing_padding:.2f}"
            return (
                " <span style='background-color:#eef5ff; color:#1a73e8; padding:2px 8px; "
                f"border-radius:4px; font-size: 14px;'>拆除物: {removed or '-'} / 基准物: {base or '-'}"
                f"{missing_text}</span>"
            )

        def detach_hint_html(step_dict, original_idx):
            if original_idx != current_idx or step_dict.get("action_type") != "detach":
                return detach_config_html(step_dict)
            status = step_dict.get("_detach_status") or {}
            if not status:
                return detach_config_html(step_dict)
            state = status.get("state", "missing")
            if state == "detached":
                evidence = status.get("evidence")
                text = "当前状态：已拆除（消失确认）" if evidence == "disappearance" else "当前状态：已拆除"
                bg = "#28a745"
            elif state == "waiting_assembly":
                bg, text = "#6c757d", "当前状态：等待前置装配"
            elif state == "attached":
                bg, text = "#1a73e8", "当前状态：已贴合，等待分离"
            elif state == "separating":
                bg, text = "#ff8c00", "当前状态：分离中"
            elif state == "disappearing":
                frames = int(status.get("missing_frames", 0) or 0)
                required = int(status.get("missing_required_frames", 30) or 30)
                bg, text = "#ff8c00", f"当前状态：安装区已空 {frames}/{required}帧"
            elif state == "waiting_attach":
                bg, text = "#6c757d", "当前状态：等待先贴合"
            elif state == "missing":
                bg, text = "#6c757d", "当前状态：等待目标"
            else:
                bg, text = "#dc3545", "当前状态：未拆除"
            return (
                detach_config_html(step_dict) +
                f" <span style='background-color:{bg}; color:white; padding:2px 8px; "
                f"border-radius:4px; font-size: 15px;'>🔧 {text}</span>"
            )

        def action_hint_html(step_dict):
            status = step_dict.get("_action_status") or {}
            state = status.get("state")
            if state == "waiting_action":
                text, bg = "等待动作", "#6c757d"
            elif state == "arming":
                frames = int(status.get("touch_frames", 0) or 0)
                required = int(status.get("required_touch_frames", 8) or 8)
                text, bg = f"动作确认中 {frames}/{required}", "#ff8c00"
            elif state == "armed":
                text, bg = "动作确认中", "#1a73e8"
            else:
                return ""
            return (
                f" <span style='background-color:{bg}; color:white; padding:2px 8px; "
                f"border-radius:4px; font-size: 15px;'>🧭 {text}</span>"
            )

        def release_hint_html(step_dict):
            status = step_dict.get("_release_status") or {}
            state = status.get("state")
            relation_ready = bool(status.get("relation_confirmed", False))
            release_frames = int(status.get("release_frames", 0) or 0)
            required_frames = int(status.get("required_release_frames", 0) or 0)
            relation_progress = int(status.get("relation_progress", 0) or 0)
            if state == "operating":
                text = (
                    "操作中：空间关系已满足，等待离手"
                    if relation_ready else "操作中：手仍在当前工序区域"
                )
                bg = "#ff8c00"
            elif state == "waiting_release":
                text = f"离手确认中 {release_frames}/{required_frames}"
                bg = "#1a73e8"
            elif state == "waiting_hand":
                text = (
                    "空间关系已满足，但尚未识别到手触达目标"
                    if relation_progress >= 100 else "等待手进入当前工序操作区"
                )
                bg = "#6c757d"
            elif state == "waiting_relation":
                text = f"已离手，等待空间关系稳定 {relation_progress}%"
                bg = "#6c757d"
            else:
                return ""
            return (
                f" <span style='background-color:{bg}; color:white; padding:2px 8px; "
                f"border-radius:4px; font-size: 15px;'>🖐️ {text}</span>"
            )

        def prereq_hint_html(step_dict):
            status = step_dict.get("_prereq_status") or {}
            missing = status.get("missing") or []
            if not missing:
                return ""
            missing_text = "、".join(str(int(idx) + 1) for idx in missing)
            return (
                " <span style='background-color:#d93025; color:white; padding:2px 8px; "
                f"border-radius:4px; font-size: 15px;'>⛔ 等待前置步骤 {missing_text}</span>"
            )

        def result_check_hint_html(step_dict):
            status = step_dict.get("_result_check_status") or {}
            if not status:
                return ""
            target = status.get("target", "")
            hits = int(status.get("hits", 0) or 0)
            required = int(status.get("required_hits", 0) or 0)
            if status.get("satisfied"):
                bg, text = "#188038", f"最终结果通过：{target}"
            elif status.get("hand_in_zone"):
                bg, text = "#ff8c00", "等待手离开后确认未识别状态"
            elif status.get("condition_met"):
                bg, text = "#1a73e8", f"最终结果确认中：{target} {hits}/{required}帧"
            elif status.get("mode") == "absent":
                bg, text = "#d93025", f"应为关闭状态，但仍识别到：{target}"
            else:
                bg, text = "#d93025", f"等待最终识别结果：{target}"
            return (
                f" <span style='background-color:{bg}; color:white; padding:2px 8px; "
                f"border-radius:4px; font-size:15px;'>✅ {text}</span>"
            )

        state_statuses = list(getattr(self.vision_thread, "toggle_state_statuses", []) or [])
        slot_statuses = list(getattr(self.vision_thread, "slot_monitor_statuses", []) or [])
        sequence_statuses = list(
            getattr(self.vision_thread, "result_sequence_statuses", []) or []
        )
        slot_expectation = dict(
            getattr(self.vision_thread, "slot_expectation_status", {}) or {}
        )
        if state_statuses or slot_statuses or sequence_statuses:
            html_content += (
                "<div style='background:#f4f7fb; border:1px solid #c8d5e8; "
                "border-radius:7px; padding:8px 10px; margin-bottom:10px;'>"
                "<div style='color:#174a8b; font-weight:bold;'>📡 实时状态监测</div>"
            )
            phase_names = {
                "ready": "就绪", "confirming_touch": "触发确认中",
                "waiting_release": "等待离手", "waiting_ready": "等待初始离手",
                "waiting_action": "等待本工序动作完成",
                "waiting_completion": "等待工序先完成100%",
                "waiting_group_actions": "等待乱序组全部动作完成",
                "target_missing": "未看到触发目标", "monitoring": "稳定监测",
                "operating": "操作遮挡中", "settling": "离手稳定中",
                "calibrating": "学习接口位置", "waiting_anchor": "等待板子",
                "invalid_anchor": "锚点无效", "waiting_targets": "等待线缆完整出现",
                "confirming_order": "确认线缆顺序",
                "waiting_change": "等待开始下一次换线", "waiting_result": "等待完整正确结果",
                "wrong": "接线结果错误", "cycle_complete": "本块板子验收通过",
            }
            for status in state_statuses:
                phase = phase_names.get(status.get("phase"), status.get("phase", ""))
                html_content += (
                    f"<div style='margin-top:3px;'><b>{status.get('name', '')}</b>："
                    f"<span style='color:#d93025; font-weight:bold;'>{status.get('state', '')}</span>"
                    f" <span style='color:#6b7280; font-size:13px;'>({phase}，已切换"
                    f"{int(status.get('toggle_count', 0) or 0)}次)</span></div>"
                )
            for status in slot_statuses:
                phase = phase_names.get(status.get("phase"), status.get("phase", ""))
                if status.get("monitor_type") == "relative_order":
                    order = status.get("actual_order") or status.get("visible_order") or []
                    slot_text = "从左/上到右/下：" + (" → ".join(order) if order else "等待线缆")
                else:
                    slot_text = "；".join(
                        f"{name}={value}" for name, value in (status.get("slots", {}) or {}).items()
                    )
                html_content += (
                    f"<div style='margin-top:3px;'><b>{status.get('name', '')}</b>：{slot_text or '待学习'}"
                    f" <span style='color:#6b7280; font-size:13px;'>({phase})</span></div>"
                )
                if status.get("configured") and status.get("step_idx") is not None:
                    if status.get("phase") == "waiting_group_actions":
                        color, result_text = (
                            "#174a8b", "组内接线动作可任意顺序，全部完成后再验收最终接线结果"
                        )
                    elif status.get("phase") == "waiting_action":
                        color, result_text = (
                            "#174a8b", "先完成本步骤原有接线动作，完成后再验收接线结果"
                        )
                    elif status.get("satisfied"):
                        color, result_text = "#188038", "当前步骤接线结果正确"
                    elif status.get("settled") and status.get("mismatches"):
                        color, result_text = "#d93025", "；".join(status.get("mismatches", []))
                    else:
                        color, result_text = "#b06000", "等待本步骤线缆完整出现并离手确认"
                    html_content += (
                        f"<div style='color:{color}; font-weight:bold; margin-left:12px;'>"
                        f"↳ {result_text}</div>"
                    )
                if status.get("continuous_enabled"):
                    if status.get("continuous_satisfied"):
                        color, result_text = "#188038", "接线位置正确"
                    elif status.get("continuous_settled") and status.get("continuous_mismatches"):
                        color = "#d93025"
                        result_text = "；".join(status.get("continuous_mismatches", []))
                    else:
                        color, result_text = "#b06000", "等待线缆完整出现并离手确认"
                    html_content += (
                        f"<div style='color:{color}; font-weight:bold; margin-left:12px;'>"
                        f"↳ {result_text}</div>"
                    )
            for status in sequence_statuses:
                phase = status.get("phase", "")
                current = int(status.get("current_number", 0) or 0)
                total = int(status.get("total_stages", 0) or 0)
                expected_text = " → ".join(status.get("expected_order", []) or []) or "未配置"
                observed_order = (
                    status.get("actual_order")
                    or status.get("visible_order")
                    or status.get("live_order")
                    or []
                )
                actual_text = " → ".join(observed_order) or "尚未纳入板子范围"
                if phase == "cycle_complete":
                    color = "#188038"
                    detail = (
                        f"✅ 本块板子全部通过（累计{int(status.get('cycle_count', 0) or 0)}块），"
                        "移出当前板子后自动回到第1关"
                    )
                elif phase == "wrong":
                    color = "#d93025"
                    detail = f"❌ 当前：{actual_text}；正确应为：{expected_text}"
                elif phase == "waiting_change":
                    color = "#1a73e8"
                    detail = f"上一关已通过，等待开始换线；本关目标：{expected_text}"
                else:
                    color = "#b06000"
                    phase_text = phase_names.get(phase, phase)
                    if phase == "waiting_targets":
                        outside = list(status.get("outside_region_targets", []) or [])
                        accepted = list(status.get("accepted_connector_targets", []) or [])
                        required = int(status.get("required_connector_count", 0) or 0)
                        if outside:
                            detail = (
                                f"目标：{expected_text}；画面已看到{'、'.join(outside)}，"
                                "但它尚未接触板子监测范围"
                            )
                        else:
                            detail = (
                                f"目标：{expected_text}；已纳入板子范围 "
                                f"{len(accepted)}/{required or len(status.get('expected_order', []) or [])} 根"
                            )
                    elif phase == "settling":
                        clear_frames = int(status.get("hand_clear_frames", 0) or 0)
                        clear_required = int(status.get("hand_clear_required", 0) or 0)
                        detail = (
                            f"当前：{actual_text}；等待离手确认 "
                            f"{clear_frames}/{clear_required} 帧"
                        )
                    elif phase == "confirming_order":
                        hits = int(status.get("order_hits", 0) or 0)
                        required = int(status.get("order_required", 0) or 0)
                        detail = (
                            f"当前：{actual_text}；顺序稳定确认 "
                            f"{hits}/{required} 帧"
                        )
                    else:
                        detail = f"目标：{expected_text}；当前：{actual_text}（{phase_text}）"
                html_content += (
                    "<div style='border-top:1px solid #d7deea; margin-top:6px; padding-top:5px;'>"
                    f"<b>🧪 {status.get('name', '')}　第{current}/{total}关："
                    f"{status.get('stage_name', '')}</b>"
                    f"<div style='color:{color}; font-weight:bold; margin-left:12px;'>{detail}</div>"
                    "</div>"
                )
            if slot_expectation.get("configured"):
                if slot_expectation.get("satisfied"):
                    color, label = "#188038", "终态布局正确"
                elif not slot_expectation.get("settled"):
                    color, label = "#b06000", "等待离手后稳定验收"
                else:
                    expected = slot_expectation.get("expected", {}) or {}
                    actual = slot_expectation.get("actual", {}) or {}
                    observed_wrong = any(
                        actual.get(name) not in (None, "", "空", "未知", "未知槽位")
                        and actual.get(name) != target
                        for name, target in expected.items()
                    )
                    if observed_wrong:
                        color, label = "#d93025", "；".join(
                            slot_expectation.get("mismatches", [])
                        )
                    else:
                        color, label = "#b06000", "已离手，等待所有槽位达到目标终态"
                html_content += (
                    f"<div style='color:{color}; font-weight:bold; margin-top:4px;'>"
                    f"🧩 当前步骤：{label}</div>"
                )
            html_content += "</div>"

        post_check = dict(
            getattr(self.vision_thread, "post_completion_check", {}) or {}
        )
        if post_check:
            post_step_idx = int(post_check.get("step_idx", current_idx) or 0)
            post_name = html.escape(str(post_check.get("name") or "附加条件验收"))
            post_message = html.escape(str(
                post_check.get("message") or "正在检查最终状态"
            ))
            post_progress = max(0, min(100, int(post_check.get("progress", 0) or 0)))
            html_content += (
                "<div style='background:#f5efff; border:2px solid #7e57c2; "
                "border-radius:9px; padding:10px 12px; margin-bottom:10px;'>"
                "<div style='color:#5e35b1; font-weight:bold; font-size:18px;'>"
                f"🧪 独立附加条件验收　步骤 {post_step_idx + 1}</div>"
                "<div style='color:#188038; font-weight:bold;'>"
                "✅ 原工序动作已完成：100%</div>"
                f"<div style='margin-top:4px;'><b>{post_name}</b></div>"
                f"<div style='color:#5f6368;'>{post_message}</div>"
                f"<div style='color:#5e35b1; font-weight:bold;'>验收进度：{post_progress}%</div>"
                "</div>"
            )

        if alert_msg and "AOI" not in alert_msg:
            if any(keyword in alert_msg for keyword in (
                "违禁", "中断", "失败", "状态违规", "布局错误", "禁止触碰",
                "接线顺序错误", "接线结果错误"
            )):
                html_content += f"<div style='background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold;'>⚠️ {alert_msg}</div>"
            else:
                html_content += f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #ffeeba;'>{alert_msg}</div>"

        if not process_steps:
            if sequence_statuses:
                html_content += "未配置普通动作工序，正在独立运行接线结果关卡。"
            else:
                html_content += "未配置任何工序步骤，仅开启自由过滤检测。"
        else:
            if current_idx >= len(process_steps):
                if alert_msg and "AOI" in alert_msg and "通过" in alert_msg:
                    html_content += "<div style='color: #155724; font-weight:bold; font-size:18px; background-color: #d4edda; padding: 10px; border-radius: 8px; border: 2px solid #28a745;'>🎉 所有工序已完成！<br><span style='color: #28a745;'>🔬 {}</span><br><span style='font-size:14px;'>即将自动开始下一轮...</span></div>".format(alert_msg)
                else:
                    html_content += "<div style='color: #155724; font-weight:bold; font-size:18px;'>🎉 当前产品工序已完成，即将开始下一轮！</div>"

            for i, step_dict in enumerate(process_steps):
                step_text = step_dict.get("text", "")
                req_count = step_dict.get("count", 1)
                runtime_status = step_dict.get("_runtime_status", "pending")
                original_step_num = step_dict.get("_display_step_num", i + 1)
                original_idx = original_step_num - 1
                row_progress = int(step_dict.get("_runtime_progress", 0) or 0)
                jump_shadow_progress = int(
                    step_dict.get("_jump_shadow_progress", 0) or 0
                )
                group_name = step_dict.get("_unordered_group", "")
                aoi_state = step_dict.get("_aoi_state")
                aoi_is_active = aoi_state in ("finding_anchor", "checking", "blocked")
                remediation_active = bool(step_dict.get("_remediation_active"))
                suppress_unordered_runtime = bool(step_dict.get("_suppress_unordered_runtime"))
                is_current_runtime_step = (
                    original_idx == current_idx and not suppress_unordered_runtime
                )

                if step_dict.get("_group_open"):
                    group_size = step_dict.get("_group_size", 0)
                    html_content += (
                        "<div style='border:2px dashed #1a73e8; background:#eef5ff; "
                        "border-radius:8px; padding:7px 9px; margin:7px 0;'>"
                        f"<div style='color:#1a73e8; font-weight:bold; font-size:14px; margin-bottom:4px;'>"
                        f"🔀 可乱序组 {group_name}，共 {group_size} 步，完成后自动按实际顺序排列</div>"
                    )

                if runtime_status == "skipped" and not remediation_active:
                    detach_hint = detach_hint_html(step_dict, original_idx)
                    html_content += f"<div style='color: #dc3545;'><b>⚠️ 步骤 {original_step_num}:</b> {step_text} <i>[已跳过]</i>{detach_hint}</div>"
                elif runtime_status in ("completed", "remedied") and not aoi_is_active:
                    done_order = step_dict.get("_unordered_done_order")
                    order_hint = f" <span style='color:#1a73e8;'>[组内第 {done_order} 个完成]</span>" if done_order else ""
                    done_label = "已补救" if runtime_status == "remedied" else "已完成"
                    detach_hint = detach_hint_html(step_dict, original_idx)
                    aoi_hint = ""
                    aoi_sim = step_dict.get("_aoi_similarity")
                    aoi_threshold = step_dict.get("_aoi_threshold")
                    if aoi_sim is not None and aoi_threshold is not None:
                        if aoi_state == "passed":
                            aoi_hint = f" <span style='background-color:#28a745; color:white; padding:2px 8px; border-radius:4px; font-size: 14px;'>🔬 AOI 通过 {float(aoi_sim):.1%}/{float(aoi_threshold):.0%}</span>"
                        elif aoi_state == "forced":
                            aoi_hint = f" <span style='background-color:#ff8c00; color:white; padding:2px 8px; border-radius:4px; font-size: 14px;'>🔬 AOI 人工放行 {float(aoi_sim):.1%}/{float(aoi_threshold):.0%}</span>"
                        elif aoi_state:
                            aoi_hint = f" <span style='background-color:#17a2b8; color:white; padding:2px 8px; border-radius:4px; font-size: 14px;'>🔬 AOI {float(aoi_sim):.1%}/{float(aoi_threshold):.0%}</span>"
                    html_content += f"<div style='color: #28a745;'><b>✅ 步骤 {original_step_num}:</b> {step_text} <i>[{done_label}]</i>{order_hint}{aoi_hint}{detach_hint}</div>"
                elif (is_current_runtime_step or row_progress > 0 or jump_shadow_progress > 0 or aoi_is_active
                      or remediation_active or step_dict.get("_unordered_active")):
                    count_str = (
                        f" <b>[{sub_count}/{req_count}次]</b>"
                        if req_count > 1 and is_current_runtime_step else ""
                    )

                    if is_pausing and is_current_runtime_step:
                        detach_hint = detach_hint_html(step_dict, original_idx)
                        html_content += f"<div style='color: #155724; background-color: #d4edda; padding: 3px; border-radius: 5px;'><b>✅ 步骤 {original_step_num}:</b> {step_text}{count_str} <i>(结果确认中...)</i>{detach_hint}</div>"
                    else:
                        shown_progress = (
                            jump_shadow_progress
                            if jump_shadow_progress > 0
                            else row_progress if row_progress > 0
                            else progress if is_current_runtime_step else 0
                        )
                        prog_text = (
                            f" [疑似跳步 {shown_progress}%]"
                            if jump_shadow_progress > 0
                            else f" [{shown_progress}%]" if shown_progress > 0 else ""
                        )

                        # 跳步警报提示
                        jump_hint = ""
                        if jump_shadow_progress > 0:
                            jump_hint = " <span style='background-color:#ff8c00; color:white; padding:2px 6px; border-radius:4px; font-size: 16px;'>👀 跳步监测中</span>"
                        elif is_current_runtime_step and alert_msg and "跳步" in alert_msg:
                            jump_hint = " <span style='background-color:#d93025; color:white; padding:2px 6px; border-radius:4px; font-size: 16px;'>⚠️ 请先完成该步骤</span>"

                        detach_hint = detach_hint_html(step_dict, original_idx)
                        action_hint = action_hint_html(step_dict)
                        release_hint = release_hint_html(step_dict)
                        prereq_hint = prereq_hint_html(step_dict)

                        # AOI 状态提示：在步骤旁边显示 AOI 比对实时状态
                        aoi_hint = ""
                        aoi_sim = step_dict.get("_aoi_similarity")
                        aoi_threshold = step_dict.get("_aoi_threshold")
                        if aoi_sim is not None and aoi_threshold is not None:
                            state_text = {
                                "finding_anchor": "寻找锚定物",
                                "checking": "比对中",
                                "blocked": "未通过",
                                "passed": "通过",
                                "forced": "人工放行",
                            }.get(aoi_state, "比对")
                            bg = "#dc3545" if aoi_state == "blocked" else ("#28a745" if aoi_state == "passed" else "#17a2b8")
                            aoi_hint = (
                                f" <span style='background-color:{bg}; color:white; padding:2px 8px; "
                                f"border-radius:4px; font-size: 15px;'>🔬 AOI {state_text} "
                                f"{float(aoi_sim):.1%}/{float(aoi_threshold):.0%}</span>"
                            )
                        if remediation_active:
                            active_color = "#ff8c00"
                            active_icon = "🛠️"
                            jump_hint = " <span style='background-color:#ff8c00; color:white; padding:2px 6px; border-radius:4px; font-size: 16px;'>补救中</span>"
                        elif aoi_is_active:
                            active_color = "#dc3545" if aoi_state == "blocked" else "#1a73e8"
                            active_icon = "🔬"
                        elif jump_shadow_progress > 0:
                            active_color = "#ff8c00"
                            active_icon = "👀"
                        else:
                            active_color = "#dc3545" if (is_current_runtime_step or shown_progress > 0) else "#6c757d"
                            active_icon = "⏳" if (is_current_runtime_step or shown_progress > 0) else "⚪"
                        html_content += f"<div style='color: {active_color}; font-size: 18px; font-weight: bold;'>{active_icon} 步骤 {original_step_num}: {step_text}{count_str}{prog_text}{jump_hint}{aoi_hint}{prereq_hint}{action_hint}{release_hint}{detach_hint}</div>"
                else:
                    req_str = f" <i>[需执行 {req_count} 次]</i>" if req_count > 1 else ""
                    html_content += f"<div style='color: #6c757d;'>⚪ 步骤 {original_step_num}: {step_text}{req_str}</div>"

                if step_dict.get("_group_close"):
                    html_content += "</div>"
        html_content += "</div>"
        if self.status_banner.toHtml() != html_content:
            scrollbar = self.status_banner.verticalScrollBar()
            old_scroll_value = scrollbar.value()
            old_scroll_max = scrollbar.maximum()
            user_in_middle = old_scroll_max > 0 and old_scroll_value < old_scroll_max - 8
            step_changed = current_idx != getattr(self, "_status_last_current_idx", None)
            self.status_banner.setHtml(html_content)
            scrollbar = self.status_banner.verticalScrollBar()
            if user_in_middle:
                scrollbar.setValue(min(old_scroll_value, scrollbar.maximum()))
            elif step_changed and current_idx > 3:
                scrollbar.setValue(scrollbar.maximum())
            elif not step_changed:
                scrollbar.setValue(min(old_scroll_value, scrollbar.maximum()))
            self._status_last_current_idx = current_idx
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))
    # 🌟 拦截关闭事件，弹窗并记录日志
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出系统',
                                     '系统即将关闭。是否需要保存本次运行的数据记录？',
                                     QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                     QMessageBox.Yes)
        if reply == QMessageBox.Cancel:
            event.ignore()
            return

        # 确认退出后再关闭录制子窗口；如果用户拒绝停止正在录制的视频，则取消退出
        if hasattr(self, '_rec_dlg') and self._rec_dlg is not None:
            if self._rec_dlg.isVisible():
                self._rec_dlg.close()
                if self._rec_dlg.isVisible():
                    event.ignore()
                    return
            self._rec_dlg = None

        if reply == QMessageBox.Yes:
            # 👇 核心修复：在关闭系统前，如果当前产品做了一半，强制结算归档！
            # 这样未完成的步骤会被标记为“跳过”，产品被盖章 NG 并存入 JSON，日志和数据彻底同步。
            if self.vision_thread.ng_tracker.current_product:
                if self.vision_thread.ng_tracker.has_product_activity(min_elapsed_sec=5):
                    self.vision_thread.ng_tracker.finalize_as_ng('软件关闭时产品未完成')
                else:
                    self.vision_thread.ng_tracker.current_product = None

            end_time = datetime.now()
            date_str = end_time.strftime("%Y-%m-%d")
            start_str = self.app_start_time.strftime("%H:%M:%S")
            end_str = end_time.strftime("%H:%M:%S")

            profile_name = self.vision_thread.ng_tracker.active_profile
            counts = self._session_delta_for_profile(profile_name)

            log_content = (
                f"日期: {date_str} | 方案: {profile_name} | 开始: {start_str} | 结束: {end_str} | "
                f"本次OK: {counts['ok']} 件 | 本次NG: {counts['ng']} 件 | "
                f"累计OK: {counts['total_ok']} 件 | 累计NG: {counts['total_ng']} 件\n"
            )
            try:
                # 🌟 保存工作记录和 NG 数据
                os.makedirs("logs", exist_ok=True)
                log_path = os.path.join(os.path.join(base_path, "logs"), "work_history.log")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_content)

                # 🌟 保存 NG 产品追踪数据
                ng_path = os.path.join(os.path.join(base_path, "logs"), "ng_records.json")
                self.vision_thread.ng_tracker.save(ng_path)
                QMessageBox.information(self, "已保存", f"工作记录及 NG 数据已保存到 logs/ 中！")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"日志写入失败: {e}")

        self.vision_thread.stop()
        self.vision_thread.alarm_light.stop()
        event.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainTesterApp()
    window.show()
    sys.exit(app.exec())





