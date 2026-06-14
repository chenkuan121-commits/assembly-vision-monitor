import sys
import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import json
import cv2
from workflow_monitor import WorkflowMonitor  # 🌟 新增导入
from ng_tracker import NGProductTracker   # 🌟 NG 产品追踪
import shutil
from intent_engine import IntentEngine
import time
import math

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
from alarm_light import AlarmLightController


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
        self.workflow_monitor = WorkflowMonitor(self.engine)  # 🌟 挂载全局监控器
        self.ng_tracker = NGProductTracker()  # 🌟 NG 产品追踪器
        self.remediation_engines = {}  # 跳步补救引擎: step_idx -> ProcessLogicEngine
        self.unordered_step_engines = {}
        self.step_progress_by_idx = {}
        self.unordered_completion_order = {}
        self.unordered_completion_seq = 0

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
        self.current_step_idx = 0
        self.current_sub_count = 0
        self.completed_cycles = 0
        self.step_start_time = 0
        self.final_sub_count = 0  # 最后一步累计完成了几次
        self.final_is_pausing = False  # 最后一步单次完成后的冷却判定状态
        self.final_pause_start_time = 0  # 冷却计时器
        self.final_last_action_time = 0  # 上一次拧螺丝的时间（用于防误触清零）
        self.step_timeout = process_editor.DEFAULT_STEP_TIMEOUT
        self.current_conf = 0.80  # YOLO 置信度阈值 (动态可调)
        self.is_pausing = False
        self.pause_start_time = 0
        self.cycle_pausing = False
        self.use_chinese_labels = True  # 🌟 默认启用中文渲染，避免 OpenCV 显示乱码
        self.MAX_LOST_FRAMES = 3000
        # 🌟 性能优化：PIL 字体只加载一次，不每帧重复 IO
        self.font_label = None
        self.font_hand = None
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
        self.capture_width = 1280
        self.capture_height = 720
        self.yolo_imgsz = None  # None = 使用模型原生分辨率
        self.record_width = 1920
        self.record_height = 1080
        self.frame_transform = "none"
        self.alarm_light = AlarmLightController()
        self._alarm_forbidden_active = False
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

    def _reset_aoi_runtime(self):
        self.aoi_state = None
        self.aoi_step_idx = None
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
        self._set_forbidden_alarm(False)
        self._alarm_aoi_blocked_active = False

    def _reset_workflow_runtime(self):
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
        self.remediation_engines.clear()
        self.unordered_step_engines.clear()
        self.step_progress_by_idx.clear()
        self.unordered_completion_order.clear()
        self.unordered_completion_seq = 0
        self.workflow_monitor.shadow_engines.clear()
        self.workflow_monitor.alarm_message = ""
        self.workflow_monitor.alarm_expiry_time = 0.0
        if hasattr(self.intent_engine, "reset_runtime_state"):
            self.intent_engine.reset_runtime_state()
        self._reset_aoi_runtime()
        self._alarm_aoi_blocked_active = False
        self.step_start_time = time.time()

    def _set_forbidden_alarm(self, active):
        active = bool(active)
        if self._alarm_forbidden_active == active:
            return
        self._alarm_forbidden_active = active
        self.alarm_light.set_forbidden_alarm(active)

    def _flash_red_alarm(self, key):
        self.alarm_light.flash_red(key=key, times=3, interval=0.3)

    def prepare_for_new_stream(self, interrupted_reason='切换视频源时产品未完成'):
        """开启新相机/视频前清理旧运行态，避免上一轮按钮指令串到下一轮。"""
        if self.ng_tracker.current_product:
            if self.ng_tracker.has_product_activity(min_elapsed_sec=5):
                self.ng_tracker.finalize_as_ng(interrupted_reason)
            else:
                self.ng_tracker.current_product = None
        self._reset_transient_requests()
        self._reset_workflow_runtime()

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

    def load_config(self, config_data):
        mapping = config_data.get("mapping", {})
        old_steps = self.process_steps
        self.process_steps = config_data.get("process_steps", [])
        self.step_timeout = config_data.get("step_timeout", process_editor.DEFAULT_STEP_TIMEOUT)

        # 缓存类名映射，避免每帧访问 self.model.names（ONNX 模型会触发 CUDA 设备检测导致崩溃）
        self.model_names = {}
        for class_id_str, info in mapping.items():
            self.model_names[int(class_id_str)] = info.get("eng_name", "")
        self.name_to_id_cache = {v: k for k, v in self.model_names.items() if v}

        self.engine.build_parser(mapping)
        self.engine.reset()
        # 👇 新增：同步给末步引擎
        self.final_step_engine.build_parser(mapping)
        self.final_step_engine.reset()
        # 同步所有补救引擎的字典映射（引擎实例按需创建）
        for eng in self.remediation_engines.values():
            eng.build_parser(mapping)
        self.remediation_engines.clear()
        self.unordered_step_engines.clear()

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
        # 切换配置只清掉进行中的产品；真正开始计件放到视频流启动后处理
        if old_steps != self.process_steps:
            self.ng_tracker.current_product = None
        self._reset_workflow_runtime()

    def _finish_and_restart_cycle(self):
        """正常完成当前产品并自动开始下一个产品循环"""
        if self.ng_tracker.current_product:
            self.ng_tracker._finalize_product()
        if self.process_steps:
            self.ng_tracker.start_product(self.process_steps)
        self._reset_workflow_runtime()

    def _step_order_group(self, step_dict):
        return str(step_dict.get("order_group", "")).strip()

    def _is_hand_touch_step(self, step_dict):
        return step_dict.get("action_type") == "hand_touch"

    def _is_detach_step(self, step_dict):
        return step_dict.get("action_type") == "detach"

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

    def _target_options(self, target):
        return self.engine.target_options(target)

    def _target_display_name(self, target):
        return self.engine.target_display_name(target, self.engine.eng_to_zh)

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

    def _evaluate_step_by_config(self, step_idx, step_dict, targets, detections, engine):
        difficulty = step_dict.get("difficulty", "中等")
        if self._is_hand_touch_step(step_dict):
            return engine.evaluate_hand_touch_step(targets, detections, difficulty)
        if self._is_detach_step(step_dict):
            return engine.evaluate_detach_step(targets, detections, difficulty)
        return engine.evaluate_step(
            targets,
            detections,
            difficulty,
            step_dict.get("count", 1),
            step_dict.get("multi_strategy", "lock")
        )

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

    def _start_aoi_check(self, step_idx, aoi_cfg):
        self.aoi_state = 'finding_anchor'
        self.aoi_step_idx = step_idx
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

    def _sync_current_idx_for_unordered_group(self, group_indices):
        pending = [i for i in group_indices if self._step_record_status(i) == 'pending']
        if pending:
            self.current_step_idx = pending[0]
            self.step_start_time = time.time()
            return False
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

        best_progress = 0
        completed_any = False
        for idx in group_indices:
            if self._step_record_status(idx) != 'pending':
                continue
            step_dict = self.process_steps[idx]
            targets = self._targets_for_step(step_dict)
            if not targets:
                continue
            if idx not in self.unordered_step_engines:
                eng = ProcessLogicEngine()
                eng.lookup_dict = self.engine.lookup_dict
                eng.eng_to_zh = self.engine.eng_to_zh
                eng.regex_pattern = self.engine.regex_pattern
                self.unordered_step_engines[idx] = eng
            is_done, step_progress = self._evaluate_step_by_config(
                idx, step_dict, targets, detections, self.unordered_step_engines[idx]
            )
            self.step_progress_by_idx[idx] = step_progress
            best_progress = max(best_progress, step_progress)
            if is_done:
                self.step_progress_by_idx[idx] = 100
                self.unordered_step_engines[idx].reset()
                aoi_cfg = step_dict.get('aoi_feature_check', {})
                if aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
                    self._start_aoi_check(idx, aoi_cfg)
                    return False, 100, f"步骤 {idx + 1} 动作完成，正在进入 AOI 比对"

                self.ng_tracker.mark_step_completed(idx)
                self.ng_tracker._check_and_restore_ok()
                self._remember_unordered_completion(idx)
                completed_any = True

        finished_group = False
        if completed_any:
            finished_group = self._sync_current_idx_for_unordered_group(group_indices)
            if finished_group:
                return True, 100, "乱序组已完成"

        pending_count = sum(1 for i in group_indices if self._step_record_status(i) == 'pending')
        alert = ""
        if pending_count > 0:
            alert = f"乱序组待完成: {pending_count}/{len(group_indices)}"
        return finished_group, best_progress, alert

    def _display_process_steps(self):
        cp = self.ng_tracker.current_product
        if not cp:
            return self.process_steps
        records = cp.get('step_records', [])

        def make_step(idx):
            step = self.process_steps[idx]
            step_copy = dict(step)
            step_copy['_display_step_num'] = idx + 1
            step_copy['_runtime_progress'] = int(self.step_progress_by_idx.get(idx, 0))
            group = self._step_order_group(step_copy)
            if group:
                step_copy['_unordered_group'] = group
                if idx in self.unordered_completion_order:
                    step_copy['_unordered_done_order'] = self.unordered_completion_order[idx]
            if idx < len(records):
                step_copy['_runtime_status'] = records[idx].get('status', 'pending')
                for key in ('aoi_similarity', 'aoi_threshold', 'aoi_state', 'aoi_best_angle'):
                    if key in records[idx]:
                        step_copy[f'_{key}'] = records[idx].get(key)
            if self.aoi_step_idx == idx and self.aoi_state:
                step_copy['_aoi_state'] = self.aoi_state
                step_copy['_aoi_similarity'] = float(self.aoi_similarity)
                step_copy['_aoi_threshold'] = float(self.aoi_threshold)
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
            # --- 接收 UI 层的干预指令 ---
            if self.reset_signal:
                if self.ng_tracker.current_product:
                    self.ng_tracker.finalize_as_ng('手动重新开始，当前产品未完成')
                self._reset_workflow_runtime()
                self.reset_signal = False
                # 启动新产品追踪
                if self.process_steps:
                    self.ng_tracker.start_product(self.process_steps)
            if self.force_skip_signal:
                if self.process_steps and self.current_step_idx < len(self.process_steps):
                    old_idx = self.current_step_idx
                    self.current_step_idx += 1
                    self.step_progress_by_idx[old_idx] = 0
                    # 🌟 NG 追踪：手动跳过 → 记录 NG
                    self.ng_tracker.on_step_advance(old_idx, '手动跳过')
                    self.current_sub_count = 0
                    self.engine.reset()
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
                # 2. 提取当前目标
                if self.current_step_idx < len(self.process_steps):
                    current_step_dict = self.process_steps[self.current_step_idx]
                    required_targets = self._targets_for_step(current_step_dict)
                # 3. 将所有涉及的 ID 添加进 YOLO 白名单
                for t in all_required_targets:
                    for option in self._target_options(t):
                        if option in name_to_id and name_to_id[option] not in active_class_ids:
                            active_class_ids.append(name_to_id[option])
            # --- YOLO 推理 ---
            detections = []
            annotated_frame = frame.copy()
            if self.model is not None and len(active_class_ids) > 0:
                infer_kwargs = dict(
                    classes=active_class_ids,
                    verbose=False,
                    conf=self.current_conf,
                    device=self.infer_device,
                )
                if self.yolo_imgsz is not None:
                    infer_kwargs['imgsz'] = self.yolo_imgsz
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
                if results and len(results) > 0:
                    names = self.model_names
                    boxes_to_draw = []
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
                            detections.append({'class': cls_name, 'bbox': draw_bbox})
                            # 收集给前端画图用的数据（严格遵循勾选框，手套除外）
                            if cls_name in ui_checked_names and cls_name not in ['glove', '手套']:
                                boxes_to_draw.append((draw_bbox, cls_name, conf, 'aabb'))
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
                            detections.append({'class': cls_name, 'bbox': draw_bbox})

                            if cls_name in ui_checked_names and cls_name not in ['glove', '手套']:
                                boxes_to_draw.append((draw_points, cls_name, conf, 'obb'))

                    # 夺回 UI 控制权：彻底抛弃 YOLO 自带的 .plot()
                    annotated_frame = frame.copy()

                    if not self.use_chinese_labels:
                        # 【路线 A：英文原版】用 OpenCV 画干净的框
                        for box_data, eng_name, conf, box_type in boxes_to_draw:
                            if box_type == 'aabb':
                                x1, y1, x2, y2 = map(int, box_data)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                text_x, text_y = x1, max(0, y1 - 10)
                            else:
                                # 🌟 画斜框 (多边形)
                                cv2.polylines(annotated_frame, [box_data], isClosed=True, color=(0, 255, 0),
                                              thickness=2)
                                # 找最上面的一个点作为文字的基准点
                                top_pt = min(box_data, key=lambda p: p[1])
                                text_x, text_y = top_pt[0], max(0, top_pt[1] - 10)

                            cv2.putText(annotated_frame, f"{eng_name} {conf:.2f}", (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # 【路线 B：中文特供版】Pillow 介入
                        if boxes_to_draw:
                            cv2_im_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            pil_im = Image.fromarray(cv2_im_rgb)
                            draw = ImageDraw.Draw(pil_im)
                            font = self.font_label
                            for box_data, eng_name, conf, box_type in boxes_to_draw:
                                zh_name = self.engine.eng_to_zh.get(eng_name, eng_name)
                                display_text = f"{zh_name} {conf:.2f}"

                                if box_type == 'aabb':
                                    x1, y1, x2, y2 = map(int, box_data)
                                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                                    text_x, text_y = x1, max(0, y1 - 45)
                                else:
                                    # 🌟 画斜框 (Pillow 多边形)
                                    poly_pts = [tuple(pt) for pt in box_data]
                                    draw.polygon(poly_pts, outline=(0, 255, 0), width=3)
                                    top_pt = min(poly_pts, key=lambda p: p[1])
                                    text_x, text_y = top_pt[0], max(0, top_pt[1] - 45)

                                try:
                                    text_bbox = draw.textbbox((text_x, text_y), display_text, font=font)
                                    draw.rectangle(text_bbox, fill=(255, 255, 255))
                                except AttributeError:
                                    draw.rectangle([text_x, text_y, text_x + 180, text_y + 45], fill=(255, 255, 255))
                                draw.text((text_x, text_y), display_text, font=font, fill=(255, 0, 0))

                            annotated_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
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
                    self.ng_tracker.mark_step_completed(old_idx)
                    cp = self.ng_tracker.current_product
                    if cp and old_idx < len(cp['step_records']):
                        cp['step_records'][old_idx]['aoi_forced'] = True
                        cp['step_records'][old_idx]['aoi_note'] = f'AOI特征比对失败_人工强制放行 (相似度: {self.aoi_similarity:.3f})'
                        cp['step_records'][old_idx]['aoi_similarity'] = float(self.aoi_similarity)
                        cp['step_records'][old_idx]['aoi_threshold'] = float(self.aoi_threshold)
                        cp['step_records'][old_idx]['aoi_state'] = 'forced'
                        cp['ng_reason'] = f'工序{old_idx+1} AOI特征比对未通过(人工放行)'
                    if self._step_order_group(self.process_steps[old_idx]):
                        self._remember_unordered_completion(old_idx)
                        self._sync_current_idx_for_unordered_group(self._unordered_group_indices(old_idx))
                    else:
                        self.current_step_idx = old_idx + 1
                    self.current_sub_count = 0
                    self.aoi_state = None
                    self.aoi_step_idx = None
                    self._alarm_aoi_blocked_active = False
                    self.aoi_stable_count = 0
                    self.is_pausing = False
                    self.engine.reset()
                    self.step_start_time = time.time()
                    alert_msg = "⚠️ AOI 特征比对人工强制放行！"
                    self.aoi_update_signal.emit(0.0, '', False)
                    # 最后一步放行后自动进入下一轮
                    if self.current_step_idx >= len(self.process_steps):
                        self._finish_and_restart_cycle()
                        alert_msg = "⚠️ AOI 强制放行，当前产品已完成，进入下一轮！"
                self.aoi_force_signal = False

            # 0. 跳步补救检测：检查工人是否返回完成了之前跳过的步骤
            if self.process_steps and self.current_step_idx > 0:
                skipped_indices = self.ng_tracker.get_skipped_indices()
                for si in skipped_indices:
                    if si >= self.current_step_idx:
                        continue
                    si_dict = self.process_steps[si]
                    si_targets = self._targets_for_step(si_dict)
                    if not si_targets:
                        continue
                    # 快速筛：目标起码要出现在画面里
                    if not self.engine.check_presence(si_targets, detections):
                        # 目标不在画面，衰减对应引擎的计数器（防止残影误触发）
                        if si in self.remediation_engines:
                            self.remediation_engines[si].hit_counter = max(0, self.remediation_engines[si].hit_counter - 1)
                        continue
                    # 创建或复用专属补救引擎
                    if si not in self.remediation_engines:
                        eng = ProcessLogicEngine()
                        eng.lookup_dict = self.engine.lookup_dict
                        eng.eng_to_zh = self.engine.eng_to_zh
                        eng.regex_pattern = self.engine.regex_pattern
                        self.remediation_engines[si] = eng
                    is_remedied, _ = self._evaluate_step_by_config(
                        si, si_dict, si_targets, detections, self.remediation_engines[si]
                    )
                    if is_remedied:
                        self.ng_tracker.mark_step_completed(si)
                        self.ng_tracker._check_and_restore_ok()
                        self.remediation_engines[si].reset()
                        zh_names = [self._target_display_name(t) for t in si_targets]
                        alert_msg = f"🔄 工人返回完成了步骤 {si+1}【{' '.join(zh_names)}】，状态已补救！"
                        break  # 一次只处理一个补救，避免 alert_msg 被覆盖

            # 1. 违禁检查
            has_forbidden, fb_class = self.engine.check_forbidden(detections, self.forbidden_targets)
            if has_forbidden:
                self._set_forbidden_alarm(True)
                zh_name = self.engine.eng_to_zh.get(fb_class, fb_class)
                alert_msg = f"画面中检测到违禁项：【{zh_name}】！"
            else:
                self._set_forbidden_alarm(False)
                # 2. 超时监控 (🌟 动态读取 step_timeout)
                if self.process_steps and self.current_step_idx < len(self.process_steps) and not self.is_pausing:
                    if time.time() - self.step_start_time > self.step_timeout:
                        alert_msg = f"⏱️ 当前步骤耗时过长 (超{self.step_timeout}秒)，请检查或强制跳过！"
                        self.ng_tracker.mark_step_timeout(self.current_step_idx, self.step_timeout)

            # 3. 工序流转与循环
            force_end_product = False
            # 如果当前还没做到最后一步，我们就偷偷在后台算最后一步的进度
            if self.process_steps and self.current_step_idx < len(self.process_steps) - 1:
                last_step_dict = self.process_steps[-1]
                last_targets = self._targets_for_step(last_step_dict)

                if last_targets:
                    last_count = last_step_dict.get("count", 1)  # 👈 这里会拿到最后一步要求拧 3 次

                    # 用专属引擎去算最后一步是不是被做完了（单次判定）
                    is_last_done, _ = self._evaluate_step_by_config(
                        len(self.process_steps) - 1, last_step_dict, last_targets, detections, self.final_step_engine
                    )

                    # 状态机 1：如果单次做完了，进入短暂的“确认暂停”状态
                    if is_last_done and not self.final_is_pausing:
                        self.final_is_pausing = True
                        self.final_pause_start_time = time.time()

                    # 状态机 2：暂停 1.5 秒后，正式算作最后一步的动作完成了 1 次
                    if self.final_is_pausing:
                        if time.time() - self.final_pause_start_time > 1.5:
                            self.final_is_pausing = False
                            self.final_sub_count += 1
                            self.final_last_action_time = time.time()
                            self.final_step_engine.reset()  # 清空动作记忆，准备抓下一个螺丝

                            # 核心判定：只有子次数真正达标（比如达到 3 次），才触发强制收卷！
                            if self.final_sub_count >= last_count:
                                force_end_product = True
                                self.final_sub_count = 0  # 触发后清零

                    # 防误触清零机制：如果你做第二步拧了 1 个螺丝，然后去做别的了
                    # 超过 10 秒没连续拧螺丝，后台就把次数清零，防止和后续步骤累加导致误判！
                    if self.final_sub_count > 0 and not self.final_is_pausing:
                        if time.time() - self.final_last_action_time > 10.0:
                            self.final_sub_count = 0

            # ==============================================================
            # 3. 工序流转与循环
            # ==============================================================
            if not has_forbidden:
                if force_end_product:
                    # 💥 触发强行交卷！
                    self._flash_red_alarm("force_end")
                    self.ng_tracker.add_jump_alarm(self.current_step_idx, "强制收卷：检测到工人直接完成了最后一步")
                    self._finish_and_restart_cycle()
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
                                    self._record_aoi_status(passed_idx, 'passed', sim, threshold, best_angle)
                                    self.step_progress_by_idx[passed_idx] = 100
                                    is_unordered_aoi = bool(self._step_order_group(self.process_steps[passed_idx]))
                                    self.ng_tracker.mark_step_completed(passed_idx)
                                    if is_unordered_aoi:
                                        self._remember_unordered_completion(passed_idx)
                                        group_indices = self._unordered_group_indices(passed_idx)
                                        finished_group = self._sync_current_idx_for_unordered_group(group_indices)
                                    else:
                                        self.current_step_idx = passed_idx + 1
                                        finished_group = False
                                    self.current_sub_count = 0
                                    self.aoi_state = None
                                    self.aoi_step_idx = None
                                    self._alarm_aoi_blocked_active = False
                                    self.aoi_stable_count = 0
                                    self.aoi_pass_flash = 30   # 绿色边框闪烁 1 秒
                                    self.aoi_pass_msg_frames = 90  # 消息保持 3 秒
                                    self.is_pausing = False
                                    self.engine.reset()
                                    self.step_start_time = time.time()
                                    self.aoi_update_signal.emit(sim, '', False)
                                    if was_blocked:
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
                                    if self.current_step_idx >= len(self.process_steps):
                                        self.aoi_pending_restart = True
                                        alert_msg = f"✅ AOI 特征比对通过! 所有步骤已完成，即将进入下一轮！"
                                    elif finished_group:
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
                        group_indices = self._unordered_group_indices(self.current_step_idx)
                        if group_indices:
                            finished_group, group_progress, group_alert = self._evaluate_unordered_group(detections)
                            progress = group_progress
                            if group_alert and not alert_msg:
                                alert_msg = group_alert
                            if finished_group:
                                self.current_sub_count = 0
                                self.engine.reset()
                                self.step_start_time = time.time()
                        else:
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

                            # 3. 状态机：暂停 1.5 秒后，正式进入下一步
                            if self.is_pausing:
                                if time.time() - self.pause_start_time > 1.5:
                                    self.is_pausing = False
                                    self.current_sub_count += 1
                                    self.engine.reset()  # 清空动作引擎记忆

                                    # 检查子次数是否全部达标
                                    if self.current_sub_count >= req_count:
                                        step_dict = self.process_steps[self.current_step_idx]
                                        aoi_cfg = step_dict.get('aoi_feature_check', {})
                                        if aoi_cfg.get('enabled', False) and self.aoi_extractor is not None:
                                            # 进入 AOI 特征比对状态，不立即推进步骤
                                            self._start_aoi_check(self.current_step_idx, aoi_cfg)
                                        else:
                                            self.ng_tracker.on_step_advance(self.current_step_idx, 'completed')
                                            self.step_progress_by_idx[self.current_step_idx] = 100
                                            self.current_step_idx += 1
                                            # 最后一步完成后自动进入下一轮
                                            if self.current_step_idx >= len(self.process_steps):
                                                self._finish_and_restart_cycle()
                                        self.current_sub_count = 0

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
            # 2. 🌟 启用全局装配监控：手随便摸，但违规组合绝对不行！
            is_jump, jump_msg, jumped_to_idx = self.workflow_monitor.check_jump_by_completion(
                detections, self.process_steps, self.current_step_idx, held_objs
            )

            if is_jump:
                self._flash_red_alarm(f"jump_{jumped_to_idx if jumped_to_idx >= 0 else self.current_step_idx}")
                alert_msg = jump_msg
                # 👇 新增这一行：真正把跳步行为记录进 NG 追踪器里，触发 NG 状态！
                self.ng_tracker.add_jump_alarm(self.current_step_idx, jump_msg)
                img_h, img_w = annotated_frame.shape[:2]

                # 1. 边框闪烁效果：只画一圈粗边框，绝不全屏覆盖遮挡视线
                border_color = (0, 0, 255) if fps_frame_count % 4 < 2 else (0, 128, 255)  # 红/橙交替闪烁
                cv2.rectangle(annotated_frame, (15, 15), (img_w - 15, img_h - 15), border_color, 25)

                # 2. 告警文字：因为 cv2.putText 不支持中文会变 ???，工业界一般直接用醒目的英文或拼音
                cv2.putText(annotated_frame, "WARNING: STEP JUMP!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 6, cv2.LINE_AA)

            # 🌟🌟🌟 修复：将缩进向左退一格，使其与 if is_jump 平级！
            # 3. 循环画出所有的手/手套和悬浮文字
            if hand_renders:
                # 强制走 Pillow 以确保中文不会变问号
                cv2_im_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_rgb)
                draw = ImageDraw.Draw(pil_im)
                font_hand = self.font_hand

                for render_box, hand_color, hand_text, box_type, landmarks, hand_id in hand_renders:
                    # 先画原有的外框和文字
                    if box_type == 'aabb':
                        hx1, hy1, hx2, hy2 = map(int, render_box)
                        draw.rectangle([hx1, hy1, hx2, hy2], outline=hand_color, width=4)
                        text_y = max(0, hy1 - 40)
                        draw.text((hx1, text_y), hand_text, font=font_hand, fill=hand_color)
                    else:
                        poly_pts = [tuple(pt) for pt in render_box]
                        draw.polygon(poly_pts, outline=hand_color, width=4)
                        top_pt = min(poly_pts, key=lambda p: p[1])
                        text_x, text_y = top_pt[0], max(0, top_pt[1] - 40)
                        draw.text((text_x, text_y), hand_text, font=font_hand, fill=hand_color)

                    # OpenCV 和 Pillow 转换完成后，再用 OpenCV 画骨架
                annotated_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

                # 🌟 重新遍历一次，专门画骨骼
                for _, _, _, _, landmarks, _ in hand_renders:
                    if landmarks is not None:
                        mp_drawing.draw_landmarks(
                            annotated_frame, landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
            else:
                for hand_box, hand_color, hand_text, _, _, _ in hand_renders:
                    hx1, hy1, hx2, hy2 = map(int, hand_box)
                    # OpenCV 用 BGR，需翻转 RGB 颜色防变蓝
                    cv_color = (hand_color[2], hand_color[1], hand_color[0])
                    cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), cv_color, 3)
                    cv2.putText(annotated_frame, hand_text, (hx1, max(0, hy1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, cv_color, 2)
            # ==============================================================
            # 🌟🌟🌟 新增代码结束 🌟🌟🌟
            # 🌟 在左上角绘制当前帧率
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, 50 if not is_jump else 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            frame_h, frame_w = raw_frame.shape[:2]
            cv2.putText(annotated_frame, f"RES: {frame_w}x{frame_h}", (20, 100 if not is_jump else 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
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

        self.setLayout(layout)

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
        event.accept()


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
        self.chk_chinese_label = QCheckBox("🀄 启用中文标签")
        self.chk_chinese_label.setStyleSheet("color: #d93025; font-weight: bold;")
        self.chk_chinese_label.setChecked(True)  # 默认启用，匹配 VisionThread 默认值
        self.chk_chinese_label.stateChanged.connect(self.toggle_chinese_label)
        btn_layout.addWidget(self.chk_chinese_label)
        self.chk_mediapipe = QCheckBox("🖐️ MediaPipe 手势识别")
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
        self.conf_slider.setValue(80)
        self.conf_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        self.conf_label = QLabel("0.80")
        self.conf_label.setFixedWidth(40)
        self.conf_label.setStyleSheet("font-weight: bold;")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        filter_outer_layout.addLayout(conf_layout)

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
        self.combo_capture_res.setCurrentIndex(0)  # 默认 1280×720
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
        self.combo_yolo_imgsz.addItems(["默认 (原生)", "640", "960", "1280"])
        self.combo_yolo_imgsz.setCurrentIndex(0)  # 默认使用模型原生分辨率
        self.combo_yolo_imgsz.setEditable(True)
        self.combo_yolo_imgsz.setToolTip("YOLO 推理分辨率，「默认」使用模型训练时的原生尺寸")
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

        # 5. AOI 特征建档
        aoi_group = QGroupBox("🔬 5. AOI 特征建档 (Golden Sample)")
        aoi_layout = QVBoxLayout()

        aoi_step_row = QHBoxLayout()
        aoi_step_row.addWidget(QLabel("目标步骤:"))
        self.combo_aoi_step = QComboBox()
        self.combo_aoi_step.setToolTip("选择要配置 AOI 特征比对的工序步骤")
        self.combo_aoi_step.currentIndexChanged.connect(self.on_aoi_step_selected)
        aoi_step_row.addWidget(self.combo_aoi_step)

        aoi_anchor_row = QHBoxLayout()
        aoi_anchor_row.addWidget(QLabel("锚定物类别:"))
        self.combo_aoi_anchor = QComboBox()
        self.combo_aoi_anchor.setToolTip("选择 AOI 比对的目标类别")
        aoi_anchor_row.addWidget(self.combo_aoi_anchor)
        aoi_layout.addLayout(aoi_step_row)
        aoi_layout.addLayout(aoi_anchor_row)

        aoi_thresh_row = QHBoxLayout()
        aoi_thresh_row.addWidget(QLabel("相似度阈值:"))
        self.aoi_thresh_slider = QSlider(Qt.Horizontal)
        self.aoi_thresh_slider.setRange(50, 99)
        self.aoi_thresh_slider.setValue(85)
        self.aoi_thresh_slider.setTickInterval(5)
        self.aoi_thresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.aoi_thresh_label = QLabel("0.85")
        self.aoi_thresh_label.setFixedWidth(35)
        self.aoi_thresh_label.setStyleSheet("font-weight: bold;")
        self.aoi_thresh_slider.valueChanged.connect(
            lambda v: self.aoi_thresh_label.setText(f"{v/100:.2f}")
        )
        aoi_thresh_row.addWidget(self.aoi_thresh_slider)
        aoi_thresh_row.addWidget(self.aoi_thresh_label)
        aoi_layout.addLayout(aoi_thresh_row)

        aoi_btn_row = QHBoxLayout()
        self.btn_aoi_capture = QPushButton("📷 抓拍标准样件")
        self.btn_aoi_capture.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_aoi_capture.setEnabled(False)
        self.btn_aoi_capture.clicked.connect(self.on_aoi_capture)
        self.btn_aoi_save = QPushButton("💾 保存 AOI 特征")
        self.btn_aoi_save.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold;")
        self.btn_aoi_save.setEnabled(False)
        self.btn_aoi_save.clicked.connect(self.on_aoi_save_feature)
        self.btn_aoi_archive = QPushButton("🧪 独立建档")
        self.btn_aoi_archive.setStyleSheet("background-color: #ff8c00; color: white; font-weight: bold;")
        self.btn_aoi_archive.clicked.connect(self.open_aoi_archive_dialog)
        aoi_btn_row.addWidget(self.btn_aoi_capture)
        aoi_btn_row.addWidget(self.btn_aoi_save)
        aoi_btn_row.addWidget(self.btn_aoi_archive)
        aoi_layout.addLayout(aoi_btn_row)

        self.aoi_preview_label = QLabel("(抓拍后显示)")
        self.aoi_preview_label.setAlignment(Qt.AlignCenter)
        self.aoi_preview_label.setMinimumSize(260, 160)
        self.aoi_preview_label.setStyleSheet("border: 1px dashed #ccc; background-color: #f0f0f0;")
        self.aoi_preview_scroll = QScrollArea()
        self.aoi_preview_scroll.setWidget(self.aoi_preview_label)
        self.aoi_preview_scroll.setWidgetResizable(False)
        self.aoi_preview_scroll.setMinimumHeight(180)
        self.aoi_preview_scroll.setMaximumHeight(320)
        self.aoi_preview_scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; background: #f8f9fa; }")
        aoi_layout.addWidget(self.aoi_preview_scroll)

        self.aoi_status_label = QLabel("💡 使用前提：加载模型 → 开启摄像头 → 选择步骤和锚定物 → 抓拍")
        self.aoi_status_label.setStyleSheet("color: #666; font-size: 11px;")
        aoi_layout.addWidget(self.aoi_status_label)

        aoi_group.setLayout(aoi_layout)
        control_layout.addWidget(aoi_group)
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
            self.btn_aoi_capture.setEnabled(False)
            self.btn_aoi_force.setEnabled(False)
            return

        if source_type == "video":
            self.btn_cam.setText("停止视频")
            self.combo_speed.setEnabled(True)
        else:
            self.btn_cam.setText("关闭相机")
            self.combo_speed.setEnabled(False)
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

    def trigger_reset(self):
        choice = QMessageBox.question(self, '确认重新开始', '当前产品会记录为“手动重新开始，未完成”。确定要重新开始下一件吗？',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if choice == QMessageBox.StandardButton.Yes:
            self.vision_thread.reset_signal = True

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
        self.btn_aoi_save.setEnabled(False)
        self._aoi_captured_vector = None
        self._aoi_captured_crop = None
        self._aoi_captured_resolution = None
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
                    cb = QCheckBox(f"{display_text}")
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
                data["step_timeout"] = profile_data.get("step_timeout", process_editor.DEFAULT_STEP_TIMEOUT)
            else:
                # 即使没有工序，也给个空壳，保证不出错
                data["process_steps"] = []
                data["forbidden_items"] = ""
                data["step_timeout"] = process_editor.DEFAULT_STEP_TIMEOUT

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
        html_content = "<div style='line-height: 1.6; font-size: 16px; padding: 5px;'>"

        if alert_msg and "AOI" not in alert_msg:
            if "违禁" in alert_msg or "中断" in alert_msg or "失败" in alert_msg:
                html_content += f"<div style='background-color: #dc3545; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold;'>⚠️ {alert_msg}</div>"
            else:
                html_content += f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #ffeeba;'>{alert_msg}</div>"

        if not process_steps:
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
                group_name = step_dict.get("_unordered_group", "")
                aoi_state = step_dict.get("_aoi_state")
                aoi_is_active = aoi_state in ("finding_anchor", "checking", "blocked")

                if step_dict.get("_group_open"):
                    group_size = step_dict.get("_group_size", 0)
                    html_content += (
                        "<div style='border:2px dashed #1a73e8; background:#eef5ff; "
                        "border-radius:8px; padding:7px 9px; margin:7px 0;'>"
                        f"<div style='color:#1a73e8; font-weight:bold; font-size:14px; margin-bottom:4px;'>"
                        f"🔀 可乱序组 {group_name}，共 {group_size} 步，完成后自动按实际顺序排列</div>"
                    )

                if runtime_status == "skipped":
                    html_content += f"<div style='color: #dc3545;'><b>⚠️ 步骤 {original_step_num}:</b> {step_text} <i>[已跳过]</i></div>"
                elif runtime_status in ("completed", "remedied") and not aoi_is_active:
                    done_order = step_dict.get("_unordered_done_order")
                    order_hint = f" <span style='color:#1a73e8;'>[组内第 {done_order} 个完成]</span>" if done_order else ""
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
                    html_content += f"<div style='color: #28a745;'><b>✅ 步骤 {original_step_num}:</b> {step_text} <i>[已完成]</i>{order_hint}{aoi_hint}</div>"
                elif original_idx == current_idx or row_progress > 0 or aoi_is_active or (group_name and runtime_status == "pending"):
                    count_str = f" <b>[{sub_count}/{req_count}次]</b>" if req_count > 1 else ""

                    if is_pausing and original_idx == current_idx:
                        html_content += f"<div style='color: #155724; background-color: #d4edda; padding: 3px; border-radius: 5px;'><b>✅ 步骤 {original_step_num}:</b> {step_text}{count_str} <i>(结果确认中...)</i></div>"
                    else:
                        shown_progress = row_progress if row_progress > 0 else (progress if original_idx == current_idx else 0)
                        prog_text = f" [{shown_progress}%]" if shown_progress > 0 else ""

                        # 跳步警报提示
                        jump_hint = ""
                        if original_idx == current_idx and alert_msg and "跳步" in alert_msg:
                            jump_hint = " <span style='background-color:#d93025; color:white; padding:2px 6px; border-radius:4px; font-size: 16px;'>⚠️ 请先完成该步骤</span>"

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
                        if aoi_is_active:
                            active_color = "#dc3545" if aoi_state == "blocked" else "#1a73e8"
                            active_icon = "🔬"
                        else:
                            active_color = "#dc3545" if (original_idx == current_idx or shown_progress > 0) else "#6c757d"
                            active_icon = "⏳" if (original_idx == current_idx or shown_progress > 0) else "⚪"
                        html_content += f"<div style='color: {active_color}; font-size: 18px; font-weight: bold;'>{active_icon} 步骤 {original_step_num}: {step_text}{count_str}{prog_text}{jump_hint}{aoi_hint}</div>"
                else:
                    req_str = f" <i>[需执行 {req_count} 次]</i>" if req_count > 1 else ""
                    html_content += f"<div style='color: #6c757d;'>⚪ 步骤 {original_step_num}: {step_text}{req_str}</div>"

                if step_dict.get("_group_close"):
                    html_content += "</div>"
        html_content += "</div>"
        if self.status_banner.toHtml() != html_content:
            self.status_banner.setHtml(html_content)
            scrollbar = self.status_banner.verticalScrollBar()
            if current_idx > 3:
                scrollbar.setValue(scrollbar.maximum())
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





