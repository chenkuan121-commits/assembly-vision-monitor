"""
快速训练模块 — OBB 旋转框标注 + 自动划分 + 训练
"""
import os
import re
import cv2
import json
import shutil
import random
import numpy as np
from datetime import datetime
from collections import defaultdict
import math # [新增] 加在文件顶部的 import 区域
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image
try:
    import yaml as _yaml_lib
except ImportError:
    _yaml_lib = None
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False


def get_training_device_info():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.get_device_properties(0)
            return "0", torch.cuda.get_device_name(0), torch.version.cuda
    except Exception as exc:
        print(f"[VisionCodex][Device] Training CUDA check failed, using CPU: {exc}", flush=True)
    return "cpu", "CPU", None


def log_training_device(message):
    print(f"[VisionCodex][Device] {message}", flush=True)


from PySide6.QtWidgets import (
    QDialog, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QLineEdit, QSpinBox,
    QProgressBar, QTextEdit, QMessageBox, QGroupBox, QWidget,
    QSplitter, QScrollArea, QComboBox, QFileDialog, QInputDialog,
    QAbstractItemView
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QEvent
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QMouseEvent, QPolygonF, QBrush, QPainterPath, QShortcut, QKeySequence
)


# ── SAM 加载 ──
SAM_AVAILABLE = False
try:
    from ultralytics import SAM
    SAM_AVAILABLE = True
except ImportError:
    pass


def get_sam_model(base_path=""):
    if not SAM_AVAILABLE:
        return None
    candidates = [
        os.path.join(base_path, "mobile_sam.pt"),
        os.path.join(base_path, "sam_b.pt"),
        "mobile_sam.pt",
        "sam_b.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return SAM(c)
            except Exception:
                continue
    try:
        return SAM('sam_b.pt')
    except Exception:
        return None


SPECIAL_CLASS_ALIASES = {
    "手": ("hand", "手"),
    "hand": ("hand", "手"),
    "手套": ("glove", "手套"),
    "glove": ("glove", "手套"),
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


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


def normalize_training_class_name(name):
    raw = str(name).strip()
    return SPECIAL_CLASS_ALIASES.get(raw.lower(), SPECIAL_CLASS_ALIASES.get(raw, (raw, raw)))


def model_class_names(class_names):
    return [normalize_training_class_name(name)[0] for name in class_names]


def model_mapping_from_class_names(class_names):
    mapping = {}
    for i, name in enumerate(class_names):
        eng_name, zh_name = normalize_training_class_name(name)
        mapping[str(i)] = {"eng_name": eng_name, "zh_name": zh_name}
    return mapping


# ── OBB 标注画布 ──
class AnnotatorWidget(QLabel):
    """丝滑版 OBB 标注画板：拖拽画框 + 旋转把手 + 拖拽移动"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.viewport_w = 640
        self.viewport_h = 420
        self.zoom_factor = 1.0
        self._updating_canvas_size = False

        # 底层依然保持: [[x1,y1, x2,y2, x3,y3, x4,y4, cls_idx, cls_name], ...]
        self.obb_boxes = []
        self.selected_box_idx = -1
        self.hovered_box_idx = -1

        # 交互状态机
        self.action_state = "IDLE"  # IDLE, DRAWING, MOVING, ROTATING
        self.drag_start_pos = None
        self.temp_box_state = None

        self.class_names = []
        self.class_colors = {}
        self.current_class_idx = 0
        self.unassigned_class_id = -1

        self.sam_model = None
        self.base_path = ""
        self.original_bgr = None

    def set_image(self, img_bgr):
        self.original_bgr = img_bgr.copy()
        h, w, _ = img_bgr.shape
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.zoom_factor = 1.0
        self._rescale()
        self.obb_boxes = []
        self.selected_box_idx = -1
        self.action_state = "IDLE"
        self.update()

    def set_class_names(self, names):
        self.class_names = names
        self.class_colors = {}
        for i, _ in enumerate(names):
            hue = (i * 67) % 360
            self.class_colors[i] = QColor.fromHsv(hue, 200, 255)

    def _hotkey_to_class_idx(self, key):
        if Qt.Key_0 <= key <= Qt.Key_9:
            return key - Qt.Key_0
        if Qt.Key_A <= key <= Qt.Key_Z:
            return 10 + (key - Qt.Key_A)
        return None

    def _assign_selected_class(self, cls_idx):
        if self.selected_box_idx < 0 or cls_idx is None or cls_idx >= len(self.class_names):
            return False
        self.obb_boxes[self.selected_box_idx][8] = cls_idx
        self.obb_boxes[self.selected_box_idx][9] = self.class_names[cls_idx]
        self.update()
        return True

    def _rescale(self):
        if self.original_pixmap is None: return
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        if img_w <= 0 or img_h <= 0:
            return

        fit_scale = min(self.viewport_w / img_w, self.viewport_h / img_h)
        fit_scale = max(0.05, fit_scale)
        self.scale_factor = max(0.05, min(12.0, fit_scale * self.zoom_factor))
        target_w = max(1, int(img_w * self.scale_factor))
        target_h = max(1, int(img_h * self.scale_factor))

        self.scaled_pixmap = self.original_pixmap.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.offset_x = 0
        self.offset_y = 0

        if not self._updating_canvas_size:
            self._updating_canvas_size = True
            self.setFixedSize(self.scaled_pixmap.size())
            self._updating_canvas_size = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._updating_canvas_size:
            self.update()

    def fit_to_viewport(self, size):
        self.viewport_w = max(50, size.width() - 8)
        self.viewport_h = max(50, size.height() - 8)
        self._rescale()
        self.update()

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            delta = ev.angleDelta().y()
            if delta > 0:
                self.zoom_factor = min(12.0, self.zoom_factor * 1.15)
            elif delta < 0:
                self.zoom_factor = max(0.2, self.zoom_factor / 1.15)
            self._rescale()
            self.update()
            ev.accept()
            return
        super().wheelEvent(ev)

    def to_original(self, x, y):
        ox = (x - self.offset_x) / self.scale_factor
        oy = (y - self.offset_y) / self.scale_factor
        pw = self.original_pixmap.width() if self.original_pixmap else 1
        ph = self.original_pixmap.height() if self.original_pixmap else 1
        margin = max(pw, ph) * 0.2
        return int(max(-margin, min(ox, pw - 1 + margin))), int(max(-margin, min(oy, ph - 1 + margin)))

    def to_display(self, ox, oy):
        return int(ox * self.scale_factor + self.offset_x), \
            int(oy * self.scale_factor + self.offset_y)

    def _point_in_obb(self, px, py, box, margin=5):
        pts = np.array(box[:8]).reshape(4, 2).astype(np.float32)
        # 💡核心修复：必须把 measureDist 设为 True，否则它只返回 -1，导致全屏判定为选中！
        dist = cv2.pointPolygonTest(pts, (float(px), float(py)), True)
        return dist >= -margin

    def _get_box_params(self, box):
        """将 8 点转换为 cx, cy, w, h, angle"""
        x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        cx = (x1 + x2 + x3 + x4) / 4
        cy = (y1 + y2 + y3 + y4) / 4
        w = math.hypot(x2 - x1, y2 - y1)
        h = math.hypot(x3 - x2, y3 - y2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return cx, cy, w, h, angle

    def _create_corners(self, cx, cy, w, h, angle_deg):
        """将 cx, cy, w, h, angle 转换为顺时针 8 个点"""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        corners = []
        for px, py in pts:
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            corners.extend([cx + rx, cy + ry])
        return corners

    def mousePressEvent(self, ev: QMouseEvent):
        self.setFocus()
        if self.scaled_pixmap is None: return
        mx, my = ev.pos().x(), ev.pos().y()
        ox, oy = self.to_original(mx, my)

        # 🎯 SAM 智能修正逻辑：如果选中了框，就覆盖它；没选中，就新建
        if ev.button() == Qt.RightButton and SAM_AVAILABLE:
            self._run_sam(ox, oy)
            return

        if ev.button() == Qt.LeftButton:
            if self.selected_box_idx >= 0:
                box = self.obb_boxes[self.selected_box_idx]
                cx, cy, w, h, angle = self._get_box_params(box)

                # 1. 检查是否点中了顶部的旋转把手
                rad = math.radians(angle - 90)  # 垂直向上
                hx = cx + math.cos(rad) * (h / 2 + 30 / self.scale_factor)
                hy = cy + math.sin(rad) * (h / 2 + 30 / self.scale_factor)
                if math.hypot(ox - hx, oy - hy) < (15 / self.scale_factor):
                    self.action_state = "ROTATING"
                    self.temp_box_state = list(box)
                    return

                # 💡替换：原本检查是否点中 4 个角的代码块
                corners = self._create_corners(cx, cy, w, h, angle)
                for i in range(0, 8, 2):
                    px, py = corners[i], corners[i + 1]
                    if math.hypot(ox - px, oy - py) < (10 / self.scale_factor):
                        self.action_state = "RESIZING"
                        self.temp_box_state = list(box)
                        # 记录对角线上的点作为死死钉住的锚点
                        self.drag_anchor_pos = (corners[(i + 4) % 8], corners[(i + 5) % 8])
                        return

            # 3. 检查是否点中了框的内部 (倒序遍历，优先点上面的)
            for idx in range(len(self.obb_boxes) - 1, -1, -1):
                if self._point_in_obb(ox, oy, self.obb_boxes[idx]):
                    self.selected_box_idx = idx
                    self.action_state = "MOVING"
                    self.drag_start_pos = (ox, oy)
                    self.temp_box_state = list(self.obb_boxes[idx])
                    self.update()
                    return

            # 4. 点了空白处 -> 开始画新框
            self.selected_box_idx = -1
            self.action_state = "DRAWING"
            self.drag_start_pos = (ox, oy)
            new_box = [ox, oy, ox, oy, ox, oy, ox, oy, self.unassigned_class_id, '']
            self.obb_boxes.append(new_box)
            self.selected_box_idx = len(self.obb_boxes) - 1
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.scaled_pixmap is None: return
        mx, my = ev.pos().x(), ev.pos().y()
        ox, oy = self.to_original(mx, my)

        if self.action_state == "DRAWING":
            sx, sy = self.drag_start_pos
            x1, y1 = min(sx, ox), min(sy, oy)
            x2, y2 = max(sx, ox), max(sy, oy)
            self.obb_boxes[self.selected_box_idx][:8] = [x1, y1, x2, y1, x2, y2, x1, y2]
            self.update()

        elif self.action_state == "MOVING":
            dx = ox - self.drag_start_pos[0]
            dy = oy - self.drag_start_pos[1]
            for i in range(0, 8, 2):
                self.obb_boxes[self.selected_box_idx][i] = self.temp_box_state[i] + dx
                self.obb_boxes[self.selected_box_idx][i + 1] = self.temp_box_state[i + 1] + dy
            self.update()

        elif self.action_state == "ROTATING":
            cx, cy, w, h, _ = self._get_box_params(self.temp_box_state)
            angle = math.degrees(math.atan2(oy - cy, ox - cx)) + 90
            self.obb_boxes[self.selected_box_idx][:8] = self._create_corners(cx, cy, w, h, angle)
            self.update()

        elif self.action_state == "RESIZING":
            cx, cy, _, _, angle = self._get_box_params(self.temp_box_state)
            rad = math.radians(angle)
            ax, ay = self.drag_anchor_pos  # 获取钉住不动的锚点

            # 计算从锚点到鼠标当前位置的向量
            vx = ox - ax
            vy = oy - ay

            # 局部坐标系的 X 轴和 Y 轴方向向量
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            # 将向量投影到局部的 X 和 Y 轴上，得到新的带符号的宽高
            proj_w = vx * cos_a + vy * sin_a
            proj_h = -vx * sin_a + vy * cos_a

            # 限制最小尺寸，防止框缩成一个点或翻转导致崩溃
            w = max(5, abs(proj_w))
            h = max(5, abs(proj_h))

            # 核心数学推导：新中心点位于 锚点 和 投影终点 的中点
            new_cx = ax + (proj_w / 2) * cos_a - (proj_h / 2) * sin_a
            new_cy = ay + (proj_w / 2) * sin_a + (proj_h / 2) * cos_a

            # 重新生成 8 个角的坐标
            self.obb_boxes[self.selected_box_idx][:8] = self._create_corners(new_cx, new_cy, w, h, angle)
            self.update()

        else:
            old_hover = self.hovered_box_idx
            self.hovered_box_idx = -1
            for idx in range(len(self.obb_boxes) - 1, -1, -1):
                if self._point_in_obb(ox, oy, self.obb_boxes[idx]):
                    self.hovered_box_idx = idx
                    break
            if old_hover != self.hovered_box_idx:
                self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        # 💡核心修复：加入 "RESIZING" 状态，确保松开鼠标后无论如何都能回到 "IDLE" 空闲状态
        if self.action_state in ["DRAWING", "MOVING", "ROTATING", "RESIZING"]:
            if self.action_state == "DRAWING" and self.selected_box_idx >= 0:
                box = self.obb_boxes[self.selected_box_idx]
                _, _, w, h, _ = self._get_box_params(box)
                if w < 5 or h < 5:  # 太小的框自动清除
                    del self.obb_boxes[self.selected_box_idx]
                    self.selected_box_idx = -1

            # 松开鼠标，彻底解除绑定
            self.action_state = "IDLE"
            self.update()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_box_idx >= 0:
                del self.obb_boxes[self.selected_box_idx]
                self.selected_box_idx = -1
                self.update()
        else:
            cls_idx = self._hotkey_to_class_idx(ev.key())
            if cls_idx is not None and self._assign_selected_class(cls_idx):
                return
        super().keyPressEvent(ev)

    def _run_sam(self, ox, oy):
        if self.original_bgr is None: return
        if self.sam_model is None:
            self.sam_model = get_sam_model(self.base_path)
            if self.sam_model is None: return

        try:
            results = self.sam_model(self.original_bgr, points=[[ox, oy]], labels=[1], verbose=False)
            if results and results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
                ys, xs = np.where(mask > 0.5)
                if len(xs) < 3: return
                points_2d = np.column_stack([xs, ys]).astype(np.float32)
                rect = cv2.minAreaRect(points_2d)
                corners = cv2.boxPoints(rect).astype(int)

                # 排序角点，转成一维 list
                cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
                angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
                corners = corners[np.argsort(angles)].flatten().tolist()

                if self.selected_box_idx >= 0 and self._point_in_obb(ox, oy, self.obb_boxes[self.selected_box_idx]):
                    # 右键点在当前框内部：修正它！
                    self.obb_boxes[self.selected_box_idx][:8] = corners
                else:
                    # 右键点在外部：创建新标注！
                    self.obb_boxes.append(corners + [self.unassigned_class_id, ''])
                    self.selected_box_idx = len(self.obb_boxes) - 1
                self.update()
        except Exception as e:
            print("SAM Error:", e)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap is None: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)

        for idx, box in enumerate(self.obb_boxes):
            pts = [QPointF(*self.to_display(box[i], box[i + 1])) for i in range(0, 8, 2)]
            poly = QPolygonF(pts)

            color = self.class_colors.get(box[8], QColor(180, 180, 180))
            is_selected = (idx == self.selected_box_idx)
            is_hovered = (idx == self.hovered_box_idx)

            # 填充半透明
            fill_color = QColor(color)
            fill_color.setAlpha(80 if is_selected else 20)
            painter.setBrush(QBrush(fill_color))
            pen_w = 3 if is_selected else (2 if is_hovered else 1)
            painter.setPen(QPen(color, pen_w, Qt.DashLine if is_selected else Qt.SolidLine))
            painter.drawPolygon(poly)

            # 绘制旋转把手 (仅选中时)
            if is_selected:
                top_mid_x = (pts[0].x() + pts[1].x()) / 2
                top_mid_y = (pts[0].y() + pts[1].y()) / 2
                cx = sum(p.x() for p in pts) / 4
                cy = sum(p.y() for p in pts) / 4
                vx, vy = top_mid_x - cx, top_mid_y - cy
                length = math.hypot(vx, vy)
                if length > 0:
                    hx, hy = top_mid_x + (vx / length) * 30, top_mid_y + (vy / length) * 30
                    painter.setPen(QPen(QColor(255, 165, 0), 2))
                    painter.drawLine(QPointF(top_mid_x, top_mid_y), QPointF(hx, hy))
                    painter.setBrush(QColor(255, 165, 0))
                    painter.drawEllipse(QPointF(hx, hy), 5, 5)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 255, 255))
                for p in pts:
                    painter.drawEllipse(p, 4, 4)

            # 绘制标签文字
            painter.setBrush(Qt.NoBrush)  # 去掉丑陋的背景填充
            painter.setPen(QPen(color, 2))  # 文字颜色和多边形框保持一致
            painter.setFont(QFont('Arial', 11, QFont.Bold))
            label = box[9] if box[9] else "未分类"
            painter.drawText(int(pts[0].x()) + 2, int(pts[0].y()) - 6, label)

        painter.end()


# ── 训练线程 ──
class _SignalWriter:
    def __init__(self, signal):
        self.signal = signal
        # 匹配所有 ANSI 终端控制字符 (用于彻底干掉乱码)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        self.current_line = ""

    def write(self, text):
        if not text: return 0

        # 1. 第一步：彻底清洗颜色乱码
        clean_text = self.ansi_escape.sub('', text)

        # 2. 第二步：逐字符模拟终端行为
        for char in clean_text:
            if char == '\r':
                # 💡 核心魔法：遇到 \r (回到行首)，直接清空当前行！
                # 这完美模拟了进度条的覆盖刷新机制，只保留最后一次满进度的数据。
                self.current_line = ""
            elif char == '\n':
                line = self.current_line.strip()
                self.current_line = ""
                if not line: continue

                # 💡 斩断进度条尾巴：把 "640: 100%|████| 3/3 [00:01...]" 切掉
                if "%|" in line:
                    idx = line.find("%|")
                    # 往回倒退，删掉跟在进度条前面的数字、空格和冒号
                    while idx > 0 and line[idx - 1] in "0123456789 :":
                        idx -= 1
                    line = line[:idx].strip()

                # 💡 精准匹配并美化输出
                if line.startswith("Epoch") and "GPU_mem" in line:
                    self.signal.emit("\n📊 " + line)
                elif re.match(r'^\d+/\d+', line):
                    self.signal.emit("🚀 " + line)
                elif line.startswith("Class") and "Images" in line:
                    self.signal.emit("🎯 " + line)
                elif line.startswith("all") and len(line.split()) >= 4:
                    self.signal.emit("✨ " + line)
                elif "best.pt" in line:
                    self.signal.emit("\n✅ " + line)
                elif "Error" in line or "Exception" in line:
                    self.signal.emit("❌ " + line)
                elif line.startswith("Ultralytics") or line.startswith("YOLO"):
                    self.signal.emit("💡 " + line)

            else:
                self.current_line += char

        return len(text)

    def flush(self):
        pass


class VideoSliceWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, int, str, object)

    def __init__(self, video_path, images_dir, interval):
        super().__init__()
        self.video_path = video_path
        self.images_dir = images_dir
        self.interval = max(1, int(interval))

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit(False, 0, "视频打开失败", [])
            return

        saved_meta = []
        frame_idx = 0
        saved = 0
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        os.makedirs(self.images_dir, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.interval == 0:
                    fn = f"{base}_f{frame_idx:06d}.jpg"
                    save_path = os.path.join(self.images_dir, fn)
                    ok, img_encode = cv2.imencode('.jpg', frame)
                    if ok:
                        img_encode.tofile(save_path)
                        h, w = frame.shape[:2]
                        saved_meta.append((fn, w, h))
                        saved += 1
                        if saved == 1 or saved % 20 == 0:
                            self.progress.emit(f"切片中... 已生成 {saved} 张")
                frame_idx += 1
        except Exception as e:
            self.finished.emit(False, saved, f"切片失败: {e}", saved_meta)
            return
        finally:
            cap.release()

        self.finished.emit(True, saved, f"切片完成: 每 {self.interval} 帧 1 张，共生成 {saved} 张", saved_meta)


class TrainWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, dataset_yaml, model_out, epochs, imgsz, batch, base_model="yolov8n-obb.pt"):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.model_out = model_out
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.base_model = base_model

    def _get_run_name(self):
        runs_dir = os.path.join(os.path.dirname(self.dataset_yaml), "runs")
        os.makedirs(runs_dir, exist_ok=True)
        count = 0
        for name in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, name)):
                count += 1

        run_index = count + 1
        if 10 <= run_index % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(run_index % 10, "th")
        return f"{run_index}{suffix}"

    def run(self):
        try:
            from ultralytics import YOLO
            self.progress.emit(f"📦 加载基座模型: {self.base_model}")
            model = YOLO(self.base_model)
            train_device, train_device_name, torch_cuda = get_training_device_info()
            cuda_text = f", torch CUDA {torch_cuda}" if torch_cuda else ""
            device_message = (
                f"Training started: base_model={self.base_model}, output={self.model_out}, "
                f"device={train_device_name}{cuda_text}"
            )
            log_training_device(device_message)
            self.progress.emit(f"训练设备: {train_device_name}{cuda_text}")
            run_name = self._get_run_name()
            runs_dir = os.path.join(os.path.dirname(self.dataset_yaml), "runs")
            run_dir = os.path.join(runs_dir, run_name)
            self.progress.emit(f"🚀 开始 OBB 训练 (epochs={self.epochs}, imgsz={self.imgsz}, batch={self.batch})")
            self.progress.emit(f"📁 本次训练输出目录: {run_dir}")

            writer = _SignalWriter(self.progress)
            try:
                with redirect_stdout(writer), redirect_stderr(writer):
                    model.train(
                        data=self.dataset_yaml,
                        epochs=self.epochs,
                        imgsz=self.imgsz,
                        batch=self.batch,
                        verbose=True,
                        project=runs_dir,
                        name=run_name,
                        exist_ok=False,
                        device=train_device,
                        # 👇 [新增] YOLO 自带的强大在线数据增强参数
                        mosaic=1.0,  # 100% 开启马赛克增强 (对微小缺陷极其有效)
                        mixup=0.1,  # 10% 概率开启 MixUp
                        degrees=15.0,  # 随机旋转 ±15 度
                        translate=0.1,  # 随机平移 10%
                        scale=0.5,  # 随机缩放 ±50%
                        hsv_h=0.015,  # 色调随机扭曲
                        hsv_s=0.7,  # 饱和度随机扭曲
                        hsv_v=0.4,  # 亮度随机扭曲
                        fliplr=0.5,  # 50% 概率水平翻转
                        workers=0,
                    )
            finally:
                writer.flush()

            best = os.path.join(run_dir, "weights", "best.pt")
            if best and os.path.exists(best):
                os.makedirs(os.path.dirname(self.model_out), exist_ok=True)
                shutil.copy(best, self.model_out)
                self.progress.emit(f"✅ 模型已保存: {self.model_out}")
                self.finished.emit(True, f"训练完成！模型: {self.model_out}")
            else:
                self.finished.emit(False, "训练完成但未找到 best.pt，请检查 runs 目录")
        except Exception as e:
            self.finished.emit(False, f"训练失败: {str(e)}")


# ── 主窗口 ──
class FastTrainerDialog(QDialog):
    def __init__(self, base_path, parent=None):
        super().__init__(parent)
        self.base_path = base_path
        self.setWindowTitle("🚀 OBB 旋转框标注训练")
        self.resize(1280, 800)

        self.project_name = ""
        self.class_names = []
        self.project_dir = ""
        self.images_dir = ""
        self.labels_dir = ""
        self.recordings_dir = ""

        self.cap = None
        self.rs_pipeline = None
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._update_preview)
        self.record_writer = None
        self.recording_video_path = ""
        self.frame_transform = "none"
        self.playback_cap = None
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._update_playback)

        self.image_list_widget = None
        self.annotator = None
        self.current_image_files = []
        self._image_meta_cache = {}
        self._image_meta_cache_dirty = False
        self.slice_worker = None

        self._init_ui()
        self._init_shortcuts()
        self._refresh_project_list()

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "📸 1. 数据采集")
        self.tabs.addTab(self._create_label_tab(), "🏷️ 2. OBB 标注")
        self.tabs.addTab(self._create_train_tab(), "🚀 3. 训练")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def _init_shortcuts(self):
        self._shortcuts = [
            QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_labels),
            QShortcut(QKeySequence("PageDown"), self, activated=self._save_and_next),
            QShortcut(QKeySequence("PageUp"), self, activated=self._save_and_prev),
        ]
        for shortcut in self._shortcuts:
            shortcut.setContext(Qt.ApplicationShortcut)

    def eventFilter(self, obj, event):
        if (
            hasattr(self, "annotation_scroll")
            and obj is self.annotation_scroll.viewport()
            and event.type() == QEvent.Resize
        ):
            QTimer.singleShot(0, self._fit_annotator_to_view)
        return super().eventFilter(obj, event)

    def _fit_annotator_to_view(self):
        if self.annotator and self.annotator.original_pixmap is not None and hasattr(self, "annotation_scroll"):
            self.annotator.fit_to_viewport(self.annotation_scroll.viewport().size())

    # ═══════════ Tab 1: 数据采集 ═══════════
    def _create_data_tab(self):
        w = QWidget()
        layout = QHBoxLayout()

        # 左侧 — 摄像头预览
        left = QVBoxLayout()
        left.addWidget(QLabel("📷 摄像头预览"))
        self.lbl_preview = QLabel("未开启摄像头")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("background: #222; color: white; min-height: 360px;")
        left.addWidget(self.lbl_preview, stretch=1)

        self.lbl_resolution = QLabel("")
        self.lbl_resolution.setAlignment(Qt.AlignCenter)
        self.lbl_resolution.setStyleSheet("color: #0f0; font-weight: bold; font-size: 14px; background: #111; padding: 4px;")
        left.addWidget(self.lbl_resolution)

        cam_ctl = QHBoxLayout()
        self.combo_cam_source = QComboBox()
        for cam_idx in range(4):
            self.combo_cam_source.addItem(f"摄像头 {cam_idx}", cam_idx)
        if HAS_REALSENSE:
            self.combo_cam_source.addItem("RealSense 彩色相机", "realsense")
        self.combo_cam_source.setToolTip("OpenCV 摄像头编号会随设备、驱动、插拔顺序变化；哪个有画面就用哪个。")
        cam_ctl.addWidget(QLabel("源:"))
        cam_ctl.addWidget(self.combo_cam_source)

        self.combo_cap_res = QComboBox()
        self.combo_cap_res.addItems(["1280×720", "1920×1080", "2560×1440", "3840×2160 (4K)"])
        self.combo_cap_res.setCurrentIndex(1)
        cam_ctl.addWidget(QLabel("分辨率:"))
        cam_ctl.addWidget(self.combo_cap_res)

        self.combo_frame_transform = QComboBox()
        self.combo_frame_transform.addItem("正常", "none")
        self.combo_frame_transform.addItem("上下翻转", "flip_v")
        self.combo_frame_transform.addItem("左右翻转", "flip_h")
        self.combo_frame_transform.addItem("180°", "rotate_180")
        self.combo_frame_transform.currentIndexChanged.connect(self._on_frame_transform_changed)
        cam_ctl.addWidget(QLabel("方向:"))
        cam_ctl.addWidget(self.combo_frame_transform)

        self.btn_open_cam = QPushButton("📷 打开摄像头")
        self.btn_open_cam.clicked.connect(self._toggle_camera)
        cam_ctl.addWidget(self.btn_open_cam)
        left.addLayout(cam_ctl)

        self.lbl_cam_status = QLabel("")
        left.addWidget(self.lbl_cam_status)
        layout.addLayout(left, stretch=3)

        # 右侧 — 项目管理 + 拍照
        right = QVBoxLayout()

        gb = QGroupBox("📁 项目")
        p_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        self.combo_projects = QComboBox()
        self.combo_projects.setToolTip("选择已有项目")
        row1.addWidget(self.combo_projects)
        self.btn_open_project = QPushButton("📂 打开")
        self.btn_open_project.clicked.connect(self._open_project)
        row1.addWidget(self.btn_open_project)
        p_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.edit_new_project = QLineEdit()
        self.edit_new_project.setPlaceholderText("输入新项目名称...")
        row2.addWidget(self.edit_new_project)
        self.btn_new_project = QPushButton("➕ 新建项目")
        self.btn_new_project.setStyleSheet("background: #28a745; color: white; font-weight: bold;")
        self.btn_new_project.clicked.connect(self._new_project)
        row2.addWidget(self.btn_new_project)
        p_layout.addLayout(row2)

        self.lbl_project_info = QLabel("")
        self.lbl_project_info.setWordWrap(True)
        p_layout.addWidget(self.lbl_project_info)
        gb.setLayout(p_layout)
        right.addWidget(gb)

        gb2 = QGroupBox("📸 拍照")
        c_layout = QVBoxLayout()
        self.btn_capture = QPushButton("📷 拍照采集")
        self.btn_capture.setStyleSheet(
            "background: #1a73e8; color: white; font-weight: bold; padding: 14px; font-size: 16px;"
        )
        self.btn_capture.clicked.connect(self._capture)
        self.btn_capture.setEnabled(False)
        c_layout.addWidget(self.btn_capture)
        self.lbl_capture_count = QLabel("")
        c_layout.addWidget(self.lbl_capture_count)
        gb2.setLayout(c_layout)
        right.addWidget(gb2)

        gb_video = QGroupBox("🎬 视频采集")
        v_layout = QVBoxLayout()
        v_btn_row = QHBoxLayout()
        self.btn_record_video = QPushButton("● 开始录制")
        self.btn_record_video.setStyleSheet("background:#dc3545; color:white; font-weight:bold;")
        self.btn_record_video.clicked.connect(self._toggle_recording)
        self.btn_record_video.setEnabled(True)
        v_btn_row.addWidget(self.btn_record_video)
        self.btn_play_video = QPushButton("▶ 播放预览")
        self.btn_play_video.clicked.connect(self._play_recording)
        self.btn_play_video.setEnabled(False)
        v_btn_row.addWidget(self.btn_play_video)
        self.btn_import_video = QPushButton("导入视频")
        self.btn_import_video.clicked.connect(self._import_video_for_slice)
        v_btn_row.addWidget(self.btn_import_video)
        self.btn_save_video = QPushButton("另存视频")
        self.btn_save_video.clicked.connect(self._save_current_video_as)
        self.btn_save_video.setEnabled(False)
        v_btn_row.addWidget(self.btn_save_video)
        v_layout.addLayout(v_btn_row)

        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("每隔"))
        self.spin_slice_interval = QSpinBox()
        self.spin_slice_interval.setRange(1, 9999)
        self.spin_slice_interval.setValue(15)
        slice_row.addWidget(self.spin_slice_interval)
        slice_row.addWidget(QLabel("帧切一张"))
        self.btn_slice_video = QPushButton("切片到图片")
        self.btn_slice_video.clicked.connect(self._slice_recording)
        self.btn_slice_video.setEnabled(False)
        slice_row.addWidget(self.btn_slice_video)
        v_layout.addLayout(slice_row)

        self.lbl_video_status = QLabel("录制前先打开摄像头并确认实时分辨率")
        self.lbl_video_status.setWordWrap(True)
        v_layout.addWidget(self.lbl_video_status)
        gb_video.setLayout(v_layout)
        right.addWidget(gb_video)

        right.addStretch()
        layout.addLayout(right, stretch=2)
        w.setLayout(layout)
        return w

    # ═══════════ Tab 2: OBB 标注 ═══════════
    def _create_label_tab(self):
        w = QWidget()
        layout = QHBoxLayout()

        # 左侧 — 图片列表
        left = QVBoxLayout()
        left.addWidget(QLabel("📁 图片列表"))
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        left.addWidget(self.image_list_widget)
        delete_row = QHBoxLayout()
        btn_delete_image = QPushButton("删除当前")
        btn_delete_image.clicked.connect(self._delete_current_image)
        delete_row.addWidget(btn_delete_image)
        btn_delete_selected = QPushButton("删除选中")
        btn_delete_selected.clicked.connect(self._delete_selected_images)
        delete_row.addWidget(btn_delete_selected)
        left.addLayout(delete_row)
        self.lbl_label_status = QLabel("")
        left.addWidget(self.lbl_label_status)
        layout.addLayout(left, stretch=1)

        # 中间 — 画布 + 工具栏
        center = QVBoxLayout()
        self.annotator = AnnotatorWidget()
        self.annotator.base_path = self.base_path
        self.annotator.setMinimumSize(640, 420)
        self.annotator.setStyleSheet("background: #111;")
        scroll = QScrollArea()
        self.annotation_scroll = scroll
        scroll.setWidget(self.annotator)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setFocusPolicy(Qt.NoFocus)  # 不抢焦点
        scroll.viewport().installEventFilter(self)
        center.addWidget(scroll, stretch=1)

        tool = QHBoxLayout()

        # 当前类别选择（新框默认用此类别）
        tool.addWidget(QLabel("当前类别:"))
        self.combo_assign_class = QComboBox()
        self.combo_assign_class.setMinimumWidth(100)
        self.combo_assign_class.currentIndexChanged.connect(self._on_current_class_changed)
        tool.addWidget(self.combo_assign_class)

        tool.addSpacing(10)
        btn_sam_hint = QLabel("🪄 右键SAM")
        btn_sam_hint.setStyleSheet("color: #1a73e8; font-weight: bold;")
        tool.addWidget(btn_sam_hint)

        tool.addSpacing(15)
        btn_clear = QPushButton("🗑️ 清空标注")
        btn_clear.clicked.connect(self._clear_labels)
        tool.addWidget(btn_clear)

        btn_save = QPushButton("💾 保存")
        btn_save.clicked.connect(self._save_labels)
        btn_save.setStyleSheet("background: #28a745; color: white; font-weight: bold;")
        tool.addWidget(btn_save)

        btn_save_prev = QPushButton("← 保存+上一张")
        btn_save_prev.clicked.connect(self._save_and_prev)
        tool.addWidget(btn_save_prev)

        btn_save_next = QPushButton("保存+下一张 →")
        btn_save_next.clicked.connect(self._save_and_next)
        btn_save_next.setStyleSheet("background: #1a73e8; color: white; font-weight: bold;")
        tool.addWidget(btn_save_next)
        center.addLayout(tool)

        hint = QLabel("左键画框 | 右键SAM | 新框默认未分类 | 选中框后按 0-9/A-Z 赋类别 | Delete删框")
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        center.addWidget(hint)
        layout.addLayout(center, stretch=3)

        # 右侧 — 类别管理
        right = QVBoxLayout()
        gb_cls = QGroupBox("🏷️ 类别管理")
        cls_layout = QVBoxLayout()
        cls_row = QHBoxLayout()
        self.edit_class = QLineEdit()
        self.edit_class.setPlaceholderText("输入类别名称...")
        cls_row.addWidget(self.edit_class)
        self.btn_add_class = QPushButton("➕ 添加")
        self.btn_add_class.clicked.connect(self._add_class)
        cls_row.addWidget(self.btn_add_class)
        cls_layout.addLayout(cls_row)
        self.list_classes = QListWidget()
        self.list_classes.setToolTip("标注时先在这里选类别，再画框")
        cls_layout.addWidget(self.list_classes)
        btn_del_class = QPushButton("🗑️ 删除选中类别")
        btn_del_class.clicked.connect(self._del_class)
        cls_layout.addWidget(btn_del_class)
        gb_cls.setLayout(cls_layout)
        right.addWidget(gb_cls)
        right.addStretch()
        layout.addLayout(right, stretch=1)

        self.list_classes.itemClicked.connect(self._on_class_list_clicked)

        w.setLayout(layout)
        return w

    def _on_class_list_clicked(self, item):
        """点击右侧类别列表时，如果有选中的框，直接修改该框的类别"""
        idx = self.list_classes.row(item)
        self.combo_assign_class.setCurrentIndex(idx)  # 同步下拉框

        if self.annotator and self.annotator.selected_box_idx >= 0:
            self.annotator.obb_boxes[self.annotator.selected_box_idx][8] = idx
            self.annotator.obb_boxes[self.annotator.selected_box_idx][9] = self.class_names[idx]
            self.annotator.update()

    # ═══════════ Tab 3: 训练 ═══════════
    def _create_train_tab(self):
        w = QWidget()
        layout = QVBoxLayout()

        info = QGroupBox("📊 数据集")
        info_layout = QVBoxLayout()
        self.lbl_dataset_info = QLabel("尚未选择数据集")
        info_layout.addWidget(self.lbl_dataset_info)
        self.btn_refresh_info = QPushButton("🔄 刷新统计")
        self.btn_refresh_info.clicked.connect(self._refresh_dataset_info)
        info_layout.addWidget(self.btn_refresh_info)
        info.setLayout(info_layout)
        layout.addWidget(info)

        params = QGroupBox("⚙️ 训练参数")
        p_layout = QVBoxLayout()
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("基座模型 (OBB):"))
        self.combo_base_model = QComboBox()
        self.combo_base_model.addItems([
            "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt",
            "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt",
        ])
        self.combo_base_model.setEditable(True)
        r1.addWidget(self.combo_base_model)
        r1.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 500)
        self.spin_epochs.setValue(20)
        r1.addWidget(self.spin_epochs)
        p_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("imgsz:"))
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 1920)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        r2.addWidget(self.spin_imgsz)
        r2.addWidget(QLabel("Batch:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(2, 64)
        self.spin_batch.setValue(8)
        r2.addWidget(self.spin_batch)
        r2.addStretch()
        p_layout.addLayout(r2)
        params.setLayout(p_layout)
        layout.addWidget(params)

        self.btn_train = QPushButton("🚀 开始训练")
        self.btn_train.setStyleSheet(
            "background: #d93025; color: white; font-weight: bold; padding: 12px; font-size: 16px;"
        )
        self.btn_train.clicked.connect(self._start_training)
        layout.addWidget(self.btn_train)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setMinimumHeight(150)
        layout.addWidget(self.text_log)

        w.setLayout(layout)
        return w

    # ── 项目管理 ──
    def _projects_dir(self):
        return os.path.join(self.base_path, "projects")

    def _is_project_loaded(self):
        if not self.project_name or not self.project_dir:
            return False
        return os.path.normcase(os.path.normpath(self.project_dir)) != os.path.normcase(os.path.normpath(self._projects_dir()))

    def _refresh_project_list(self):
        self.combo_projects.blockSignals(True)
        self.combo_projects.clear()
        self.combo_projects.addItem("")
        proj_root = self._projects_dir()
        if os.path.exists(proj_root):
            for name in sorted(os.listdir(proj_root)):
                sub = os.path.join(proj_root, name)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "images")):
                    self.combo_projects.addItem(name)
        self.combo_projects.blockSignals(False)

    def _restore_class_names(self):
        if not self.project_dir:
            return []

        class_file = os.path.join(self.project_dir, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                self.class_names = names
                return names

        class_ids = set()
        candidate_dirs = [self.labels_dir]
        for sub in ("train", "val"):
            candidate_dirs.append(os.path.join(self.labels_dir, sub))

        for label_dir in candidate_dirs:
            if not label_dir or not os.path.exists(label_dir):
                continue
            for fn in os.listdir(label_dir):
                if not fn.endswith('.txt'):
                    continue
                try:
                    with open(os.path.join(label_dir, fn), 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 9:
                                class_ids.add(int(parts[0]))
                except Exception:
                    pass

        if class_ids:
            self.class_names = [f"class_{i}" for i in sorted(class_ids)]
            self._save_classes_to_txt()
            return self.class_names

        return []

    def _new_project(self):
        name = self.edit_new_project.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        proj_root = self._projects_dir()
        proj_path = os.path.join(proj_root, name)
        if os.path.exists(proj_path):
            QMessageBox.warning(self, "提示", f"项目 '{name}' 已存在，请直接打开")
            return

        self.project_name = name
        self.project_dir = proj_path
        self.images_dir = os.path.join(proj_path, "images")
        self.labels_dir = os.path.join(proj_path, "labels")
        self.recordings_dir = os.path.join(proj_path, "recordings")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)

        self.class_names = []
        self._image_meta_cache = {}
        self._image_meta_cache_dirty = True
        self._refresh_class_ui()
        self.lbl_project_info.setText(f"✅ 当前项目: {name}\n📁 {self.images_dir}")
        self.btn_capture.setEnabled(True)
        self._update_record_buttons()
        self._update_capture_info()
        self._refresh_image_list()
        self._save_image_meta_cache(force=True)
        self._refresh_project_list()
        self.edit_new_project.clear()
        self._log(f"已创建项目: {name}")

    def _open_project(self):
        name = self.combo_projects.currentText().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请选择项目")
            return

        proj_root = self._projects_dir()
        self.project_name = name
        self.project_dir = os.path.join(proj_root, name)
        self.images_dir = os.path.join(self.project_dir, "images")
        self.labels_dir = os.path.join(self.project_dir, "labels")
        self.recordings_dir = os.path.join(self.project_dir, "recordings")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)

        # 💡优先从本地 classes.txt 加载类别名，不存在时再从标注文件恢复
        self.class_names = []
        self._restore_class_names()
        self._load_image_meta_cache()

        self._refresh_class_ui()
        self.lbl_project_info.setText(f"✅ 当前项目: {name}\n📁 {self.images_dir}")
        self.btn_capture.setEnabled(True)
        self._update_record_buttons()
        self._update_capture_info()
        self._refresh_image_list()
        self._log(f"已打开项目: {name}")

    # ── 类别管理 ──
    def _add_class(self):
        cls_name = self.edit_class.text().strip()
        if not cls_name: return
        if cls_name in self.class_names:
            QMessageBox.warning(self, "提示", "类别已存在")
            return
        self.class_names.append(cls_name)
        self._save_classes_to_txt()  # 💡核心修复：添加后立即保存到本地
        self._refresh_class_ui()
        self.edit_class.clear()

    def _del_class(self):
        row = self.list_classes.currentRow()
        if 0 <= row < len(self.class_names):
            reply = QMessageBox.question(self, "确认",
                f"删除类别 '{self.class_names[row]}'？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.class_names[row]
                self._save_classes_to_txt()  # 💡核心修复：删除后立即保存到本地
                self._refresh_class_ui()

    def _save_classes_to_txt(self):
        """将当前内存里的类别实时保存到项目目录下的 classes.txt 中"""
        if not self.project_dir:
            return
        class_file = os.path.join(self.project_dir, "classes.txt")
        with open(class_file, 'w', encoding='utf-8') as f:
            for name in self.class_names:
                f.write(f"{name}\n")
    def _refresh_class_ui(self):
        self.list_classes.clear()
        self.combo_assign_class.blockSignals(True)
        self.combo_assign_class.clear()
        for i, name in enumerate(self.class_names):
            hotkey = str(i) if i < 10 else (chr(ord('A') + i - 10) if i < 36 else "-")
            self.list_classes.addItem(f"{hotkey}: {name}")
            self.combo_assign_class.addItem(name)
        self.combo_assign_class.blockSignals(False)
        if self.annotator:
            self.annotator.set_class_names(self.class_names)
            idx = self.combo_assign_class.currentIndex()
            self.annotator.current_class_idx = idx if idx >= 0 else 0

    def _on_current_class_changed(self, idx):
        if self.annotator:
            self.annotator.current_class_idx = idx if idx >= 0 else 0
            # 核心逻辑：一旦下拉框改变，如果当前有选中的框，立刻修改该框的类别
            if self.annotator.selected_box_idx >= 0 and idx >= 0:
                self.annotator.obb_boxes[self.annotator.selected_box_idx][8] = idx
                if idx < len(self.class_names):
                    self.annotator.obb_boxes[self.annotator.selected_box_idx][9] = self.class_names[idx]
                self.annotator.update()

    # ── 摄像头 ──
    def _release_camera(self):
        try:
            self.capture_timer.stop()
        except Exception:
            pass
        self._stop_recording()
        self._stop_playback()
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[TrainerCamera] release ignored: {e}")
        self.cap = None
        if self.rs_pipeline:
            try:
                self.rs_pipeline.stop()
            except Exception as e:
                print(f"[TrainerRealSense] stop ignored: {e}")
        self.rs_pipeline = None

    def _toggle_camera(self):
        if self.cap is None and self.rs_pipeline is None:
            try:
                src_data = self.combo_cam_source.currentData()
                res_text = self.combo_cap_res.currentText().split("(")[0].strip()
                w_str, h_str = res_text.split("×")
                cw, ch = int(w_str), int(h_str)

                if src_data == "realsense":
                    if not HAS_REALSENSE:
                        raise RuntimeError("当前环境没有安装 pyrealsense2")
                    self.rs_pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.color)
                    self.rs_pipeline.start(config)
                    frame = self._read_camera_frame()
                    if frame is None:
                        raise RuntimeError("无法读取 RealSense 彩色画面")
                    actual_h, actual_w = frame.shape[:2]
                else:
                    src = int(src_data)
                    self.cap = cv2.VideoCapture(src)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    if not self.cap.isOpened():
                        raise RuntimeError("无法打开摄像头")

                    actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.lbl_resolution.setText(f"🎥 当前分辨率: {actual_w} × {actual_h}")

                self.capture_timer.start(33)
                self.btn_open_cam.setText("⏹️ 关闭摄像头")
                self.lbl_cam_status.setText("✅ 摄像头已开启")
                self._update_record_buttons()
            except Exception as e:
                self.lbl_cam_status.setText(f"❌ {e}")
                self.lbl_resolution.setText("")
                self._release_camera()
        else:
            self._release_camera()
            self.btn_open_cam.setText("📷 打开摄像头")
            self.lbl_cam_status.setText("已关闭")
            self.lbl_resolution.setText("")
            self.lbl_preview.setText("未开启摄像头")
            self._stop_recording()
            self._stop_playback()
            self._update_record_buttons()

    def _read_camera_frame(self):
        if self.rs_pipeline is not None:
            frames = self.rs_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return apply_frame_transform(frame, self.frame_transform)
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return apply_frame_transform(frame, self.frame_transform)
        return None

    def _camera_open(self):
        return self.cap is not None or self.rs_pipeline is not None

    def _on_frame_transform_changed(self):
        self.frame_transform = self.combo_frame_transform.currentData() or "none"

    def _update_record_buttons(self):
        if hasattr(self, 'btn_record_video'):
            self.btn_record_video.setEnabled(True)
        has_recording = bool(self.recording_video_path and os.path.exists(self.recording_video_path))
        if hasattr(self, 'btn_play_video'):
            self.btn_play_video.setEnabled(has_recording)
        if hasattr(self, 'btn_slice_video'):
            self.btn_slice_video.setEnabled(has_recording and bool(self.project_name))
        if hasattr(self, 'btn_save_video'):
            self.btn_save_video.setEnabled(has_recording)

    def _stop_recording(self):
        if self.record_writer is not None:
            try:
                self.record_writer.release()
            except Exception:
                pass
        self.record_writer = None
        if hasattr(self, 'btn_record_video'):
            self.btn_record_video.setText("● 开始录制")
            self.btn_record_video.setStyleSheet("background:#dc3545; color:white; font-weight:bold;")

    def _toggle_recording(self):
        if self.record_writer is not None:
            self._stop_recording()
            self.lbl_video_status.setText(f"录制完成: {self.recording_video_path}")
            print(f"[FastTrainer] recording saved: {self.recording_video_path}")
            self._update_record_buttons()
            return
        if not self._camera_open():
            self.lbl_video_status.setText("请先打开摄像头")
            QMessageBox.warning(self, "提示", "请先打开摄像头")
            return
        if not self.project_name:
            self.lbl_video_status.setText("请先新建或打开项目")
            QMessageBox.warning(self, "提示", "请先新建或打开项目")
            return
        frame = self._read_camera_frame()
        if frame is None:
            self.lbl_video_status.setText("当前摄像头没有画面")
            QMessageBox.warning(self, "提示", "当前摄像头没有画面")
            return
        os.makedirs(self.recordings_dir, exist_ok=True)
        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_video_path = os.path.join(self.recordings_dir, f"{ts}_{w}x{h}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.record_writer = cv2.VideoWriter(self.recording_video_path, fourcc, 30.0, (w, h))
        if not self.record_writer.isOpened():
            self.record_writer = None
            self.lbl_video_status.setText("视频录制器打开失败")
            QMessageBox.warning(self, "提示", "视频录制器打开失败")
            return
        self.record_writer.write(frame)
        print(f"[FastTrainer] recording started: {self.recording_video_path}")
        self.btn_record_video.setText("■ 停止录制")
        self.btn_record_video.setStyleSheet("background:#6c757d; color:white; font-weight:bold;")
        self.lbl_video_status.setText(f"录制中... {w}×{h}")

    def _stop_playback(self):
        try:
            self.playback_timer.stop()
        except Exception:
            pass
        if self.playback_cap is not None:
            try:
                self.playback_cap.release()
            except Exception:
                pass
        self.playback_cap = None

    def _play_recording(self):
        if not self.recording_video_path or not os.path.exists(self.recording_video_path):
            QMessageBox.warning(self, "提示", "还没有可播放的视频")
            return
        self._stop_playback()
        self.playback_cap = cv2.VideoCapture(self.recording_video_path)
        if not self.playback_cap.isOpened():
            self.playback_cap = None
            QMessageBox.warning(self, "提示", "视频打开失败")
            return
        self.lbl_video_status.setText(f"正在播放预览: {os.path.basename(self.recording_video_path)}")
        self.playback_timer.start(33)

    def _import_video_for_slice(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频用于切片", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self._stop_playback()
        self.recording_video_path = path
        self.lbl_video_status.setText(f"已导入视频: {os.path.basename(path)}")
        self._update_record_buttons()

    def _save_current_video_as(self):
        if not self.recording_video_path or not os.path.exists(self.recording_video_path):
            QMessageBox.warning(self, "提示", "当前没有可保存的视频")
            return
        default_name = os.path.basename(self.recording_video_path)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "另存当前视频", default_name, "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        if not save_path:
            return
        try:
            if os.path.abspath(save_path) != os.path.abspath(self.recording_video_path):
                shutil.copy2(self.recording_video_path, save_path)
            self.lbl_video_status.setText(f"视频已保存: {save_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _update_playback(self):
        if self.playback_cap is None:
            return
        ret, frame = self.playback_cap.read()
        if not ret:
            self._stop_playback()
            self.lbl_video_status.setText("播放结束")
            return
        self._show_preview_frame(frame)

    def _slice_recording(self):
        if not self.recording_video_path or not os.path.exists(self.recording_video_path):
            QMessageBox.warning(self, "提示", "请先录制一个视频")
            return
        if not self.images_dir:
            QMessageBox.warning(self, "提示", "请先新建或打开项目")
            return
        if self.slice_worker is not None and self.slice_worker.isRunning():
            QMessageBox.information(self, "切片中", "当前视频切片还在进行，请稍等")
            return
        interval = max(1, self.spin_slice_interval.value())

        self.btn_slice_video.setEnabled(False)
        self.lbl_video_status.setText("切片中... 请稍等")
        self.slice_worker = VideoSliceWorker(self.recording_video_path, self.images_dir, interval)
        self.slice_worker.progress.connect(self._on_slice_progress)
        self.slice_worker.finished.connect(self._on_slice_done)
        self.slice_worker.start()

    def _on_slice_progress(self, message):
        self.lbl_video_status.setText(message)

    def _on_slice_done(self, success, saved, message, saved_meta):
        for fn, w, h in saved_meta:
            self._remember_image_resolution(fn, w, h)
        self._save_image_meta_cache()
        self._refresh_image_list()
        self._update_capture_info()
        self.lbl_video_status.setText(message)
        self._update_record_buttons()
        if self.slice_worker is not None:
            self.slice_worker.deleteLater()
            self.slice_worker = None
        if not success:
            QMessageBox.warning(self, "切片失败", message)

    def _show_preview_frame(self, frame):
        h, w = frame.shape[:2]
        max_w, max_h = 640, 360
        scale = min(max_w / w, max_h / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        display = cv2.resize(frame, (display_w, display_h))
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, display_w, display_h, 3 * display_w, QImage.Format_RGB888)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg))

    def _update_preview(self):
        if self.cap is None and self.rs_pipeline is None:
            return
        frame = self._read_camera_frame()
        if frame is None:
            return
        if self.record_writer is not None:
            self.record_writer.write(frame)
            h, w = frame.shape[:2]
            self.lbl_video_status.setText(f"录制中... {w}×{h}")
        if self.playback_cap is None:
            self._show_preview_frame(frame)

    def _capture(self):
        if self.cap is None and self.rs_pipeline is None:
            QMessageBox.warning(self, "提示", "请先打开摄像头")
            return
        if not self.project_name:
            QMessageBox.warning(self, "提示", "请先新建或打开项目")
            return
        frame = self._read_camera_frame()
        if frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fn = f"{ts}.jpg"
        # 👇 [替换为这三行新代码]：绕过 OpenCV 的中文路径限制
        save_path = os.path.join(self.images_dir, fn)
        _, img_encode = cv2.imencode('.jpg', frame)
        img_encode.tofile(save_path)
        h, w = frame.shape[:2]
        self._remember_image_resolution(fn, w, h)
        self._save_image_meta_cache()
        self._update_capture_info()
        self._append_image_list_item(fn)
        self.lbl_cam_status.setText(f"📸 已拍: {fn} | {w}×{h}")

    def _update_capture_info(self):
        if not self.images_dir or not os.path.exists(self.images_dir):
            return
        total = sum(1 for f in os.listdir(self.images_dir)
                    if f.lower().endswith(IMAGE_EXTENSIONS))
        self.lbl_capture_count.setText(f"已采集: {total} 张图片")

    def _image_meta_cache_path(self):
        if not self.project_dir:
            return ""
        return os.path.join(self.project_dir, ".image_meta_cache.json")

    def _load_image_meta_cache(self):
        self._image_meta_cache = {}
        self._image_meta_cache_dirty = False
        cache_path = self._image_meta_cache_path()
        if not cache_path or not os.path.exists(cache_path):
            return
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._image_meta_cache = data
        except Exception:
            self._image_meta_cache = {}

    def _save_image_meta_cache(self, force=False):
        if not force and not self._image_meta_cache_dirty:
            return
        cache_path = self._image_meta_cache_path()
        if not cache_path:
            return
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._image_meta_cache, f, ensure_ascii=False, indent=2)
            self._image_meta_cache_dirty = False
        except Exception as e:
            print(f"[FastTrainer] image metadata cache save ignored: {e}")

    def _remember_image_resolution(self, fn, width, height):
        img_path = os.path.join(self.images_dir, fn)
        try:
            stat = os.stat(img_path)
            self._image_meta_cache[fn] = {
                "width": int(width),
                "height": int(height),
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
            self._image_meta_cache_dirty = True
        except Exception:
            pass

    def _image_resolution_from_header(self, img_path):
        try:
            with Image.open(img_path) as img:
                return img.size
        except Exception:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]
            return w, h

    def _image_resolution_text(self, img_path):
        fn = os.path.basename(img_path)
        try:
            stat = os.stat(img_path)
            cached = self._image_meta_cache.get(fn)
            if (
                cached
                and int(cached.get("size", -1)) == int(stat.st_size)
                and abs(float(cached.get("mtime", -1)) - float(stat.st_mtime)) < 0.001
            ):
                return f"{int(cached['width'])}×{int(cached['height'])}"

            size = self._image_resolution_from_header(img_path)
            if size is None:
                return "未知分辨率"
            w, h = size
            self._remember_image_resolution(fn, w, h)
            return f"{w}×{h}"
        except Exception:
            return "未知分辨率"

    # ── 标注 ──
    def _image_list_item_text(self, fn):
        label_fn = fn.rsplit('.', 1)[0] + '.txt'
        has = os.path.exists(os.path.join(self.labels_dir, label_fn))
        res_text = self._image_resolution_text(os.path.join(self.images_dir, fn))
        return ("🏷️ " if has else "📷 ") + f"{fn}  [{res_text}]"

    def _update_image_list_item(self, row):
        if self.image_list_widget is None or row < 0 or row >= len(self.current_image_files):
            return
        item = self.image_list_widget.item(row)
        if item is not None:
            item.setText(self._image_list_item_text(self.current_image_files[row]))

    def _append_image_list_item(self, fn):
        if self.image_list_widget is None:
            return
        self.current_image_files.append(fn)
        self.image_list_widget.addItem(self._image_list_item_text(fn))

    def _refresh_image_list(self):
        if self.image_list_widget is None or not self.images_dir:
            return
        current_name = None
        current_row = self.image_list_widget.currentRow()
        if 0 <= current_row < len(self.current_image_files):
            current_name = self.current_image_files[current_row]
        self.image_list_widget.blockSignals(True)
        self.image_list_widget.clear()
        self.current_image_files = []
        if os.path.exists(self.images_dir):
            for fn in sorted(os.listdir(self.images_dir)):
                if fn.lower().endswith(IMAGE_EXTENSIONS):
                    self.current_image_files.append(fn)
                    self.image_list_widget.addItem(self._image_list_item_text(fn))
        if current_name in self.current_image_files:
            self.image_list_widget.setCurrentRow(self.current_image_files.index(current_name))
        self.image_list_widget.blockSignals(False)
        self._save_image_meta_cache()
        self._update_label_status()

    def _on_image_selected(self, idx):
        if idx < 0 or idx >= len(self.current_image_files):
            return
        fn = self.current_image_files[idx]
        img_path = os.path.join(self.images_dir, fn)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.annotator.set_class_names(self.class_names)
            idx = self.combo_assign_class.currentIndex()
            self.annotator.current_class_idx = idx if idx >= 0 else 0
            self.annotator.set_image(img)
            self._fit_annotator_to_view()
            label_fn = fn.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_fn)
            if os.path.exists(label_path):
                self._load_obb_labels(label_path, img.shape[1], img.shape[0])
            self._update_label_status()

    def _load_obb_labels(self, label_path, img_w, img_h):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化)
                if len(parts) >= 9:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:9]]
                    # 反归一化
                    for i in range(0, 8, 2):
                        coords[i] *= img_w
                        coords[i + 1] *= img_h
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"cls:{cls_id}"
                    boxes.append(coords + [cls_id, cls_name])
        self.annotator.obb_boxes = boxes
        self.annotator.selected_box_idx = -1
        self.annotator.update()

    def _clear_labels(self):
        if self.annotator.original_pixmap is None:
            return
        reply = QMessageBox.question(self, "确认", "清空当前图片的所有标注？",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.annotator.obb_boxes = []
            self.annotator.selected_box_idx = -1
            self.annotator.update()

    def _delete_current_image(self):
        return self._delete_selected_images(use_current_if_none=True)

    def _delete_selected_images(self, use_current_if_none=False):
        if self.image_list_widget is None:
            return
        rows = sorted({idx.row() for idx in self.image_list_widget.selectedIndexes()})
        if not rows and use_current_if_none:
            row = self.image_list_widget.currentRow()
            if row >= 0:
                rows = [row]
        rows = [r for r in rows if 0 <= r < len(self.current_image_files)]
        if not rows:
            QMessageBox.warning(self, "提示", "请先选择要删除的图片")
            return
        names = [self.current_image_files[r] for r in rows]
        shown = "\n".join(names[:8])
        more = "" if len(names) <= 8 else f"\n...等 {len(names)} 张"
        reply = QMessageBox.question(
            self, "确认删除",
            f"删除选中的 {len(names)} 张图片及对应标注？\n{shown}{more}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        for fn in names:
            img_path = os.path.join(self.images_dir, fn)
            label_path = os.path.join(self.labels_dir, fn.rsplit('.', 1)[0] + '.txt')
            for path in (img_path, label_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        QMessageBox.warning(self, "删除失败", f"{path}\n{e}")
                        return
            self._image_meta_cache.pop(fn, None)

        self.annotator.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
        self.annotator.obb_boxes = []
        self.annotator.selected_box_idx = -1
        self._image_meta_cache_dirty = True
        self._save_image_meta_cache()
        first_row = rows[0]
        for row in sorted(rows, reverse=True):
            del self.current_image_files[row]
            self.image_list_widget.takeItem(row)
        if self.image_list_widget.count() > 0:
            self.image_list_widget.setCurrentRow(min(first_row, self.image_list_widget.count() - 1))
        self._update_capture_info()

    def _update_label_status(self):
        if self.image_list_widget is None:
            return
        idx = self.image_list_widget.currentRow()
        if idx < 0 or idx >= len(self.current_image_files):
            self.lbl_label_status.setText("")
            return
        fn = self.current_image_files[idx]
        label_fn = fn.rsplit('.', 1)[0] + '.txt'
        has = os.path.exists(os.path.join(self.labels_dir, label_fn))
        boxes = len(self.annotator.obb_boxes) if self.annotator else 0
        cls_counts = defaultdict(int)
        for box in (self.annotator.obb_boxes if self.annotator else []):
            name = box[9] if box[9] else "未分类"
            cls_counts[name] += 1
        lines = [f"{'🏷️ 已标注' if has else '📷 未标注'} | OBB框数: {boxes}"]
        for name, cnt in cls_counts.items():
            lines.append(f"  {name}: {cnt}")
        self.lbl_label_status.setText("\n".join(lines))

    def _save_labels(self):
        return self._do_save_labels(advance=False)

    def _save_and_next(self):
        return self._do_save_labels(advance=True)

    def _save_and_prev(self):
        return self._do_save_labels(advance=-1)

    def _do_save_labels(self, advance=False):
        if self.annotator.original_pixmap is None:
            QMessageBox.warning(self, "提示", "请先选择图片")
            return False
        row = self.image_list_widget.currentRow()
        if row < 0:
            return False
        fn = self.current_image_files[row]
        img_w = self.annotator.original_pixmap.width()
        img_h = self.annotator.original_pixmap.height()
        label_fn = fn.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_fn)

        for box in self.annotator.obb_boxes:
            cls_id = box[8]
            if cls_id is None or cls_id < 0 or cls_id >= len(self.class_names):
                QMessageBox.warning(self, "提示", "存在未分类的框，请先选中框并按 0-9/A-Z 赋类别")
                return False

        with open(label_path, 'w') as f:
            for box in self.annotator.obb_boxes:
                cls_id = box[8]
                # 8 个坐标归一化
                norm = []
                for i in range(0, 8, 2):
                    x = max(0.0, min(float(box[i]), img_w - 1))
                    y = max(0.0, min(float(box[i + 1]), img_h - 1))
                    norm.append(f"{x / img_w:.6f}")
                    norm.append(f"{y / img_h:.6f}")
                f.write(f"{cls_id} " + " ".join(norm) + "\n")

        self._update_image_list_item(row)
        self._update_label_status()

        if advance is True:
            next_row = row + 1
            if next_row < self.image_list_widget.count():
                self.image_list_widget.setCurrentRow(next_row)
        elif advance == -1:
            prev_row = row - 1
            if prev_row >= 0:
                self.image_list_widget.setCurrentRow(prev_row)
        return True

    # ── 数据集划分 ──
    def _split_dataset(self):
        """8:2 划分 train/val"""
        if not self.images_dir or not self.labels_dir:
            return False

        images = sorted([f for f in os.listdir(self.images_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        labels = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.txt')])

        # 配对 (图片和标注都存在)
        paired = []
        for img in images:
            label = img.rsplit('.', 1)[0] + '.txt'
            if label in labels:
                paired.append((img, label))

        if len(paired) < 5:
            QMessageBox.warning(self, "提示", f"有效配对太少 ({len(paired)} 对)，至少需要 5 对")
            return False

        random.shuffle(paired)
        split_idx = int(len(paired) * 0.8)
        train_pairs = paired[:split_idx]
        val_pairs = paired[split_idx:]

        # 创建目录结构
        for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            os.makedirs(os.path.join(self.project_dir, sub), exist_ok=True)

        # 复制文件
        for pairs, subset in [(train_pairs, 'train'), (val_pairs, 'val')]:
            for img_fn, lbl_fn in pairs:
                shutil.copy2(os.path.join(self.images_dir, img_fn),
                            os.path.join(self.project_dir, 'images', subset, img_fn))
                shutil.copy2(os.path.join(self.labels_dir, lbl_fn),
                            os.path.join(self.project_dir, 'labels', subset, lbl_fn))

        self._log(f"📊 划分完成: train={len(train_pairs)}, val={len(val_pairs)}")
        return True

    # ── 训练 ──
    def _refresh_dataset_info(self):
        if not self.images_dir or not os.path.exists(self.images_dir):
            self.lbl_dataset_info.setText("尚未选择数据集")
            return
        images = [f for f in os.listdir(self.images_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        labels = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')] if os.path.exists(self.labels_dir) else []

        cls_box_count = defaultdict(int)
        labeled = 0
        for fn in labels:
            try:
                with open(os.path.join(self.labels_dir, fn)) as f:
                    has_box = False
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            cls_id = int(parts[0])
                            name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"cls:{cls_id}"
                            cls_box_count[name] += 1
                            has_box = True
                    if has_box:
                        labeled += 1
            except Exception:
                pass

        lines = [f"图片: {len(images)} | 已标注: {labeled}/{len(images)}"]
        if cls_box_count:
            lines.append("各类别 OBB 框数:")
            for name, cnt in sorted(cls_box_count.items()):
                lines.append(f"  {name}: {cnt}框")
        self.lbl_dataset_info.setText("\n".join(lines))

    def _start_training(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            QMessageBox.information(self, "训练中", "当前训练还在进行，请等待完成。")
            return

        # 👇 [新增] 强制检查项目是否真正加载
        if not self._is_project_loaded():
            QMessageBox.warning(self, "操作错误", "请先在【数据采集】页面选择项目，并点击【📂 打开】按钮！")
            return

        if not os.path.exists(self.project_dir):
            QMessageBox.warning(self, "提示", "请先打开/创建项目")
            return
        images = [f for f in os.listdir(self.images_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        labels = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')] if os.path.exists(self.labels_dir) else []
        valid = 0
        for fn in labels:
            try:
                with open(os.path.join(self.labels_dir, fn)) as f:
                    for line in f:
                        if len(line.strip().split()) >= 9:
                            valid += 1
                            break
            except Exception:
                pass
        if len(images) < 10 or valid < 5:
            QMessageBox.warning(self, "提示",
                f"至少需要 10 张图片和 5 张已标注。\n当前: {len(images)} 图 / {valid} 已标注")
            return

        if not self.class_names:
            restored_names = self._restore_class_names()
            if restored_names:
                self._refresh_class_ui()
                self._log(f"🧩 已从项目标注恢复类别: {', '.join(restored_names)}")
            else:
                QMessageBox.warning(self, "提示", "当前项目没有可用的类别名称，请先添加类别后再训练")
                return
        else:
            self._save_classes_to_txt()

        self._log("📊 正在划分数据集...")
        if not self._split_dataset():
            return

        # 生成 dataset.yaml
        yaml_path = os.path.join(self.project_dir, "dataset.yaml")
        train_names = model_class_names(self.class_names)
        yaml_data = {
            "path": self.project_dir.replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(train_names)},
            "nc": len(train_names),
        }
        if _yaml_lib is None:
            QMessageBox.critical(self, "错误", "请安装 pyyaml: pip install pyyaml")
            return
        with open(yaml_path, 'w', encoding='utf-8') as f:
            _yaml_lib.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        self._log(f"📝 dataset.yaml 已生成")

        model_out = os.path.join(self.base_path, "models", f"{self.project_name}.pt")
        self.progress_bar.setVisible(True)
        self.text_log.clear()
        self.btn_train.setEnabled(False)

        base_model = self.combo_base_model.currentText().strip()
        epochs = self.spin_epochs.value()
        imgsz = self.spin_imgsz.value()
        batch = self.spin_batch.value()

        self.worker = TrainWorker(yaml_path, model_out, epochs, imgsz, batch, base_model)
        self.worker.progress.connect(self._log)
        self.worker.finished.connect(self._on_train_done)
        self.worker.start()

    def _on_train_done(self, success, msg):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        if success:
            self._log(f"✅ {msg}")
            config_path = os.path.join(self.base_path, "configs", f"{self.project_name}_map.json")
            mapping = model_mapping_from_class_names(self.class_names)
            config_data = {
                "model_path": f"models/{self.project_name}.pt",
                "mapping": mapping,
                "process_steps": [],
                "forbidden_items": "",
                "profiles": {"默认方案": {"process_steps": [], "forbidden_items": "", "step_timeout": 300}},
                "active_profile": "默认方案",
                "step_timeout": 300,
                "is_obb": True,
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=4)
            self._log(f"📝 配置已生成: {config_path}")
            # 自动刷新主界面的模型列表，加载刚训练好的模型
            try:
                parent = self.parent()
                if parent and hasattr(parent, 'refresh_model_list'):
                    parent.refresh_model_list()
                    # 自动选中刚训练的模型
                    if hasattr(parent, 'combo_models'):
                        idx = parent.combo_models.findText(self.project_name)
                        if idx >= 0:
                            parent.combo_models.setCurrentIndex(idx)
            except Exception:
                pass
            QMessageBox.information(self, "完成",
                f"OBB 模型: models/{self.project_name}.pt\n\n模型已自动加载到主界面！")
        else:
            self._log(f"❌ {msg}")
            QMessageBox.critical(self, "失败", msg)

    def _log(self, text):
        self.text_log.append(text)

    def closeEvent(self, event):
        # 如果训练还在进行，隐藏窗口而不是关闭它
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.hide()
            event.ignore()
            # 可以弹个气泡提示用户：训练已转入后台
        elif self.slice_worker is not None and self.slice_worker.isRunning():
            self.hide()
            event.ignore()
        else:
            self._save_image_meta_cache()
            self.text_log.clear()
            self._release_camera()
            super().closeEvent(event)
