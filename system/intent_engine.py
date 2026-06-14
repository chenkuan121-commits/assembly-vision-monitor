import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math

class IntentEngine:
    def __init__(self):
        # ---------- 开关 ----------
        self.enable_mediapipe = False  # 默认关闭，由主界面手动开启

        # 这些类别即使被 YOLO 检测到也不能算"拿起"（身体部位/穿戴物）
        self.held_blacklist = {'glove', 'hand', '手套', '手'}

        # ---------- 拧螺丝方向识别 (doubao 完整算法) ----------
        self.twist_trackers = {}          # hand_id -> tracker
        self.TWIST_CONTINUITY_FRAMES = 5#要求超过多少帧
        self.TWIST_TRIGGER_PIXELS = 40#总位移要达到多少
        self.TWIST_COOLDOWN_TICKS = 10
        self.TWIST_HISTORY_LEN = 20
        self.twist_tools = {'screw', 'torque_driver', '螺丝刀', '力矩扳手'}

        # ---------- 全局锁定 (完全仿照 doubao.py) ----------
        self.global_held_object = None    # 当前手里拿着的物体
        self.hand_id_counter = 0

        # ---------- 抓取锁定机制 (防遮挡/防跳动) ----------
        self.held_lock_frames = 0          # 当前物品已连续锁定的帧数
        self.held_lost_frames = 0          # 锁定物品连续未被检测到的帧数
        self.pending_held_object = None    # 待确认的候选物品
        self.pending_held_frames = 0       # 候选物品的连续出现帧数
        self.switch_candidate = None       # 遮挡期间出现的新候选物品
        self.switch_candidate_frames = 0   # 新候选物品的连续出现帧数

        self.LOCK_ESTABLISH_FRAMES = 6     # 连续抓取多少帧后锁定
        self.MAX_LOST_TOLERANCE = 40       # 锁定后最多容忍多少帧检测不到
        self.SWITCH_THRESHOLD = 12         # 遮挡期间新物品连续出现多少帧才切换

        # MediaPipe 手部检测：懒加载，避免默认关闭时仍创建 TFLite delegate
        self.mp_hands = mp.solutions.hands
        self.hands = None
        # 🌟 性能优化：降分辨率 + 跳帧
        self.mp_process_every_n = 2  # 每 N 帧跑一次 MediaPipe
        self.mp_frame_counter = 0
        self.mp_cached_results = None
        self.mp_cached_scale = None  # (scale_w, scale_h) 用于还原坐标

    def reset_runtime_state(self):
        """清理跨帧运行态，保留 MediaPipe 开关和模型对象。"""
        self.twist_trackers.clear()
        self.global_held_object = None
        self.hand_id_counter = 0
        self.held_lock_frames = 0
        self.held_lost_frames = 0
        self.pending_held_object = None
        self.pending_held_frames = 0
        self.switch_candidate = None
        self.switch_candidate_frames = 0
        self.mp_frame_counter = 0
        self.mp_cached_results = None
        self.mp_cached_scale = None

    def _ensure_hands(self):
        if self.hands is None:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )

    # ------------------ 捏握判定 (doubao 原版) ------------------
    def _is_pinching(self, hand_landmarks, img_w, img_h):
        p4 = hand_landmarks.landmark[4]
        p8 = hand_landmarks.landmark[8]
        p12 = hand_landmarks.landmark[12]
        x4, y4 = p4.x * img_w, p4.y * img_h
        x8, y8 = p8.x * img_w, p8.y * img_h
        x12, y12 = p12.x * img_w, p12.y * img_h
        dist_4_8 = math.hypot(x4 - x8, y4 - y8)
        dist_4_12 = math.hypot(x4 - x12, y4 - y12)

        # 核心修复：以 1280 宽度为基准计算缩放比例
        scale_ratio_w = img_w / 1280.0
        dynamic_pinch_threshold = 150 * scale_ratio_w

        return (dist_4_8 < dynamic_pinch_threshold) and (dist_4_12 < dynamic_pinch_threshold)

    # ------------------ 外接框重叠检测 ------------------
    def _check_overlap(self, hand_bbox, obj_bbox):
        hx1, hy1, hx2, hy2 = hand_bbox
        ox1, oy1, ox2, oy2 = obj_bbox
        return not (hx2 < ox1 or hx1 > ox2 or hy2 < oy1 or hy1 > oy2)

    # ------------------ 捏合点中心 (拇指食指之间) ------------------
    def _get_pinch_center(self, hand_landmarks, img_w, img_h):
        """返回拇指尖(4)和食指尖(8)之间的中点，即实际抓取位置"""
        p4 = hand_landmarks.landmark[4]
        p8 = hand_landmarks.landmark[8]
        cx = (p4.x + p8.x) / 2 * img_w
        cy = (p4.y + p8.y) / 2 * img_h
        return cx, cy

    # ------------------ 最优重叠物体选择 ------------------
    def _find_best_overlap(self, hand_landmarks, hand_bbox, yolo_detections, img_w, img_h):
        """
        从所有与手重叠的物体中选出最优候选。
        优先级：当前锁定的物品 > 离捏合点最近的物品
        返回: (best_class, best_distance) 或 (None, inf)
        """
        pinch_cx, pinch_cy = self._get_pinch_center(hand_landmarks, img_w, img_h)
        best_obj = None
        best_dist = float('inf')

        for det in yolo_detections:
            obj_bbox = det['bbox']
            if self._check_overlap(hand_bbox, obj_bbox):
                cls_name = det['class']
                # 手套/手是穿戴物，不算"拿起"
                if cls_name in self.held_blacklist:
                    continue
                # 如果已经锁定了某物品，且它仍在重叠列表中，直接返回它
                if cls_name == self.global_held_object:
                    return cls_name, 0
                # 计算物体中心到捏合点的距离
                obj_cx = (obj_bbox[0] + obj_bbox[2]) / 2
                obj_cy = (obj_bbox[1] + obj_bbox[3]) / 2
                dist = ((obj_cx - pinch_cx) ** 2 + (obj_cy - pinch_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_obj = cls_name

        return best_obj, best_dist

    # ------------------ 拧螺丝方向识别 (doubao 完整算法) ------------------
    def _get_or_create_tracker(self, hand_id):
        if hand_id not in self.twist_trackers:
            self.twist_trackers[hand_id] = {
                'angle_history': deque(maxlen=self.TWIST_HISTORY_LEN),
                'direction_history': deque(maxlen=self.TWIST_CONTINUITY_FRAMES),
                'cooldown_cw': 0,
                'cooldown_ccw': 0,
                'last_state': None
            }
        return self.twist_trackers[hand_id]

    def _update_twist_direction(self, hand_id, hand_landmarks, img_w, img_h):
        tracker = self._get_or_create_tracker(hand_id)

        # 分方向冷却
        if tracker['cooldown_cw'] > 0:
            tracker['cooldown_cw'] -= 1
            return tracker['last_state']
        if tracker['cooldown_ccw'] > 0:
            tracker['cooldown_ccw'] -= 1
            return tracker['last_state']

        # 核心修复：以 720 高度为基准计算 Y 轴缩放比例
        scale_ratio_h = img_h / 720.0
        dynamic_noise_threshold = 2 * scale_ratio_h
        dynamic_trigger_pixels = self.TWIST_TRIGGER_PIXELS * scale_ratio_h

        p4 = hand_landmarks.landmark[4]
        p8 = hand_landmarks.landmark[8]
        y_diff = (p8.y - p4.y) * img_h
        tracker['angle_history'].append(y_diff)

        if len(tracker['angle_history']) >= 2:
            diff = tracker['angle_history'][-1] - tracker['angle_history'][-2]
            # 使用动态防抖阈值
            if diff > dynamic_noise_threshold:
                tracker['direction_history'].append(1)
            elif diff < -dynamic_noise_threshold:
                tracker['direction_history'].append(-1)
            else:
                tracker['direction_history'].append(0)
        else:
            tracker['direction_history'].append(0)

        if len(tracker['angle_history']) < self.TWIST_CONTINUITY_FRAMES:
            return None

        total_diff = tracker['angle_history'][-1] - tracker['angle_history'][0]
        recent_dirs = list(tracker['direction_history'])[-self.TWIST_CONTINUITY_FRAMES:]

        # 使用动态触发阈值
        valid_cw = (sum(recent_dirs) >= self.TWIST_CONTINUITY_FRAMES - 1) and (total_diff > dynamic_trigger_pixels)
        valid_ccw = (sum(recent_dirs) <= -(self.TWIST_CONTINUITY_FRAMES - 1)) and (total_diff < -dynamic_trigger_pixels)

        if valid_cw:
            tracker['last_state'] = "顺时针拧"
            tracker['cooldown_cw'] = self.TWIST_COOLDOWN_TICKS
            tracker['angle_history'].clear()
            tracker['direction_history'].clear()
            return tracker['last_state']
        elif valid_ccw:
            tracker['last_state'] = "逆时针拧"
            tracker['cooldown_ccw'] = self.TWIST_COOLDOWN_TICKS
            tracker['angle_history'].clear()
            tracker['direction_history'].clear()
            return tracker['last_state']
        else:
            return None

    def _is_required_target(self, class_name, required_targets):
        for target in required_targets or []:
            if isinstance(target, (tuple, list, set)):
                if class_name in target:
                    return True
            elif class_name == target:
                return True
        return False

    def _is_twist_tool(self, class_name, eng_to_zh_dict):
        zh_name = eng_to_zh_dict.get(class_name, class_name)
        return (
            class_name in self.twist_tools
            or zh_name in self.twist_tools
            or "screw" in str(class_name).lower()
        )

    # ------------------ 主入口：完全模仿 doubao.py 逻辑 ------------------
    def process_intent(self, frame, yolo_detections, required_targets, progress, is_pausing, step_text, eng_to_zh_dict):
        if not self.enable_mediapipe:
            return [], [], []
        self._ensure_hands()

        img_h, img_w, _ = frame.shape

        # 1. 检测手部 (🌟 跳帧优化：每 N 帧跑一次 MediaPipe，其余帧复用缓存)
        self.mp_frame_counter += 1
        if self.mp_frame_counter % self.mp_process_every_n == 1 or self.mp_cached_results is None:
            # 降分辨率：限制短边为 480px，大幅减少 MediaPipe CPU 开销
            mp_scale = 480.0 / min(img_h, img_w)
            if mp_scale < 1.0:
                mp_w, mp_h = int(img_w * mp_scale), int(img_h * mp_scale)
                mp_frame = cv2.resize(frame, (mp_w, mp_h))
            else:
                mp_w, mp_h = img_w, img_h
                mp_frame = frame
            mp_results = self.hands.process(cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB))
            self.mp_cached_results = mp_results
            self.mp_cached_scale = (img_w / mp_w, img_h / mp_h)
        else:
            mp_results = self.mp_cached_results

        hand_results = []
        held_objects_this_frame = []
        final_renders = []

        any_hand_gripping = False
        touched_objects_this_frame = []

        # 用于临时存放双手的数据
        hands_data = []

        if mp_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
                pts = np.array([[int(lm.x * img_w), int(lm.y * img_h)] for lm in hand_landmarks.landmark])
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)
                current_hand_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                is_grip = self._is_pinching(hand_landmarks, img_w, img_h)
                if is_grip:
                    any_hand_gripping = True

                # 用最优选择取代简单的第一个重叠物体
                best_obj, best_dist = None, float('inf')
                if is_grip:
                    best_obj, best_dist = self._find_best_overlap(
                        hand_landmarks, current_hand_bbox, yolo_detections, img_w, img_h
                    )
                    if best_obj:
                        touched_objects_this_frame.append(best_obj)

                hands_data.append({
                    'id': f"hand_{idx}",
                    'landmarks': hand_landmarks,
                    'bbox': current_hand_bbox,
                    'is_grip': is_grip,
                    'touched_obj': best_obj
                })

        # 3. 🌟🌟🌟 抓取锁定状态机 (防遮挡/防跳动) 🌟🌟🌟
        if any_hand_gripping:
            # 汇总所有手当前重叠到的物体
            all_overlapping = list(set(
                h['touched_obj'] for h in hands_data if h['is_grip'] and h['touched_obj']
            ))

            if self.global_held_object is not None:
                # ---- 状态：已锁定某物品 ----
                if self.global_held_object in all_overlapping:
                    # 物品仍可见且在手边 → 刷新锁定，清零丢失计数器
                    self.held_lost_frames = 0
                    self.held_lock_frames += 1
                    self.switch_candidate = None
                    self.switch_candidate_frames = 0
                else:
                    # 锁定的物品暂时不可见（可能被遮挡）
                    self.held_lost_frames += 1

                    if len(all_overlapping) > 0:
                        # 有其他物品出现，但不立即切换——先观察是否持续出现
                        new_candidate = all_overlapping[0]
                        if self.switch_candidate == new_candidate:
                            self.switch_candidate_frames += 1
                        else:
                            self.switch_candidate = new_candidate
                            self.switch_candidate_frames = 1

                        # 只有当丢失原物品超过容忍上限，且新候选持续出现，才切换
                        if (self.held_lost_frames > self.MAX_LOST_TOLERANCE and
                                self.switch_candidate_frames >= self.SWITCH_THRESHOLD):
                            self.global_held_object = new_candidate
                            self.held_lock_frames = self.switch_candidate_frames
                            self.held_lost_frames = 0
                            self.switch_candidate = None
                            self.switch_candidate_frames = 0
                    else:
                        # 没有任何物体重叠，候选也要衰减
                        self.switch_candidate_frames = max(0, self.switch_candidate_frames - 1)
                        if self.switch_candidate_frames <= 0:
                            self.switch_candidate = None

                    # 如果丢失太久且无有效候选，彻底解锁
                    if self.held_lost_frames > self.MAX_LOST_TOLERANCE and self.switch_candidate is None:
                        self.global_held_object = None
                        self.held_lock_frames = 0
                        self.held_lost_frames = 0
            else:
                # ---- 状态：未锁定，观察候选物品 ----
                if len(all_overlapping) > 0:
                    candidate = all_overlapping[0]
                    if self.pending_held_object == candidate:
                        self.pending_held_frames += 1
                        if self.pending_held_frames >= self.LOCK_ESTABLISH_FRAMES:
                            # 候选物品连续出现足够帧数 → 建立锁定
                            self.global_held_object = candidate
                            self.held_lock_frames = self.pending_held_frames
                            self.held_lost_frames = 0
                            self.pending_held_object = None
                            self.pending_held_frames = 0
                    else:
                        self.pending_held_object = candidate
                        self.pending_held_frames = 1
                else:
                    # 没有物体，衰减待确认
                    self.pending_held_frames = max(0, self.pending_held_frames - 1)
                    if self.pending_held_frames <= 0:
                        self.pending_held_object = None
        else:
            # ---- 手松开 → 立即释放所有锁定 ----
            self.global_held_object = None
            self.held_lock_frames = 0
            self.held_lost_frames = 0
            self.pending_held_object = None
            self.pending_held_frames = 0
            self.switch_candidate = None
            self.switch_candidate_frames = 0

        # 4. 构建每只手的返回数据和渲染指令
        for h_data in hands_data:
            hand_id = h_data['id']
            hand_landmarks = h_data['landmarks']
            current_hand_bbox = h_data['bbox']
            is_grip = h_data['is_grip']

            # 方向识别（仅当手里有工具时）
            twist_dir = None
            if self.global_held_object and self._is_twist_tool(self.global_held_object, eng_to_zh_dict):
                twist_dir = self._update_twist_direction(hand_id, hand_landmarks, img_w, img_h)

            box_color = (200, 200, 200)
            display_text = "手 (空闲)"

            if self.global_held_object and is_grip:
                held_zh = eng_to_zh_dict.get(self.global_held_object, self.global_held_object)
                if required_targets and not self._is_required_target(self.global_held_object, required_targets):
                    box_color = (0, 255, 255)
                    display_text = f"🖐️ 接触: {held_zh}"
                else:
                    box_color = (0, 255, 0)
                    if is_pausing:
                        display_text = f"✅ 完成"
                    elif progress > 0:
                        display_text = f"⏳ 正在执行 [{progress}%]"
                    else:
                        display_text = f"🎯 拿起: {held_zh}"

                if twist_dir:
                    display_text += f" [{twist_dir}]"

                if self.global_held_object not in held_objects_this_frame:
                    held_objects_this_frame.append(self.global_held_object)

            hand_results.append({
                'id': hand_id,
                'bbox': current_hand_bbox,
                'landmarks': hand_landmarks,
                'touched_obj': self.global_held_object if is_grip else None,
                'cx': (current_hand_bbox[0] + current_hand_bbox[2]) / 2,
                'cy': (current_hand_bbox[1] + current_hand_bbox[3]) / 2,
                'is_grip': is_grip,
                'twist_direction': twist_dir
            })

            final_renders.append((
                current_hand_bbox,
                box_color,
                display_text,
                'aabb',
                hand_landmarks,
                hand_id
            ))

        return hand_results, held_objects_this_frame, final_renders
