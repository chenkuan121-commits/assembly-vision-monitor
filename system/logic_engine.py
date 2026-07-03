import re
class ProcessLogicEngine:
    HAND_CLASSES = {'hand', 'glove', '手', '手套'}
    ASSEMBLY_KEYWORDS = (
        '组装', '装配', '安装', '装上', '装入', '插入',
        '连接', '扣合', '压合', '合上', '拼装'
    )
    ACTION_EVIDENCE_FRAMES = 30

    def __init__(self):
        self.hit_counter = 0
        self.lookup_dict = {}
        self.eng_to_zh = {}  # 🌟 新增：用于将英文标签翻译回中文报警
        self.regex_pattern = None
        self.box_memory = {}
        self.memory_ttl = 30
        self.blind_zones = []
        self.detach_attached_seen = False
        self.detach_seen_removed = False
        self.detach_seen_base = False
        self.detach_status = {}
        self.action_armed = False
        self.action_gate_initialized = False
        self.action_seen_unmet = False
        self.action_seen_no_touch = False
        self.action_touch_frames = 0
        self.action_status = {}
        self.release_hand_seen = False
        self.release_no_hand_frames = 0
        self.release_status = {}

    def check_jump_step(self, held_object, all_steps, current_idx):
        """
        跨越步骤预判：检查手里拿的东西，是不是未来步骤才需要的
        """
        if not held_object or not all_steps:
            return False, ""

        # 如果当前步骤就需要它，那绝对没跳步
        current_step_dict = all_steps[current_idx] if current_idx < len(all_steps) else {}
        current_targets = self.parse_step_text(current_step_dict.get("text", ""))
        if held_object in current_targets:
            return False, ""

        # 往后遍历未来的所有步骤
        for i in range(current_idx + 1, len(all_steps)):
            future_targets = self.parse_step_text(all_steps[i].get("text", ""))
            if held_object in future_targets:
                zh_name = self.eng_to_zh.get(held_object, held_object)
                return True, f"🚫 跳步警告！【{zh_name}】是步骤 {i + 1} 才需要的物品，请先放下！"

        return False, ""
    def is_in_blind_zone(self, bbox, margin=20):
        """判断当前检测框的中心点是否落入了已完成的盲区"""
        # 算一下当前物品的中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        for zone in self.blind_zones:
            # zone 格式是 [x1, y1, x2, y2]
            # margin 是稍微向外扩展一点屏蔽范围，防止边缘误触
            if (zone[0] - margin < center_x < zone[2] + margin) and \
                    (zone[1] - margin < center_y < zone[3] + margin):
                return True  # 掉进盲区了！
        return False
    def build_parser(self, mapping_dict):
        self.lookup_dict = {}
        self.eng_to_zh = {}
        for _, info in mapping_dict.items():
            eng = info.get("eng_name", "")
            zh = info.get("zh_name", "")
            if zh:
                key = zh.lower()
                self.lookup_dict.setdefault(key, [])
                if eng and eng not in self.lookup_dict[key]:
                    self.lookup_dict[key].append(eng)
            if eng:
                self.lookup_dict[eng.lower()] = [eng]
            if eng and zh: self.eng_to_zh[eng] = zh  # 存入反向映射
        if not self.lookup_dict: return
        all_terms = sorted(self.lookup_dict.keys(), key=len, reverse=True)
        self.regex_pattern = re.compile('|'.join(map(re.escape, all_terms)), re.IGNORECASE)

    @staticmethod
    def target_options(target):
        if isinstance(target, (tuple, list, set)):
            return tuple(target)
        return (target,)

    @classmethod
    def target_matches(cls, class_name, target):
        return class_name in cls.target_options(target)

    @classmethod
    def is_hand_class(cls, class_name):
        cls_name = str(class_name or '')
        return cls_name in cls.HAND_CLASSES or cls_name.lower() in cls.HAND_CLASSES

    @classmethod
    def target_is_hand(cls, target):
        return any(cls.is_hand_class(option) for option in cls.target_options(target))

    @classmethod
    def has_hand_target(cls, targets):
        return any(cls.target_is_hand(target) for target in (targets or []))

    @classmethod
    def is_assembly_like_step(cls, targets, step_text=""):
        """Return True for spatial steps that represent putting two non-hand objects together."""
        if not targets or cls.has_hand_target(targets):
            return False
        non_hand_targets = [target for target in targets if not cls.target_is_hand(target)]
        if len(non_hand_targets) < 2:
            return False
        return any(keyword in str(step_text or '') for keyword in cls.ASSEMBLY_KEYWORDS)

    @classmethod
    def target_display_name(cls, target, eng_to_zh):
        options = cls.target_options(target)
        names = []
        for option in options:
            if option is None:
                continue
            zh = eng_to_zh.get(option, option)
            zh = str(zh)
            if zh not in names:
                names.append(zh)
        return "/".join(names) if names else ""

    def parse_step_text(self, text):
        if not self.regex_pattern: return []
        detected_labels = []
        for match in self.regex_pattern.finditer(text):
            word = match.group(0).lower()
            if word in self.lookup_dict:
                values = self.lookup_dict[word]
                val = values[0] if len(values) == 1 else tuple(values)
                if val not in detected_labels:
                    detected_labels.append(val)
        return detected_labels

    def check_presence(self, required_targets, current_detections):
        """轻量检查：目标是否全部出现在当前帧 (不累积状态，不修改 hit_counter/box_memory/blind_zones)"""
        if not required_targets:
            return False
        current_classes = {d['class'] for d in current_detections}
        return all(any(option in current_classes for option in self.target_options(t)) for t in required_targets)

    def check_forbidden(self, current_detections, forbidden_targets):
        if not forbidden_targets: return False, None
        for det in current_detections:
            for target in forbidden_targets:
                if self.target_matches(det['class'], target):
                    return True, det['class']
        return False, None

    def expand_bbox(self, bbox, padding_ratio):
        x1, y1, x2, y2 = bbox
        pad_w = (x2 - x1) * padding_ratio
        pad_h = (y2 - y1) * padding_ratio
        return [x1 - pad_w, y1 - pad_h, x2 + pad_w, y2 + pad_h]

    def check_intersection(self, box_a, box_b):
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])
        return (inter_x2 > inter_x1) and (inter_y2 > inter_y1)

    def target_groups_intersect(self, first_targets, second_targets, current_detections,
                                padding_ratio=0.0):
        """Return whether any detection from two configured target groups intersects."""
        try:
            padding_ratio = max(0.0, float(padding_ratio))
        except (TypeError, ValueError):
            padding_ratio = 0.0

        first_boxes = []
        second_boxes = []
        for det in current_detections:
            cls_name = det.get('class', '')
            bbox = det.get('bbox')
            if not bbox:
                continue
            expanded = self.expand_bbox(bbox, padding_ratio)
            if any(self.target_matches(cls_name, target) for target in (first_targets or [])):
                first_boxes.append(expanded)
            if any(self.target_matches(cls_name, target) for target in (second_targets or [])):
                second_boxes.append(expanded)

        return any(
            self.check_intersection(first_box, second_box)
            for first_box in first_boxes
            for second_box in second_boxes
        )

    def hands_overlap_targets(self, required_targets, current_detections, target_boxes=None, padding_ratio=0.0):
        """Return True when a visible hand/glove overlaps enough targets to be real action evidence."""
        hand_boxes = []
        if target_boxes is None:
            target_boxes = {}

        for det in current_detections:
            cls_name = det.get('class', '')
            if self.is_hand_class(cls_name):
                hand_boxes.append(self.expand_bbox(det['bbox'], 0.02))
                continue
            for target in required_targets:
                if self.target_is_hand(target):
                    continue
                if self.target_matches(cls_name, target) and target not in target_boxes:
                    target_boxes[target] = self.expand_bbox(det['bbox'], padding_ratio)

        if not hand_boxes:
            return False

        target_boxes = {
            target: box
            for target, box in target_boxes.items()
            if not self.target_is_hand(target)
        }
        if not target_boxes:
            return False

        required_touch_count = 2 if len(target_boxes) >= 2 else 1
        for hand_box in hand_boxes:
            touched_targets = 0
            for target_box in target_boxes.values():
                if self.check_intersection(hand_box, target_box):
                    touched_targets += 1
            if touched_targets >= required_touch_count:
                return True
        return False

    def _update_action_gate(self, required_targets, current_detections, active_boxes, relation_met,
                            padding_ratio, required_touch_frames=None, touch_only=False):
        """Freeze static spatial relations until fresh action evidence appears."""
        required_touch_frames = int(required_touch_frames or self.ACTION_EVIDENCE_FRAMES)
        touch_now = self.hands_overlap_targets(
            required_targets, current_detections, active_boxes, padding_ratio
        )

        if not self.action_gate_initialized:
            self.action_gate_initialized = True
            self.action_seen_unmet = not relation_met
            self.action_seen_no_touch = not touch_now
            self.action_touch_frames = 0
        else:
            if not relation_met:
                self.action_seen_unmet = True
            if not touch_now:
                self.action_seen_no_touch = True
                self.action_touch_frames = 0

        evidence = ""
        if not self.action_armed:
            if relation_met and self.action_seen_unmet and not touch_only:
                self.action_armed = True
                evidence = "relation_changed"
            elif touch_now and self.action_seen_no_touch:
                self.action_touch_frames += 1
                if self.action_touch_frames >= required_touch_frames:
                    self.action_armed = True
                    evidence = "touch"
            elif not touch_now:
                self.action_touch_frames = 0

        if self.action_armed:
            state = "armed"
        elif touch_now and self.action_seen_no_touch and self.action_touch_frames > 0:
            state = "arming"
        elif relation_met:
            state = "waiting_action"
        else:
            state = "waiting_relation"

        self.action_status = {
            "state": state,
            "armed": bool(self.action_armed),
            "relation_met": bool(relation_met),
            "touch_now": bool(touch_now),
            "touch_frames": int(self.action_touch_frames),
            "required_touch_frames": int(required_touch_frames),
            "touch_only": bool(touch_only),
            "evidence": evidence,
        }
        return self.action_armed

    def _difficulty_params(self, difficulty_str):
        if "简单" in difficulty_str or "숌데" in difficulty_str:
            return 15, 0.2, 0
        if "困难" in difficulty_str or "위켜" in difficulty_str:
            return 90, 0, 2.0
        return 40, 0.08, 1.0

    def _evaluation_params(self, difficulty_str, stable_frames_override=None,
                           padding_ratio_override=None):
        """Resolve per-step overrides while keeping difficulty as the fallback and penalty source."""
        hit_threshold, padding_ratio, penalty = self._difficulty_params(difficulty_str)
        if stable_frames_override not in (None, "", 0, "0"):
            try:
                hit_threshold = max(1, int(stable_frames_override))
            except (TypeError, ValueError):
                pass
        if padding_ratio_override not in (None, "", -1, "-1"):
            try:
                padding_ratio = max(0.0, float(padding_ratio_override))
            except (TypeError, ValueError):
                pass
        return hit_threshold, padding_ratio, penalty

    def hand_touch_relation_met(self, required_targets, current_detections, padding_ratio=0.0):
        """Return whether any visible hand/glove currently touches any configured target."""
        hand_boxes = []
        target_boxes = []

        for det in current_detections:
            cls_name = det.get('class', '')
            cls_key = cls_name.lower() if isinstance(cls_name, str) else cls_name
            if cls_key in self.HAND_CLASSES or cls_name in self.HAND_CLASSES:
                hand_boxes.append(self.expand_bbox(det['bbox'], 0.05))
            elif any(self.target_matches(cls_name, target) for target in required_targets):
                target_boxes.append(self.expand_bbox(det['bbox'], padding_ratio))

        return any(
            self.check_intersection(hand_box, target_box)
            for hand_box in hand_boxes
            for target_box in target_boxes
        )

    def hand_in_operation_zone(self, required_targets, current_detections, padding_ratio=0.15):
        """Return whether any visible hand overlaps this step's local target work zone."""
        hand_boxes = []
        target_boxes = {}

        try:
            padding_ratio = max(0.0, float(padding_ratio))
        except (TypeError, ValueError):
            padding_ratio = 0.15

        for det in current_detections:
            cls_name = det.get('class', '')
            bbox = det.get('bbox')
            if not bbox:
                continue
            if self.is_hand_class(cls_name):
                hand_boxes.append(self.expand_bbox(bbox, 0.02))
                continue
            for target in required_targets:
                if self.target_is_hand(target):
                    continue
                if self.target_matches(cls_name, target) and target not in target_boxes:
                    target_boxes[target] = self.expand_bbox(bbox, padding_ratio)

        # 装配过程中目标可能被手或另一个零件遮挡；使用已有的原始框记忆维持局部操作区。
        for target in required_targets:
            if self.target_is_hand(target) or target in target_boxes:
                continue
            memory = self.box_memory.get(target)
            if memory:
                raw_bbox = memory.get('raw_bbox', memory.get('bbox'))
                if raw_bbox:
                    target_boxes[target] = self.expand_bbox(raw_bbox, padding_ratio)

        return any(
            self.check_intersection(hand_box, target_box)
            for hand_box in hand_boxes
            for target_box in target_boxes.values()
        )

    def evaluate_hand_touch_step(self, required_targets, current_detections, difficulty_str,
                                 stable_frames_override=None, padding_ratio_override=None):
        """Complete a pick-from-bin step when a hand/glove box reaches a configured target box."""
        if not required_targets:
            return False, 0

        hit_threshold, padding_ratio, penalty = self._evaluation_params(
            difficulty_str, stable_frames_override, padding_ratio_override
        )
        step_met = self.hand_touch_relation_met(
            required_targets, current_detections, padding_ratio
        )

        if step_met:
            self.hit_counter += 1
        else:
            self.hit_counter = max(0, self.hit_counter - penalty)

        is_completed = self.hit_counter >= hit_threshold
        progress = min(100, int((self.hit_counter / hit_threshold) * 100))
        return is_completed, progress

    def evaluate_detach_step(self, required_targets, current_detections, difficulty_str, assembly_ready=True,
                             stable_frames_override=None, padding_ratio_override=None):
        """Complete a removal step after two targets are first attached and then separated.

        The first parsed target is treated as the removed part, for example "white_cable".
        The second parsed target is treated as the base/host part, for example "board".
        Completion is counted only when both targets are visible and no longer intersect
        for enough stable frames. This avoids one-frame box jitter being treated as removal.
        """
        if len(required_targets) < 2:
            self.detach_status = {"state": "missing", "stable_frames": 0, "required_frames": 0}
            return False, 0

        removed_target = required_targets[0]
        base_target = required_targets[1]
        hit_threshold, padding_ratio, penalty = self._evaluation_params(
            difficulty_str, stable_frames_override, padding_ratio_override
        )

        current_boxes = {}
        for det in current_detections:
            cls_name = det.get('class')
            for target in (removed_target, base_target):
                if self.target_matches(cls_name, target):
                    current_boxes[target] = self.expand_bbox(det['bbox'], padding_ratio)

        removed_box = current_boxes.get(removed_target)
        base_box = current_boxes.get(base_target)
        if removed_box is not None:
            self.detach_seen_removed = True
        if base_box is not None:
            self.detach_seen_base = True

        if not assembly_ready:
            self.hit_counter = max(0, self.hit_counter - penalty)
            self.detach_attached_seen = False
            self.detach_status = {
                "state": "waiting_assembly",
                "removed_present": removed_box is not None,
                "base_present": base_box is not None,
                "attached_seen": False,
                "stable_frames": int(self.hit_counter),
                "required_frames": int(hit_threshold),
                "progress": 0,
                "removed_target": removed_target,
                "base_target": base_target,
            }
            return False, 0

        both_visible = removed_box is not None and base_box is not None
        attached_now = both_visible and self.check_intersection(removed_box, base_box)
        separated_now = both_visible and not self.check_intersection(removed_box, base_box)
        if attached_now:
            self.detach_attached_seen = True
            self.hit_counter = max(0, self.hit_counter - penalty)
            relation = "attached"
        elif separated_now and self.detach_attached_seen:
            self.hit_counter += 1
            relation = "separated"
        elif separated_now:
            # 拆除必须先看到两个目标处于安装/重叠状态。
            # 如果从未安装过，只是两个物品分开出现在画面中，不能算作已拆除。
            self.hit_counter = max(0, self.hit_counter - penalty)
            relation = "waiting_attach"
        else:
            self.hit_counter = max(0, self.hit_counter - penalty)
            relation = "missing"

        is_completed = self.hit_counter >= hit_threshold
        progress = min(100, int((self.hit_counter / hit_threshold) * 100))
        if is_completed:
            state = "detached"
        elif relation == "separated":
            state = "separating"
        elif relation == "attached":
            state = "attached"
        elif relation == "waiting_attach":
            state = "waiting_attach"
        else:
            state = "missing"
        self.detach_status = {
            "state": state,
            "removed_present": removed_box is not None,
            "base_present": base_box is not None,
            "attached_seen": bool(self.detach_attached_seen),
            "stable_frames": int(self.hit_counter),
            "required_frames": int(hit_threshold),
            "progress": int(progress),
            "removed_target": removed_target,
            "base_target": base_target,
        }
        return is_completed, progress

    def evaluate_step(self, required_targets, current_detections, difficulty_str, required_count=1, strategy="lock",
                      action_gate_enabled=True, action_touch_frames=None, action_touch_only=False,
                      stable_frames_override=None, padding_ratio_override=None,
                      require_hand_release=False, hand_release_padding=0.15,
                      hand_release_frames=12):
        if not required_targets: return False, 0

        hit_threshold, padding_ratio, penalty = self._evaluation_params(
            difficulty_str, stable_frames_override, padding_ratio_override
        )

        # 🌟 修复 1：让记忆“变老”并自动清理，消灭拿走后产生的永久残影
        keys_to_remove = []
        for k in self.box_memory:
            self.box_memory[k]['age'] += 1
            if self.box_memory[k]['age'] > self.memory_ttl:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.box_memory[k]

        current_found_boxes = {}
        for det in current_detections:
            cls_name = det['class']
            matched_target = next((target for target in required_targets if self.target_matches(cls_name, target)), None)
            if matched_target is not None:
                if required_count > 1 and strategy == "lock" and self.is_in_blind_zone(det['bbox']):
                    continue
                bbox_expanded = self.expand_bbox(det['bbox'], padding_ratio)
                current_found_boxes[matched_target] = bbox_expanded
                # 实况看到就更新坐标，寿命清零
                self.box_memory[matched_target] = {'bbox': bbox_expanded, 'raw_bbox': det['bbox'], 'age': 0}

        active_boxes = {}
        all_present = True

        # 🌟 修复 2：引入“主动元件”丢失监控
        active_target_missing = False

        for idx, target in enumerate(required_targets):
            if target in current_found_boxes:
                active_boxes[target] = current_found_boxes[target]
            elif target in self.box_memory:
                active_boxes[target] = self.box_memory[target]['bbox']

                # 【神级逻辑】：主谓宾分离
                # required_targets[0] 是“主动元件”（比如盖子、PCB），它必须被实况看到！它如果用了记忆，说明被拿走。
                # required_targets[1:] 是“被动元件”（比如底座），它们允许被主动元件合法遮挡！
                if idx == 0:
                    active_target_missing = True
            else:
                all_present = False
                break

        step_met = False
        if all_present:
            if len(required_targets) == 1:
                step_met = True
            elif len(required_targets) == 2:
                if self.check_intersection(active_boxes[required_targets[0]], active_boxes[required_targets[1]]):
                    step_met = True
            elif len(required_targets) >= 3:
                main_tool = active_boxes[required_targets[0]]
                for target in required_targets[1:]:
                    if self.check_intersection(main_tool, active_boxes[target]):
                        step_met = True
                        break

        valid_step_met = step_met and not active_target_missing
        if action_gate_enabled:
            action_ready = self._update_action_gate(
                required_targets, current_detections, active_boxes if all_present else {}, valid_step_met,
                padding_ratio, required_touch_frames=action_touch_frames, touch_only=action_touch_only
            )
        else:
            action_ready = True
            self.action_status = {}

        # 🌟 修复 3：智能决断进度条命运与记忆刷新
        if step_met and action_ready:
            if not active_target_missing:
                # 【合理遮挡】主动元件（盖子/PCB）在画面中，被动元件（底座）哪怕被遮挡了也没关系！进度条照涨！
                self.hit_counter += 1

                # 【续命机制】如果底座被正确覆盖了，我们在它被覆盖期间，不断刷新它的寿命！
                # 这样它就不会在长达 90 帧的困难装配过程中，因为超过 30 帧记忆上限而突然消失！
                for target in required_targets[1:]:
                    if target not in current_found_boxes and target in self.box_memory:
                        self.box_memory[target]['age'] = 0
            else:
                # 【错误遮挡/拿走】动用了主动元件的残影（说明盖子或PCB被拿走了），悬停卡死
                pass
        else:
            self.hit_counter = max(0, self.hit_counter - penalty)

        relation_progress = min(100, int((self.hit_counter / hit_threshold) * 100))
        relation_confirmed = (
            self.hit_counter >= hit_threshold and valid_step_met and action_ready
        )

        if require_hand_release:
            try:
                required_release_frames = max(1, int(hand_release_frames))
            except (TypeError, ValueError):
                required_release_frames = 12

            hand_in_zone = self.hand_in_operation_zone(
                required_targets, current_detections, hand_release_padding
            )
            if hand_in_zone:
                self.release_hand_seen = True
                self.release_no_hand_frames = 0
            elif self.release_hand_seen:
                self.release_no_hand_frames += 1
            else:
                self.release_no_hand_frames = 0

            release_confirmed = (
                self.release_hand_seen
                and self.release_no_hand_frames >= required_release_frames
            )
            is_completed = relation_confirmed and release_confirmed
            progress = 100 if is_completed else min(99, relation_progress)

            if is_completed:
                release_state = "completed"
            elif hand_in_zone:
                release_state = "operating"
            elif not self.release_hand_seen:
                release_state = "waiting_hand"
            elif not relation_confirmed:
                release_state = "waiting_relation"
            else:
                release_state = "waiting_release"

            self.release_status = {
                "state": release_state,
                "enabled": True,
                "hand_seen": bool(self.release_hand_seen),
                "hand_in_zone": bool(hand_in_zone),
                "release_frames": int(self.release_no_hand_frames),
                "required_release_frames": int(required_release_frames),
                "relation_confirmed": bool(relation_confirmed),
                "relation_progress": int(relation_progress),
                "progress": int(progress),
            }
        else:
            is_completed = self.hit_counter >= hit_threshold
            progress = relation_progress
            self.release_status = {}

        if is_completed:
            if required_count > 1 and strategy == "lock" and len(required_targets) > 0:
                best_target = None
                min_area = float('inf')

                for target in required_targets:
                    if target in self.box_memory and target != 'hand':
                        raw_box = self.box_memory[target].get('raw_bbox', self.box_memory[target]['bbox'])
                        area = (raw_box[2] - raw_box[0]) * (raw_box[3] - raw_box[1])
                        if area < min_area:
                            min_area = area
                            best_target = target

                if not best_target:
                    for t in required_targets:
                        if t != 'hand' and t in self.box_memory:
                            best_target = t
                            break

                if best_target and best_target in self.box_memory:
                    target_raw_box = self.box_memory[best_target].get('raw_bbox', active_boxes[best_target])
                    self.blind_zones.append(target_raw_box)

        return is_completed, progress

    # 🌟 修改点 3：增加 clear_blind_zones 参数，控制要不要清空盲区
    def reset(self, clear_blind_zones=True):
        self.hit_counter = 0
        self.box_memory.clear()
        self.detach_attached_seen = False
        self.detach_seen_removed = False
        self.detach_seen_base = False
        self.detach_status = {}
        self.action_armed = False
        self.action_gate_initialized = False
        self.action_seen_unmet = False
        self.action_seen_no_touch = False
        self.action_touch_frames = 0
        self.action_status = {}
        self.release_hand_seen = False
        self.release_no_hand_frames = 0
        self.release_status = {}
        if clear_blind_zones:
            self.blind_zones.clear()
if __name__ == '__main__':
    c = ProcessLogicEngine()
    c.hit_counter = 3
    print(c.__dict__)
