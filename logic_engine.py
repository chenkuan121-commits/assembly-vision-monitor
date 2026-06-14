import re
class ProcessLogicEngine:
    HAND_CLASSES = {'hand', 'glove', '手', '手套'}

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
    def target_display_name(cls, target, eng_to_zh):
        options = cls.target_options(target)
        names = []
        for option in options:
            zh = eng_to_zh.get(option, option)
            if zh not in names:
                names.append(zh)
        return "/".join(names)

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

    def _difficulty_params(self, difficulty_str):
        if "简单" in difficulty_str or "숌데" in difficulty_str:
            return 15, 0.2, 0
        if "困难" in difficulty_str or "위켜" in difficulty_str:
            return 90, 0, 2.0
        return 50, 0.08, 1.0

    def evaluate_hand_touch_step(self, required_targets, current_detections, difficulty_str):
        """Complete a pick-from-bin step when a hand/glove box reaches a configured target box."""
        if not required_targets:
            return False, 0

        hit_threshold, padding_ratio, penalty = self._difficulty_params(difficulty_str)
        hand_boxes = []
        target_boxes = []

        for det in current_detections:
            cls_name = det.get('class', '')
            cls_key = cls_name.lower() if isinstance(cls_name, str) else cls_name
            if cls_key in self.HAND_CLASSES or cls_name in self.HAND_CLASSES:
                hand_boxes.append(self.expand_bbox(det['bbox'], 0.05))
            elif any(self.target_matches(cls_name, target) for target in required_targets):
                target_boxes.append(self.expand_bbox(det['bbox'], padding_ratio))

        step_met = any(
            self.check_intersection(hand_box, target_box)
            for hand_box in hand_boxes
            for target_box in target_boxes
        )

        if step_met:
            self.hit_counter += 1
        else:
            self.hit_counter = max(0, self.hit_counter - penalty)

        is_completed = self.hit_counter >= hit_threshold
        progress = min(100, int((self.hit_counter / hit_threshold) * 100))
        return is_completed, progress

    def evaluate_detach_step(self, required_targets, current_detections, difficulty_str):
        """Complete a removal step after two targets are first attached and then separated.

        The first parsed target is treated as the removed part, for example "white_cable".
        The second parsed target is treated as the base/host part, for example "board".
        Completion is counted when either:
        - both targets are visible and no longer intersect for enough frames, or
        - the removed part was seen before, then leaves the view while the base remains visible.
        """
        if len(required_targets) < 2:
            return False, 0

        removed_target = required_targets[0]
        base_target = required_targets[1]
        hit_threshold, padding_ratio, penalty = self._difficulty_params(difficulty_str)

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

        attached_now = (
            removed_box is not None
            and base_box is not None
            and self.check_intersection(removed_box, base_box)
        )
        if attached_now:
            self.detach_attached_seen = True
            self.hit_counter = max(0, self.hit_counter - penalty)
        else:
            loose_separated_now = (
                base_box is not None
                and removed_box is not None
                and not self.check_intersection(removed_box, base_box)
            )
            removed_missing_after_seen = (
                base_box is not None
                and removed_box is None
                and self.detach_seen_removed
                and self.detach_seen_base
            )
            if loose_separated_now or removed_missing_after_seen:
                self.hit_counter += 1
            else:
                self.hit_counter = max(0, self.hit_counter - penalty)

        is_completed = self.hit_counter >= hit_threshold
        progress = min(100, int((self.hit_counter / hit_threshold) * 100))
        return is_completed, progress

    def evaluate_step(self, required_targets, current_detections, difficulty_str, required_count=1, strategy="lock"):
        if not required_targets: return False, 0

        hit_threshold, padding_ratio, penalty = self._difficulty_params(difficulty_str)

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

        # 🌟 修复 3：智能决断进度条命运与记忆刷新
        if step_met:
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

        is_completed = self.hit_counter >= hit_threshold
        progress = min(100, int((self.hit_counter / hit_threshold) * 100))

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
        if clear_blind_zones:
            self.blind_zones.clear()
if __name__ == '__main__':
    c = ProcessLogicEngine()
    c.hit_counter = 3
    print(c.__dict__)
