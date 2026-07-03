import time
import re
from collections import deque
from logic_engine import ProcessLogicEngine


class WorkflowMonitor:
    def __init__(self, base_logic_engine):
        self.base_logic = base_logic_engine
        self.shadow_engines = {}  # 存放未来步骤的独立进度计算器
        self.alarm_message = ""
        self.alarm_expiry_time = 0.0
        self.restart_guard_active = False
        self.restart_guard_frames = 0
        self.restart_guard_idle_frames = 0
        self.monitor_scope = "next_2"
        self.strong_action_enabled = False
        self.strong_action_frames = 30
        self.ignore_static_intersection = True
        self.shadow_sub_counts = {}
        self.shadow_cooldown_until = {}
        self.shadow_wait_release = set()
        self.prewarning_windows = {}

    def configure(self, monitor_scope=None, strong_action_enabled=None, strong_action_frames=None,
                  ignore_static_intersection=None):
        if monitor_scope is not None:
            self.monitor_scope = str(monitor_scope or "next_2")
        if strong_action_enabled is not None:
            self.strong_action_enabled = bool(strong_action_enabled)
        if strong_action_frames is not None:
            self.strong_action_frames = max(1, int(strong_action_frames))
        if ignore_static_intersection is not None:
            self.ignore_static_intersection = bool(ignore_static_intersection)
        self.reset_runtime()

    def reset_runtime(self, clear_restart_guard=False):
        self.shadow_engines.clear()
        self.shadow_sub_counts.clear()
        self.shadow_cooldown_until.clear()
        self.shadow_wait_release.clear()
        self.prewarning_windows.clear()
        self.alarm_message = ""
        self.alarm_expiry_time = 0.0
        if clear_restart_guard:
            self.restart_guard_active = False
            self.restart_guard_frames = 0
            self.restart_guard_idle_frames = 0

    def arm_restart_guard(self, frames=75):
        self.reset_runtime(clear_restart_guard=True)
        self.restart_guard_active = True
        self.restart_guard_frames = int(frames)
        self.restart_guard_idle_frames = 0

    def clear_prewarning_runtime(self):
        self.prewarning_windows.clear()

    def _order_group(self, step_dict):
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

    def _requires_hand_release(self, step_dict):
        return (
            step_dict.get("action_type", "spatial") == "spatial"
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

    def _prewarning_targets(self, step_dict):
        configured = str(step_dict.get("prewarning_target", "") or "").strip()
        text = configured or step_dict.get("text", "")
        return self.base_logic.parse_step_text(text)

    def _prewarning_padding(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("prewarning_padding_ratio", 0.35)))
        except (TypeError, ValueError):
            return 0.35

    def _prewarning_window_config(self, step_dict):
        try:
            window_frames = max(1, int(step_dict.get("prewarning_window_frames", 5)))
        except (TypeError, ValueError):
            window_frames = 5
        try:
            hit_frames = max(1, int(step_dict.get("prewarning_hit_frames", 3)))
        except (TypeError, ValueError):
            hit_frames = 3
        return min(hit_frames, window_frames), window_frames

    def _step_required_count(self, step_dict):
        try:
            return max(1, int(step_dict.get("count", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def _step_cooldown_seconds(self, step_dict):
        try:
            return max(0.0, float(step_dict.get("cooldown", 1.5) or 0.0))
        except (TypeError, ValueError):
            return 1.5

    def _hand_touch_active(self, engine, step_dict, targets, detections, difficulty):
        _, padding_ratio, _ = engine._evaluation_params(
            difficulty, None, self._step_padding_ratio(step_dict)
        )
        return engine.hand_touch_relation_met(targets, detections, padding_ratio)

    def _is_assembly_step(self, step_dict, targets=None):
        if step_dict.get("action_type", "spatial") != "spatial":
            return False
        if targets is None:
            targets = self._targets_for_step(step_dict)
        return self.base_logic.is_assembly_like_step(targets, step_dict.get("text", ""))

    def _targets_for_step(self, step_dict):
        if self._is_detach_step(step_dict):
            configured = []
            for key in ("detach_removed", "detach_base"):
                value = str(step_dict.get(key, "")).strip()
                if value:
                    configured.extend(self.base_logic.parse_step_text(value))
            if len(configured) >= 2:
                unique = []
                for target in configured:
                    if target not in unique:
                        unique.append(target)
                return unique[:2]
        return self.base_logic.parse_step_text(step_dict.get("text", ""))

    def _target_option_set(self, targets):
        options = set()
        for target in targets or []:
            options.update(self.base_logic.target_options(target))
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

    def _prereqs_satisfied(self, step_dict, completed_step_indices):
        prereqs = self._prerequisite_indices(step_dict)
        if not prereqs:
            return True
        completed = completed_step_indices or set()
        return all(idx in completed for idx in prereqs)

    def _prerequisite_mode(self, step_dict):
        mode = str(step_dict.get("prerequisite_mode", "alarm_only") or "alarm_only")
        if mode not in ("block_and_alarm", "alarm_only", "block_only"):
            return "alarm_only"
        return mode

    def _detach_prereq_satisfied(self, step_idx, all_steps, completed_step_indices):
        detach_targets = self._targets_for_step(all_steps[step_idx])
        detach_options = self._target_option_set(detach_targets)
        if len(detach_options) < 2:
            return True

        matched_prior_assembly = None
        for prior_idx in range(step_idx - 1, -1, -1):
            prior_step = all_steps[prior_idx]
            prior_targets = self._targets_for_step(prior_step)
            if not self._is_assembly_step(prior_step, prior_targets):
                continue
            prior_options = self._target_option_set(prior_targets)
            if detach_options.issubset(prior_options):
                matched_prior_assembly = prior_idx
                break

        if matched_prior_assembly is None:
            return True
        return matched_prior_assembly in (completed_step_indices or set())

    def _has_non_hand_intersections(self, current_detections):
        boxes = []
        for det in current_detections:
            if self.base_logic.is_hand_class(det.get('class', '')):
                continue
            boxes.append(det.get('bbox'))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i] and boxes[j] and self.base_logic.check_intersection(boxes[i], boxes[j]):
                    return True
        return False

    def _is_idle_frame(self, current_detections, held_objects):
        if held_objects:
            return False
        if any(self.base_logic.is_hand_class(det.get('class', '')) for det in current_detections):
            return False
        return not self._has_non_hand_intersections(current_detections)

    def _update_restart_guard(self, current_detections, held_objects):
        if not self.restart_guard_active:
            return False
        self.restart_guard_frames -= 1
        if self._is_idle_frame(current_detections, held_objects):
            self.restart_guard_idle_frames += 1
        else:
            self.restart_guard_idle_frames = 0

        if self.restart_guard_idle_frames >= 3 or self.restart_guard_frames <= 0:
            self.restart_guard_active = False
            self.restart_guard_frames = 0
            self.restart_guard_idle_frames = 0
            return False
        return True

    def _scope_limit(self):
        scope = str(self.monitor_scope or "next_2")
        if scope in ("disabled", "off", "none", "0"):
            return 0
        if scope in ("next_1", "1"):
            return 1
        if scope in ("next_2", "2"):
            return 2
        return None

    def _future_step_indices(self, all_steps, current_idx, current_group):
        limit = self._scope_limit()
        if limit == 0:
            return []

        indices = []
        logical_seen = 0
        i = current_idx + 1
        while i < len(all_steps):
            step_dict = all_steps[i]
            if current_group and self._order_group(step_dict) == current_group:
                i += 1
                continue

            group = self._order_group(step_dict)
            if group:
                j = i
                while j < len(all_steps) and self._order_group(all_steps[j]) == group:
                    indices.append(j)
                    j += 1
                i = j
            else:
                indices.append(i)
                i += 1

            logical_seen += 1
            if limit is not None and logical_seen >= limit:
                break
        return indices

    def _guarded_step_indices(self, all_steps, current_idx, completed_step_indices):
        indices = []
        completed = completed_step_indices or set()
        # 只有显式开启“硬监测”的步骤才越过全局跳步范围。
        # 包含当前步骤，以便乱序组违规推进后仍能识别剩余成员的前置违规动作。
        for i in range(current_idx, len(all_steps)):
            step_dict = all_steps[i]
            if not bool(step_dict.get("prerequisite_hard_monitor", False)):
                continue
            if self._prerequisite_mode(step_dict) == "block_only":
                continue
            prereqs = self._prerequisite_indices(step_dict)
            if prereqs and not all(idx in completed for idx in prereqs):
                indices.append(i)
        return indices

    def check_prewarning(self, current_detections, all_steps, current_idx,
                         completed_step_indices=None):
        """Warn on early hand approach without advancing workflow or recording NG."""
        if not all_steps or current_idx >= len(all_steps):
            self.prewarning_windows.clear()
            return False, "", -1

        completed = completed_step_indices or set()
        eligible_indices = set()
        warning_result = None
        for i in range(current_idx, len(all_steps)):
            step_dict = all_steps[i]
            if not bool(step_dict.get("prewarning_enabled", False)):
                continue

            prereqs = self._prerequisite_indices(step_dict)
            missing_prereqs = [idx for idx in prereqs if idx not in completed]
            if not missing_prereqs:
                continue

            targets = self._prewarning_targets(step_dict)
            if not targets:
                continue

            eligible_indices.add(i)
            hit_frames, window_frames = self._prewarning_window_config(step_dict)
            history = self.prewarning_windows.get(i)
            if history is None or history.maxlen != window_frames:
                previous = list(history or [])[-window_frames:]
                history = deque(previous, maxlen=window_frames)
                self.prewarning_windows[i] = history

            touched = self.base_logic.hand_touch_relation_met(
                targets, current_detections, self._prewarning_padding(step_dict)
            )
            history.append(1 if touched else 0)
            if sum(history) < hit_frames or warning_result is not None:
                continue

            target_names = [
                self.base_logic.target_display_name(target, self.base_logic.eng_to_zh)
                for target in targets
                if not self.base_logic.target_is_hand(target)
            ]
            target_text = "、".join(name for name in target_names if name) or "预警目标"
            missing_text = "、".join(str(idx + 1) for idx in missing_prereqs)
            message = (
                f"⚠️ 提前预警：步骤 {i + 1} 的前置步骤 {missing_text} 尚未完成，"
                f"禁止手接近/触达：【{target_text}】"
            )
            warning_result = (True, message, i)

        for idx in list(self.prewarning_windows):
            if idx not in eligible_indices:
                self.prewarning_windows.pop(idx, None)
        return warning_result or (False, "", -1)

    def check_jump_by_completion(self, current_detections, all_steps, current_idx, held_objects,
                                 completed_step_indices=None):
        now = time.time()

        if self._update_restart_guard(current_detections, held_objects):
            return False, "", -1

        if now < self.alarm_expiry_time:
            return True, self.alarm_message, -1

        if not all_steps or current_idx >= len(all_steps):
            return False, "", -1

        current_step = all_steps[current_idx]
        current_group = self._order_group(current_step)
        current_targets = self._targets_for_step(current_step)
        current_target_options = set()
        for target in current_targets:
            current_target_options.update(self.base_logic.target_options(target))

        # 遍历配置范围内的未来步骤，默认只看后 2 步，避免全局乱扫桌面物品。
        # 另外，开启“前置硬监测”且前置未完成的重点步骤会额外被监控，不受范围限制。
        monitored_indices = []
        guarded_indices = self._guarded_step_indices(
            all_steps, current_idx, completed_step_indices
        )
        # 当前步骤不属于“未来范围”，但只要模式包含报警，就必须能识别实际违规动作。
        current_prereqs = self._prerequisite_indices(current_step)
        current_missing_prereqs = [
            idx for idx in current_prereqs
            if idx not in (completed_step_indices or set())
        ]
        current_mode = self._prerequisite_mode(current_step)
        if current_missing_prereqs and current_mode != "block_only":
            monitored_indices.append(current_idx)
        # 当前步骤的前置违规优先处理，再处理正常范围和更远的前置守卫步骤。
        if current_idx in guarded_indices:
            if current_idx not in monitored_indices:
                monitored_indices.append(current_idx)
        for i in self._future_step_indices(all_steps, current_idx, current_group):
            if i not in monitored_indices:
                monitored_indices.append(i)
        for i in guarded_indices:
            if i not in monitored_indices:
                monitored_indices.append(i)

        for i in monitored_indices:
            step_dict = all_steps[i]

            future_targets = self._targets_for_step(step_dict)

            # 普通空间步骤只有一个目标时，单靠“物品出现”容易误报；手/手套触达步骤允许单目标跳步监控。
            if not future_targets:
                continue
            if len(future_targets) < 2 and not self._is_hand_touch_step(step_dict):
                continue

            # 如果未来需要的物品，现在本来就需要，就不算跳步
            if i != current_idx and all(
                any(option in current_target_options for option in self.base_logic.target_options(target))
                for target in future_targets
            ):
                continue

            explicit_prereqs = self._prerequisite_indices(step_dict)
            explicit_prereq_violation = bool(
                explicit_prereqs
                and not self._prereqs_satisfied(step_dict, completed_step_indices)
            )
            prerequisite_mode = self._prerequisite_mode(step_dict)
            if explicit_prereq_violation and prerequisite_mode == "block_only":
                if i in self.shadow_engines:
                    self.shadow_engines[i].reset()
                continue
            # 显式“前置步骤”是最高优先级的硬门禁。即使这是拆除工序，且其对应的
            # 历史装配步骤尚未在本轮记录为完成，也必须继续观察真实的拆除动作，
            # 否则会漏掉“未完成前置条件就直接拆除”的跳步报警。
            if (self._is_detach_step(step_dict)
                    and not explicit_prereq_violation
                    and not self._detach_prereq_satisfied(i, all_steps, completed_step_indices)):
                if i in self.shadow_engines:
                    self.shadow_engines[i].reset()
                continue

            # 🌟 核心逻辑：影子状态机每帧都观察未来步骤，这样它能分清“开局就满足”
            # 和“先不满足、后面由动作变成满足”。
            if i not in self.shadow_engines:
                self.shadow_engines[i] = ProcessLogicEngine()
                # 复制字典映射
                self.shadow_engines[i].lookup_dict = self.base_logic.lookup_dict
                self.shadow_engines[i].eng_to_zh = self.base_logic.eng_to_zh
                self.shadow_engines[i].regex_pattern = self.base_logic.regex_pattern

            # 👇 【关键修改 3】动态读取你配置的真实难度和次数，而不是定死为“中等”
            future_diff = step_dict.get("difficulty", "中等 (标准) 🟡")
            future_count = self._step_required_count(step_dict)
            future_strat = step_dict.get("multi_strategy", "lock")

            # 手触达多次必须是真正分开的多次动作，持续按住不能重复计数。
            if self._is_hand_touch_step(step_dict) and i in self.shadow_wait_release:
                if self._hand_touch_active(
                        self.shadow_engines[i], step_dict, future_targets,
                        current_detections, future_diff):
                    continue
                self.shadow_wait_release.discard(i)
                self.shadow_engines[i].reset(clear_blind_zones=False)
                continue

            # time 策略与主流程一样，在两次动作之间执行配置的冷却时间。
            if now < self.shadow_cooldown_until.get(i, 0.0):
                continue

            # 让影子引擎像主引擎一样，严谨地去计算有没有发生空间交叉！
            if self._is_hand_touch_step(step_dict):
                is_completed, _ = self.shadow_engines[i].evaluate_hand_touch_step(
                    future_targets, current_detections, future_diff,
                    stable_frames_override=self._step_stable_frames(step_dict),
                    padding_ratio_override=self._step_padding_ratio(step_dict),
                )
            elif self._is_detach_step(step_dict):
                is_completed, _ = self.shadow_engines[i].evaluate_detach_step(
                    future_targets, current_detections, future_diff, assembly_ready=True,
                    stable_frames_override=self._step_stable_frames(step_dict),
                    padding_ratio_override=self._step_padding_ratio(step_dict),
                )
            else:
                action_confirm_frames = self._step_action_confirm_frames(step_dict)
                gate_enabled = self.strong_action_enabled or (
                    self.ignore_static_intersection and action_confirm_frames > 0
                )
                is_completed, _ = self.shadow_engines[i].evaluate_step(
                    future_targets, current_detections, future_diff, future_count, future_strat,
                    action_gate_enabled=gate_enabled,
                    action_touch_frames=(
                        self.strong_action_frames if self.strong_action_enabled
                        else action_confirm_frames
                    ),
                    action_touch_only=self.strong_action_enabled,
                    stable_frames_override=self._step_stable_frames(step_dict),
                    padding_ratio_override=self._step_padding_ratio(step_dict),
                    require_hand_release=self._requires_hand_release(step_dict),
                    hand_release_padding=self._hand_release_padding(step_dict),
                    hand_release_frames=self._hand_release_frames(step_dict),
                )

            if is_completed:
                completed_count = self.shadow_sub_counts.get(i, 0) + 1
                if completed_count < future_count:
                    self.shadow_sub_counts[i] = completed_count
                    if self._is_hand_touch_step(step_dict):
                        self.shadow_wait_release.add(i)
                    if "time" in str(future_strat).lower():
                        self.shadow_cooldown_until[i] = now + self._step_cooldown_seconds(step_dict)
                    preserve_blind_zones = "lock" in str(future_strat).lower()
                    self.shadow_engines[i].reset(clear_blind_zones=not preserve_blind_zones)
                    continue

                # 触发跳步！重置该影子状态机，并写入 2.0 秒的锁存时间
                self.shadow_engines[i].reset()
                self.shadow_sub_counts.pop(i, None)
                self.shadow_cooldown_until.pop(i, None)
                self.shadow_wait_release.discard(i)
                zh_names = [
                    self.base_logic.target_display_name(t, self.base_logic.eng_to_zh)
                    for t in future_targets
                ]
                prereqs = self._prerequisite_indices(step_dict)
                if prereqs and not self._prereqs_satisfied(step_dict, completed_step_indices):
                    prereq_text = "、".join(str(idx + 1) for idx in prereqs)
                    if prerequisite_mode == "block_and_alarm":
                        self.alarm_message = (
                            f"🚫 前置条件未完成！步骤 {i + 1} 必须先完成步骤 {prereq_text}，"
                            f"已阻止提前执行：【{' 与 '.join(zh_names)}】"
                        )
                    else:
                        self.alarm_message = (
                            f"🚫 前置条件未完成！步骤 {i + 1} 必须先完成步骤 {prereq_text}，"
                            f"已记录违规并继续流程：【{' 与 '.join(zh_names)}】"
                        )
                else:
                    self.alarm_message = f"🚫 严重跳步！必须先完成当前步骤，再执行步骤 {i + 1}：【{' 与 '.join(zh_names)}】"
                self.alarm_expiry_time = now + 2.0

                # 返回 True, 报警信息, 以及跳到了哪一步
                return True, self.alarm_message, i

        return False, "", -1
