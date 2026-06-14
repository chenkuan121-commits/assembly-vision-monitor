import time
import copy
from logic_engine import ProcessLogicEngine


class WorkflowMonitor:
    def __init__(self, base_logic_engine):
        self.base_logic = base_logic_engine
        self.shadow_engines = {}  # 存放未来步骤的独立进度计算器
        self.alarm_message = ""
        self.alarm_expiry_time = 0.0

    def _order_group(self, step_dict):
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
                    configured.extend(self.base_logic.parse_step_text(value))
            if len(configured) >= 2:
                unique = []
                for target in configured:
                    if target not in unique:
                        unique.append(target)
                return unique[:2]
        return self.base_logic.parse_step_text(step_dict.get("text", ""))

    def check_jump_by_completion(self, current_detections, all_steps, current_idx, held_objects):
        now = time.time()

        if now < self.alarm_expiry_time:
            return True, self.alarm_message, -1

        if not all_steps or current_idx >= len(all_steps) - 1:
            return False, "", -1

        current_found_boxes = {det['class']: det['bbox'] for det in current_detections}
        current_step = all_steps[current_idx]
        current_group = self._order_group(current_step)
        current_targets = self._targets_for_step(current_step)
        current_target_options = set()
        for target in current_targets:
            current_target_options.update(self.base_logic.target_options(target))

        # 遍历所有未来步骤（从紧邻下一步到最后一步）
        for i in range(current_idx + 1, len(all_steps)):
            step_dict = all_steps[i]
            if current_group and self._order_group(step_dict) == current_group:
                continue

            future_targets = self._targets_for_step(step_dict)

            # 普通空间步骤只有一个目标时，单靠“物品出现”容易误报；手/手套触达步骤允许单目标跳步监控。
            if not future_targets:
                continue
            if len(future_targets) < 2 and not self._is_hand_touch_step(step_dict):
                continue

            # 如果未来需要的物品，现在本来就需要，就不算跳步
            if all(
                any(option in current_target_options for option in self.base_logic.target_options(target))
                for target in future_targets
            ):
                continue

            # 👇 【关键修改 1】引入主引擎的残影记忆，只要物品在视野里或记忆里，就算作存在
            all_present = all(
                any(option in current_found_boxes for option in self.base_logic.target_options(target))
                or (target in self.base_logic.box_memory)
                for target in future_targets
            )

            # 👇 【关键修改 2】极简门槛：只要画面里有手/手套，或者已经触发了 MediaPipe 握持，就放行！
            hand_in_view = any(det['class'] in ['hand', 'glove', '手', '手套'] for det in current_detections)
            is_anyone_held = any(
                option in held_objects
                for target in future_targets
                for option in self.base_logic.target_options(target)
            ) or hand_in_view

            if all_present and is_anyone_held:
                # 🌟 核心逻辑 3：影子状态机完全接管空间判断
                if i not in self.shadow_engines:
                    self.shadow_engines[i] = ProcessLogicEngine()
                    # 复制字典映射
                    self.shadow_engines[i].lookup_dict = self.base_logic.lookup_dict
                    self.shadow_engines[i].eng_to_zh = self.base_logic.eng_to_zh

                # 👇 【关键修改 3】动态读取你配置的真实难度和次数，而不是定死为“中等”
                future_diff = step_dict.get("difficulty", "中等 (标准) 🟡")
                future_count = step_dict.get("count", 1)
                future_strat = step_dict.get("multi_strategy", "lock")

                # 让影子引擎像主引擎一样，严谨地去计算有没有发生空间交叉！
                if self._is_hand_touch_step(step_dict):
                    is_completed, _ = self.shadow_engines[i].evaluate_hand_touch_step(
                        future_targets, current_detections, future_diff
                    )
                elif self._is_detach_step(step_dict):
                    is_completed, _ = self.shadow_engines[i].evaluate_detach_step(
                        future_targets, current_detections, future_diff
                    )
                else:
                    is_completed, _ = self.shadow_engines[i].evaluate_step(
                        future_targets, current_detections, future_diff, future_count, future_strat
                    )

                if is_completed:
                    # 触发跳步！重置该影子状态机，并写入 2.0 秒的锁存时间
                    self.shadow_engines[i].reset()
                    zh_names = [
                        self.base_logic.target_display_name(t, self.base_logic.eng_to_zh)
                        for t in future_targets
                    ]
                    self.alarm_message = f"🚫 严重跳步！必须先完成当前步骤，再执行步骤 {i + 1}：【{' 与 '.join(zh_names)}】"
                    self.alarm_expiry_time = now + 2.0

                    # 返回 True, 报警信息, 以及跳到了哪一步
                    return True, self.alarm_message, i
            else:
                # 如果条件不满足（手拿开了，或者东西放下了），正常扣减跳步进度
                if i in self.shadow_engines:
                    self.shadow_engines[i].hit_counter = max(0, self.shadow_engines[i].hit_counter - 1)

        return False, "", -1
