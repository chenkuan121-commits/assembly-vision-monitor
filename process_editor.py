import os
import json
import re
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QDialog, QListWidget, QTextEdit, QMessageBox,
                               QComboBox, QGroupBox, QAbstractItemView, QSpinBox, QDoubleSpinBox, QInputDialog,
                               QLineEdit)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont

DEFAULT_PROFILE_NAME = "默认方案"
DEFAULT_STEP_TIMEOUT = 300


def normalize_profile_config(config_data):
    """Ensure old config files always expose at least one profile."""
    root_profile = {
        "process_steps": config_data.get("process_steps", []),
        "forbidden_items": config_data.get("forbidden_items", ""),
        "step_timeout": config_data.get("step_timeout", DEFAULT_STEP_TIMEOUT),
    }
    profiles = config_data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        profiles = {
            DEFAULT_PROFILE_NAME: root_profile
        }
        config_data["profiles"] = profiles
    elif DEFAULT_PROFILE_NAME not in profiles:
        profiles[DEFAULT_PROFILE_NAME] = root_profile
    elif root_profile["process_steps"] and not profiles[DEFAULT_PROFILE_NAME].get("process_steps"):
        profiles[DEFAULT_PROFILE_NAME] = root_profile

    for profile in profiles.values():
        profile.setdefault("process_steps", [])
        profile.setdefault("forbidden_items", "")
        profile.setdefault("step_timeout", DEFAULT_STEP_TIMEOUT)

    active = config_data.get("active_profile")
    if active not in profiles:
        active = DEFAULT_PROFILE_NAME if DEFAULT_PROFILE_NAME in profiles else next(iter(profiles))
    config_data["active_profile"] = active
    return config_data


class KeywordHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None, color_hex="#0f9d58"):
        super().__init__(parent)
        self.pattern = None
        self.highlight_format = QTextCharFormat()
        self.highlight_format.setForeground(QColor(color_hex))
        self.highlight_format.setFontWeight(QFont.Bold)

    def update_dict(self, lookup_dict):
        if not lookup_dict:
            self.pattern = None
            self.rehighlight()
            return
        all_terms = sorted(lookup_dict.keys(), key=len, reverse=True)
        escaped_terms = [re.escape(term) for term in all_terms]
        self.pattern = re.compile('|'.join(escaped_terms), re.IGNORECASE)
        self.rehighlight()

    def highlightBlock(self, text):
        if not self.pattern: return
        for match in self.pattern.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self.highlight_format)

class ProcessGuideDialog(QDialog):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📝 智能工序与安全配置大厅 (支持多套方案管理)")
        self.resize(950, 700)
        self.config_path = config_path
        self.config_data = {}
        self.lookup_dict = {}

        self.active_profile_name = "默认方案"

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # 🌟 多套方案管理目录
        profile_group = QGroupBox("📁 工序方案目录")
        p_layout = QHBoxLayout()
        self.combo_profiles = QComboBox()
        self.combo_profiles.currentTextChanged.connect(self.switch_profile)
        self.btn_add_profile = QPushButton("➕ 新建")
        self.btn_del_profile = QPushButton("🗑️ 删除")
        self.btn_add_profile.clicked.connect(self.add_profile)
        self.btn_del_profile.clicked.connect(self.del_profile)
        p_layout.addWidget(self.combo_profiles, stretch=2)
        p_layout.addWidget(self.btn_add_profile)
        p_layout.addWidget(self.btn_del_profile)
        profile_group.setLayout(p_layout)
        left_layout.addWidget(profile_group)

        # 步骤列表
        self.step_list = QListWidget()
        self.step_list.currentRowChanged.connect(self.on_step_selected)
        self.step_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.step_list.model().rowsMoved.connect(self.renumber_steps)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("➕ 添加步骤")
        self.btn_del = QPushButton("🗑️ 删除步骤")
        self.btn_add.clicked.connect(self.add_step)
        self.btn_del.clicked.connect(self.del_step)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)

        left_layout.addWidget(QLabel("📋 当前方案步骤 (长按可拖拽排序)："))
        left_layout.addWidget(self.step_list)
        left_layout.addLayout(btn_layout)
        main_layout.addLayout(left_layout, stretch=3)

        right_layout = QVBoxLayout()

        global_group = QGroupBox("🛡️ 全局安全合规配置 (当前方案)")
        global_group.setStyleSheet("QGroupBox { font-weight: bold; color: #d93025; }")
        g_layout = QVBoxLayout()
        g_layout.addWidget(QLabel("🚫 画面绝对禁止出现 (逗号分隔):"))
        self.forbidden_input = QTextEdit()
        self.forbidden_input.setFixedHeight(35)
        self.forbidden_input.textChanged.connect(self.save_current_step)
        self.forbidden_highlighter = KeywordHighlighter(self.forbidden_input.document(), color_hex="#d93025")
        g_layout.addWidget(self.forbidden_input)

        # 🌟 新增：全局超时时间配置
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("⏱️ 单步操作超时警告 (秒):"))
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(5, 300)  # 5秒到5分钟
        self.spin_timeout.setValue(DEFAULT_STEP_TIMEOUT)
        self.spin_timeout.valueChanged.connect(self.save_current_step)
        timeout_layout.addWidget(self.spin_timeout)
        timeout_layout.addStretch()
        g_layout.addLayout(timeout_layout)

        global_group.setLayout(g_layout)
        right_layout.addWidget(global_group)

        step_group = QGroupBox("✍️ 当前步骤细节配置")
        s_layout = QVBoxLayout()

        diff_layout = QHBoxLayout()
        diff_layout.addWidget(QLabel("🎯 判定难度:"))
        self.combo_difficulty = QComboBox()
        self.combo_difficulty.addItems(["简单 (宽松) 🟢", "中等 (标准) 🟡", "困难 (严苛) 🔴"])
        self.combo_difficulty.currentIndexChanged.connect(self.save_current_step)
        diff_layout.addWidget(self.combo_difficulty)

        diff_layout.addWidget(QLabel("   🔁 执行次数:"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 99)
        self.spin_count.valueChanged.connect(self.toggle_strategy_ui)
        diff_layout.addWidget(self.spin_count)
        diff_layout.addStretch()
        s_layout.addLayout(diff_layout)

        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("判定方式:"))
        self.combo_action_type = QComboBox()
        self.combo_action_type.addItems(["空间相交/装配", "手/手套触达目标", "拆除/分离两个物品"])
        self.combo_action_type.currentIndexChanged.connect(self.on_action_type_changed)
        action_layout.addWidget(self.combo_action_type)

        action_layout.addWidget(QLabel("   乱序组:"))
        self.input_order_group = QLineEdit()
        self.input_order_group.setPlaceholderText("同组编号相同，如 A；留空=固定顺序")
        self.input_order_group.textChanged.connect(self.save_current_step)
        action_layout.addWidget(self.input_order_group)
        action_layout.addStretch()
        s_layout.addLayout(action_layout)

        self.detach_layout = QHBoxLayout()
        self.detach_layout.addWidget(QLabel("拆除物:"))
        self.input_detach_removed = QLineEdit()
        self.input_detach_removed.setPlaceholderText("如 白色线缆 / white_cable")
        self.input_detach_removed.textChanged.connect(self.save_current_step)
        self.detach_layout.addWidget(self.input_detach_removed)
        self.detach_layout.addWidget(QLabel("基准物:"))
        self.input_detach_base = QLineEdit()
        self.input_detach_base.setPlaceholderText("如 板子 / board")
        self.input_detach_base.textChanged.connect(self.save_current_step)
        self.detach_layout.addWidget(self.input_detach_base)
        s_layout.addLayout(self.detach_layout)

        # 重复策略与冷却时间 (只在 count > 1 时显示)
        self.strategy_layout = QHBoxLayout()
        self.strategy_layout.addWidget(QLabel("⚙️ 重复策略:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["lock (空间锁定/盲区)", "time (时间间隔/冷却)"])
        self.combo_strategy.currentIndexChanged.connect(self.toggle_strategy_ui)
        self.strategy_layout.addWidget(self.combo_strategy)

        self.lbl_cooldown = QLabel("   ⏱️ 间隔时间:")
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(1, 60)
        self.spin_cooldown.setSingleStep(0.1)
        self.spin_cooldown.setSuffix(" 秒")
        self.spin_cooldown.setValue(1.5)
        self.spin_cooldown.valueChanged.connect(self.save_current_step)

        self.strategy_layout.addWidget(self.lbl_cooldown)
        self.strategy_layout.addWidget(self.spin_cooldown)
        self.strategy_layout.addStretch()
        s_layout.addLayout(self.strategy_layout)

        s_layout.addWidget(QLabel("📝 步骤指令描述 (输入已知目标自动标绿)："))
        self.text_editor = QTextEdit()
        self.text_editor.setStyleSheet("font-size: 14px; line-height: 1.5;")
        self.text_editor.textChanged.connect(self.save_current_step)
        self.step_highlighter = KeywordHighlighter(self.text_editor.document(), color_hex="#0f9d58")
        s_layout.addWidget(self.text_editor)
        step_group.setLayout(s_layout)
        right_layout.addWidget(step_group)

        self.btn_save_all = QPushButton("💾 保存所有方案配置")
        self.btn_save_all.setMinimumHeight(45)
        self.btn_save_all.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold; font-size: 14px;")
        self.btn_save_all.clicked.connect(self.save_to_json)
        right_layout.addWidget(self.btn_save_all)

        main_layout.addLayout(right_layout, stretch=5)
        self.setLayout(main_layout)
        self._update_detach_ui_visibility()

    def load_data(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)

            mapping = self.config_data.get("mapping", {})
            for _, info in mapping.items():
                if info["zh_name"]: self.lookup_dict[info["zh_name"].lower()] = info["eng_name"]
                if info["eng_name"]: self.lookup_dict[info["eng_name"].lower()] = info["eng_name"]

            self.step_highlighter.update_dict(self.lookup_dict)
            self.forbidden_highlighter.update_dict(self.lookup_dict)

            normalize_profile_config(self.config_data)
            self.active_profile_name = self.config_data["active_profile"]

            self.combo_profiles.blockSignals(True)
            self.combo_profiles.clear()
            self.combo_profiles.addItems(self.config_data["profiles"].keys())
            self.combo_profiles.setCurrentText(self.active_profile_name)
            self.combo_profiles.blockSignals(False)

            self.refresh_list_for_profile()

    def add_profile(self):
        name, ok = QInputDialog.getText(self, "新建工序方案", "请输入新方案名称:")
        if ok and name.strip():
            name = name.strip()
            if name in self.config_data["profiles"]:
                QMessageBox.warning(self, "错误", "方案名已存在！")
                return
            self.config_data["profiles"][name] = {"process_steps": [], "forbidden_items": "", "step_timeout": DEFAULT_STEP_TIMEOUT}
            self.combo_profiles.addItem(name)
            self.combo_profiles.setCurrentText(name)

    def del_profile(self):
        if self.combo_profiles.count() <= 1:
            QMessageBox.warning(self, "警告", "必须保留至少一个方案！")
            return
        name = self.combo_profiles.currentText()
        reply = QMessageBox.question(self, '确认删除', f"确定删除方案【{name}】吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.config_data["profiles"][name]
            self.combo_profiles.removeItem(self.combo_profiles.currentIndex())

    def switch_profile(self, profile_name):
        if profile_name and profile_name in self.config_data.get("profiles", {}):
            self.active_profile_name = profile_name
            self.refresh_list_for_profile()

    def refresh_list_for_profile(self):
        self.step_list.blockSignals(True)
        self.step_list.clear()
        profile_data = self.config_data["profiles"].get(self.active_profile_name, {})

        self.forbidden_input.blockSignals(True)
        self.forbidden_input.setPlainText(profile_data.get("forbidden_items", ""))
        self.forbidden_input.blockSignals(False)

        self.spin_timeout.blockSignals(True)
        self.spin_timeout.setValue(profile_data.get("step_timeout", DEFAULT_STEP_TIMEOUT))
        self.spin_timeout.blockSignals(False)

        for step_data in profile_data.get("process_steps", []):
            if isinstance(step_data, str): step_data = {"text": step_data}
            if "difficulty" not in step_data: step_data["difficulty"] = "中等 (标准) 🟡"
            if "count" not in step_data: step_data["count"] = 1
            if "multi_strategy" not in step_data: step_data["multi_strategy"] = "lock"
            if "cooldown" not in step_data: step_data["cooldown"] = 1.5
            if "action_type" not in step_data: step_data["action_type"] = "spatial"
            if "order_group" not in step_data: step_data["order_group"] = ""
            if "detach_removed" not in step_data: step_data["detach_removed"] = ""
            if "detach_base" not in step_data: step_data["detach_base"] = ""

            item = f"步骤 {self.step_list.count() + 1}"
            self.step_list.addItem(item)
            self.step_list.item(self.step_list.count() - 1).setData(Qt.UserRole, step_data)

        self.step_list.blockSignals(False)
        if self.step_list.count() > 0:
            self.step_list.setCurrentRow(0)
        else:
            self.clear_right_panel()

    def clear_right_panel(self):
        self.text_editor.blockSignals(True)
        self.spin_count.blockSignals(True)
        self.combo_difficulty.blockSignals(True)
        self.combo_strategy.blockSignals(True)
        self.spin_cooldown.blockSignals(True)
        self.combo_action_type.blockSignals(True)
        self.input_order_group.blockSignals(True)
        self.input_detach_removed.blockSignals(True)
        self.input_detach_base.blockSignals(True)
        self.text_editor.clear()
        self.spin_count.setValue(1)
        self.combo_difficulty.setCurrentIndex(1)
        self.combo_strategy.setCurrentIndex(0)
        self.spin_cooldown.setValue(1.5)
        self.combo_action_type.setCurrentIndex(0)
        self.input_order_group.clear()
        self.input_detach_removed.clear()
        self.input_detach_base.clear()
        self.text_editor.blockSignals(False)
        self.spin_count.blockSignals(False)
        self.combo_difficulty.blockSignals(False)
        self.combo_strategy.blockSignals(False)
        self.spin_cooldown.blockSignals(False)
        self.combo_action_type.blockSignals(False)
        self.input_order_group.blockSignals(False)
        self.input_detach_removed.blockSignals(False)
        self.input_detach_base.blockSignals(False)
        self.toggle_strategy_ui()

    def _sync_profile_from_list(self):
        if self.active_profile_name not in self.config_data.get("profiles", {}):
            return
        steps_data = []
        for i in range(self.step_list.count()):
            data = self.step_list.item(i).data(Qt.UserRole)
            if data is not None:
                steps_data.append(data)
        profile = self.config_data["profiles"][self.active_profile_name]
        profile["process_steps"] = steps_data
        profile["forbidden_items"] = self.forbidden_input.toPlainText().strip()
        profile["step_timeout"] = self.spin_timeout.value()

    def renumber_steps(self, parent, start, end, destination, row):
        for i in range(self.step_list.count()):
            self.step_list.item(i).setText(f"步骤 {i + 1}")
        self._sync_profile_from_list()
        self.on_step_selected(self.step_list.currentRow())

    def toggle_strategy_ui(self):
        count = self.spin_count.value()
        # 如果次数 > 1，显示策略选择框
        is_multi = count > 1
        self.combo_strategy.setVisible(is_multi)
        self.strategy_layout.itemAt(0).widget().setVisible(is_multi)

        # 如果选了“时间间隔”，才显示冷却时间设置
        is_time_strategy = "time" in self.combo_strategy.currentText()
        show_cooldown = is_multi and is_time_strategy
        self.lbl_cooldown.setVisible(show_cooldown)
        self.spin_cooldown.setVisible(show_cooldown)
        self._update_detach_ui_visibility()

        self.save_current_step()

    def _set_layout_visible(self, layout, visible):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(visible)

    def _update_detach_ui_visibility(self):
        self._set_layout_visible(self.detach_layout, self.combo_action_type.currentIndex() == 2)

    def on_action_type_changed(self):
        self._update_detach_ui_visibility()
        self.save_current_step()

    def on_step_selected(self, index):
        if index >= 0:
            item = self.step_list.item(index)
            step_data = item.data(Qt.UserRole) or {"text": "", "difficulty": "中等 (标准) 🟡", "count": 1,
                                                   "multi_strategy": "lock", "cooldown": 1.5,
                                                   "action_type": "spatial", "order_group": ""}

            self.text_editor.blockSignals(True)
            self.combo_difficulty.blockSignals(True)
            self.spin_count.blockSignals(True)
            self.combo_strategy.blockSignals(True)
            self.spin_cooldown.blockSignals(True)
            self.combo_action_type.blockSignals(True)
            self.input_order_group.blockSignals(True)
            self.input_detach_removed.blockSignals(True)
            self.input_detach_base.blockSignals(True)

            self.text_editor.setPlainText(step_data.get("text", ""))
            self.combo_difficulty.setCurrentText(step_data.get("difficulty", "中等 (标准) 🟡"))
            self.spin_count.setValue(step_data.get("count", 1))
            action_type = step_data.get("action_type", "spatial")
            action_idx = 2 if action_type == "detach" else (1 if action_type == "hand_touch" else 0)
            self.combo_action_type.setCurrentIndex(action_idx)
            self.input_order_group.setText(step_data.get("order_group", ""))
            self.input_detach_removed.setText(step_data.get("detach_removed", ""))
            self.input_detach_base.setText(step_data.get("detach_base", ""))

            strat = step_data.get("multi_strategy", "lock")
            idx = 1 if "time" in strat else 0
            self.combo_strategy.setCurrentIndex(idx)
            self.spin_cooldown.setValue(step_data.get("cooldown", 1.5))

            self.text_editor.blockSignals(False)
            self.combo_difficulty.blockSignals(False)
            self.spin_count.blockSignals(False)
            self.combo_strategy.blockSignals(False)
            self.spin_cooldown.blockSignals(False)
            self.combo_action_type.blockSignals(False)
            self.input_order_group.blockSignals(False)
            self.input_detach_removed.blockSignals(False)
            self.input_detach_base.blockSignals(False)

            self.toggle_strategy_ui()

    def save_current_step(self):
        row = self.step_list.currentRow()
        if row >= 0:
            item = self.step_list.item(row)
            strat = "time" if self.combo_strategy.currentIndex() == 1 else "lock"
            existing_data = item.data(Qt.UserRole) or {}
            action_type = "spatial"
            if self.combo_action_type.currentIndex() == 1:
                action_type = "hand_touch"
            elif self.combo_action_type.currentIndex() == 2:
                action_type = "detach"

            step_data = {
                "text": self.text_editor.toPlainText(),
                "difficulty": self.combo_difficulty.currentText(),
                "count": self.spin_count.value(),
                "multi_strategy": strat,
                "cooldown": self.spin_cooldown.value(),
                "action_type": action_type,
                "order_group": self.input_order_group.text().strip(),
                "detach_removed": self.input_detach_removed.text().strip(),
                "detach_base": self.input_detach_base.text().strip()
            }
            # 保留编辑器不管理的扩展字段 (如 aoi_feature_check)
            for key in existing_data:
                if key not in step_data:
                    step_data[key] = existing_data[key]
            item.setData(Qt.UserRole, step_data)

        # 实时同步到当前 Profile 数据中；即使没有步骤，也要保存全局违禁项和超时时间
        self._sync_profile_from_list()

    def add_step(self):
        new_idx = self.step_list.count() + 1
        self.step_list.addItem(f"步骤 {new_idx}")
        self.step_list.item(new_idx - 1).setData(Qt.UserRole, {"text": "", "difficulty": "中等 (标准) 🟡", "count": 1,
                                                                "multi_strategy": "lock", "cooldown": 1.5,
                                                                "action_type": "spatial", "order_group": "",
                                                                "detach_removed": "", "detach_base": ""})
        self.step_list.setCurrentRow(new_idx - 1)
        self.text_editor.setFocus()

    def del_step(self):
        row = self.step_list.currentRow()
        if row >= 0:
            item = self.step_list.takeItem(row)
            del item
            for i in range(self.step_list.count()):
                self.step_list.item(i).setText(f"步骤 {i + 1}")
            self._sync_profile_from_list()
            if self.step_list.count() > 0:
                self.step_list.setCurrentRow(min(row, self.step_list.count() - 1))
            else:
                self.clear_right_panel()

    def save_to_json(self):
        self.save_current_step()  # 确保最后修改保存了
        self.config_data["active_profile"] = self.active_profile_name

        # 兼容旧的主界面逻辑：把选中的方案，覆盖到根节点，这样 main_tester 不用改太多代码！
        profile = self.config_data["profiles"][self.active_profile_name]
        self.config_data["process_steps"] = profile["process_steps"]
        self.config_data["forbidden_items"] = profile["forbidden_items"]
        self.config_data["step_timeout"] = profile.get("step_timeout", DEFAULT_STEP_TIMEOUT)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=4)

        QMessageBox.information(self, "成功", f"方案【{self.active_profile_name}】及所有配置已成功保存！")
        self.accept()
