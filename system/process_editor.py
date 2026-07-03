import os
import json
import re
from PySide6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QDialog, QListWidget, QTextEdit, QMessageBox,
                               QComboBox, QGroupBox, QAbstractItemView, QSpinBox, QDoubleSpinBox, QInputDialog,
                               QLineEdit, QCheckBox, QStyle, QStyleOptionButton, QWidget, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import (QSyntaxHighlighter, QTextCharFormat, QColor, QFont,
                           QPainter, QPainterPath, QPen)

DEFAULT_PROFILE_NAME = "默认方案"
DEFAULT_STEP_TIMEOUT = 300
DEFAULT_JUMP_MONITOR_SCOPE = "next_2"
DEFAULT_JUMP_STRONG_ACTION_ENABLED = False
DEFAULT_JUMP_STRONG_ACTION_FRAMES = 30
DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION = True
DEFAULT_ACTION_CONFIRM_FRAMES = 30
DEFAULT_PREWARNING_PADDING_RATIO = 0.35
DEFAULT_PREWARNING_HIT_FRAMES = 3
DEFAULT_PREWARNING_WINDOW_FRAMES = 5
DEFAULT_HAND_RELEASE_PADDING = 0.15
DEFAULT_HAND_RELEASE_FRAMES = 12
DEFAULT_PREREQUISITE_HARD_MONITOR = False
DEFAULT_PREREQUISITE_MODE = "alarm_only"
DEFAULT_WRONG_PAIR_CONFIRM_FRAMES = 3
DEFAULT_WRONG_PAIR_PADDING_RATIO = 0.10
DIFFICULTY_REFERENCE_TEXT = (
    "📌 难度默认参考：简单＝15 帧 / 外扩 0.20；"
    "中等＝40 帧 / 外扩 0.08；困难＝90 帧 / 外扩 0.00。"
    "自定义连续帧不为 0、框外扩不为 -1 时，会分别覆盖这里的默认值。"
)


class VisibleCheckBox(QCheckBox):
    """QCheckBox with a guaranteed visible tick on styles that omit the native mark."""

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, option, self)
        if indicator.isEmpty():
            return

        color_name = self.property("checkmarkColor") or "#0b57d0"
        color = QColor("#8a94a3") if not self.isEnabled() else QColor(str(color_name))
        pen = QPen(color)
        pen.setWidthF(max(2.0, indicator.width() * 0.14))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        path = QPainterPath()
        path.moveTo(indicator.left() + indicator.width() * 0.20,
                    indicator.top() + indicator.height() * 0.53)
        path.lineTo(indicator.left() + indicator.width() * 0.43,
                    indicator.top() + indicator.height() * 0.76)
        path.lineTo(indicator.left() + indicator.width() * 0.82,
                    indicator.top() + indicator.height() * 0.25)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.drawPath(path)


def normalize_profile_config(config_data):
    """Ensure old config files always expose at least one profile."""
    root_profile = {
        "process_steps": config_data.get("process_steps", []),
        "forbidden_items": config_data.get("forbidden_items", ""),
        "step_timeout": config_data.get("step_timeout", DEFAULT_STEP_TIMEOUT),
        "jump_monitor_scope": config_data.get("jump_monitor_scope", DEFAULT_JUMP_MONITOR_SCOPE),
        "jump_strong_action_enabled": config_data.get(
            "jump_strong_action_enabled", DEFAULT_JUMP_STRONG_ACTION_ENABLED
        ),
        "jump_strong_action_frames": config_data.get(
            "jump_strong_action_frames", DEFAULT_JUMP_STRONG_ACTION_FRAMES
        ),
        "jump_ignore_static_intersection": config_data.get(
            "jump_ignore_static_intersection", DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
        ),
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
        profile.setdefault("jump_monitor_scope", DEFAULT_JUMP_MONITOR_SCOPE)
        profile.setdefault("jump_strong_action_enabled", DEFAULT_JUMP_STRONG_ACTION_ENABLED)
        profile.setdefault("jump_strong_action_frames", DEFAULT_JUMP_STRONG_ACTION_FRAMES)
        profile.setdefault("jump_ignore_static_intersection", DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION)

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


class MappingLineEdit(QLineEdit):
    """A target-name input that makes mapping mistakes immediately visible."""

    VALID_STYLE = (
        "QLineEdit { color:#0b8043; background:#edf7ef; border:1px solid #34a853; "
        "border-radius:4px; padding:3px 6px; font-weight:600; }"
        "QLineEdit:disabled { color:#6b8f76; background:#f2f6f3; border-color:#a8c7b0; }"
    )
    INVALID_STYLE = (
        "QLineEdit { color:#b3261e; background:#fff2f0; border:1px solid #d93025; "
        "border-radius:4px; padding:3px 6px; }"
        "QLineEdit:disabled { color:#9d7773; background:#f8f3f2; border-color:#d7b8b5; }"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lookup_dict = {}
        self.pattern = None
        self.base_tooltip = ""
        self.textChanged.connect(self.refresh_mapping_state)

    def set_mapping_dict(self, lookup_dict):
        self.lookup_dict = lookup_dict or {}
        terms = sorted(self.lookup_dict.keys(), key=len, reverse=True)
        self.pattern = (
            re.compile('|'.join(re.escape(term) for term in terms), re.IGNORECASE)
            if terms else None
        )
        self.refresh_mapping_state()

    def set_mapping_tooltip(self, tooltip):
        self.base_tooltip = tooltip
        self.refresh_mapping_state()

    def refresh_mapping_state(self):
        text = self.text().strip()
        if not text:
            self.setStyleSheet("")
            self.setToolTip(self.base_tooltip)
            return

        matches = list(self.pattern.finditer(text)) if self.pattern else []
        unmatched_parts = []
        cursor = 0
        for match in matches:
            unmatched_parts.append(text[cursor:match.start()])
            cursor = match.end()
        unmatched_parts.append(text[cursor:])
        unmatched = ''.join(unmatched_parts)
        # Dedicated object fields accept multiple mapping names separated by common delimiters.
        unmatched = re.sub(r"[\s,，、;；/|+&]+", "", unmatched)

        if matches and not unmatched:
            recognized = "、".join(match.group(0) for match in matches)
            self.setStyleSheet(self.VALID_STYLE)
            status = f"\u2713 已识别：{recognized}"
        else:
            self.setStyleSheet(self.INVALID_STYLE)
            status = "⚠ 存在未识别的物品名，请检查模型标签映射或拼写。"
        self.setToolTip("\n".join(part for part in (self.base_tooltip, status) if part))


class ProcessGuideDialog(QDialog):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📝 智能工序与安全配置大厅 (支持多套方案管理)")
        self.resize(1120, 780)
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
        right_layout.setSpacing(10)
        right_layout.setAlignment(Qt.AlignTop)

        description_group = QGroupBox("① 📝 步骤指令描述")
        description_group.setStyleSheet(
            "QGroupBox { font-weight:bold; color:#176b3a; }"
        )
        description_group.setMaximumHeight(190)
        description_layout = QVBoxLayout()
        description_hint = QLabel("请先写本步要做什么；模型映射中已识别的物品名会自动标绿。")
        description_hint.setStyleSheet("color:#5f6b63;")
        description_layout.addWidget(description_hint)
        self.text_editor = QTextEdit()
        self.text_editor.setMinimumHeight(82)
        self.text_editor.setMaximumHeight(120)
        self.text_editor.setPlaceholderText("例如：将红表笔插入箱子1接口")
        self.text_editor.setStyleSheet(
            "font-size:14px; line-height:1.5; border:1px solid #9fc9ad; border-radius:5px; padding:5px;"
        )
        self.text_editor.textChanged.connect(self.save_current_step)
        self.step_highlighter = KeywordHighlighter(self.text_editor.document(), color_hex="#0f9d58")
        description_layout.addWidget(self.text_editor)
        description_group.setLayout(description_layout)
        right_layout.addWidget(description_group)

        global_group = QGroupBox("⑤ 🛡️ 全局流程监控（当前方案）")
        global_group.setStyleSheet("QGroupBox { font-weight:bold; color:#174a8b; }")
        g_layout = QVBoxLayout()

        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("⏱️ 单步操作超时警告 (秒):"))
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(5, 300)  # 5秒到5分钟
        self.spin_timeout.setValue(DEFAULT_STEP_TIMEOUT)
        self.spin_timeout.valueChanged.connect(self.save_current_step)
        timeout_layout.addWidget(self.spin_timeout)
        timeout_layout.addStretch()
        g_layout.addLayout(timeout_layout)

        jump_scope_layout = QHBoxLayout()
        jump_scope_layout.addWidget(QLabel("🚦 跳步监控范围:"))
        self.combo_jump_scope = QComboBox()
        self.combo_jump_scope.addItem("关闭", "disabled")
        self.combo_jump_scope.addItem("只看后 1 步", "next_1")
        self.combo_jump_scope.addItem("只看后 2 步（推荐）", "next_2")
        self.combo_jump_scope.addItem("监控全部后续步骤", "all")
        self.combo_jump_scope.setToolTip(
            "全局设置：从当前工序向后监控几个逻辑工序。"
            "相邻且编号相同的乱序组算一个逻辑工序，但组内每一道都会被识别。"
        )
        self.combo_jump_scope.currentIndexChanged.connect(self.save_current_step)
        jump_scope_layout.addWidget(self.combo_jump_scope)
        jump_scope_layout.addStretch()
        g_layout.addLayout(jump_scope_layout)

        self.chk_jump_strong_action = VisibleCheckBox("跳步强动作确认：手/手套连续触达本步骤目标")
        self.chk_jump_strong_action.setToolTip(
            "只影响跳步报警。未来步骤只有 1 个目标时，一只手触达该目标即可；"
            "有 2 个及以上目标时，当前规则要求同一个手/手套检测框同时触达至少 2 个目标。"
        )
        self.chk_jump_strong_action.stateChanged.connect(self.save_current_step)
        g_layout.addWidget(self.chk_jump_strong_action)

        strong_frames_layout = QHBoxLayout()
        strong_frames_layout.addWidget(QLabel("   强动作连续帧数:"))
        self.spin_jump_strong_frames = QSpinBox()
        self.spin_jump_strong_frames.setRange(5, 120)
        self.spin_jump_strong_frames.setValue(DEFAULT_JUMP_STRONG_ACTION_FRAMES)
        self.spin_jump_strong_frames.valueChanged.connect(self.save_current_step)
        strong_frames_layout.addWidget(self.spin_jump_strong_frames)
        strong_frames_layout.addWidget(QLabel("帧"))
        strong_frames_layout.addStretch()
        g_layout.addLayout(strong_frames_layout)

        self.chk_jump_ignore_static = VisibleCheckBox("静态相交/靠近不触发跳步")
        self.chk_jump_ignore_static.setToolTip(
            "勾选后，零件开始时就靠近或重叠不会直接报跳步，需要观察到新的组装/触达动作。"
        )
        self.chk_jump_ignore_static.stateChanged.connect(self.save_current_step)
        g_layout.addWidget(self.chk_jump_ignore_static)

        global_group.setLayout(g_layout)

        step_group = QGroupBox("② ⚙️ 当前步骤基础配置")
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

        self.lbl_difficulty_reference = QLabel(DIFFICULTY_REFERENCE_TEXT)
        self.lbl_difficulty_reference.setWordWrap(True)
        self.lbl_difficulty_reference.setStyleSheet(
            "background:#eef5ff; border:1px solid #b8d4ff; border-radius:5px; "
            "padding:6px; color:#174a8b;"
        )
        s_layout.addWidget(self.lbl_difficulty_reference)

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

        self.action_rule_group = QGroupBox("③ 🎯 空间相交 / 装配专用设置")
        self.action_rule_group.setStyleSheet("QGroupBox { font-weight:bold; color:#174a8b; }")
        action_rule_layout = QVBoxLayout()
        self.target_validation_hint = QLabel(
            "<span style='color:#0b8043; font-weight:600;'>● 绿色：物品名已识别</span>"
            "&nbsp;&nbsp;&nbsp;"
            "<span style='color:#b3261e; font-weight:600;'>● 红色：存在未识别或拼错</span>"
        )
        self.target_validation_hint.setStyleSheet(
            "background:#f7f9f8; border:1px solid #d9e2dc; border-radius:4px; padding:5px;"
        )
        action_rule_layout.addWidget(self.target_validation_hint)

        self.action_confirm_layout = QHBoxLayout()
        self.action_confirm_layout.addWidget(QLabel("空间装配动作确认帧 (0=关闭):"))
        self.spin_action_confirm_frames = QSpinBox()
        self.spin_action_confirm_frames.setRange(0, 300)
        self.spin_action_confirm_frames.setSpecialValueText("关闭")
        self.spin_action_confirm_frames.setValue(DEFAULT_ACTION_CONFIRM_FRAMES)
        self.spin_action_confirm_frames.setSuffix(" 帧")
        self.spin_action_confirm_frames.setToolTip(
            "仅用于‘空间相交/装配’工序。若系统没有先看到目标分开再装到一起，"
            "则要求同一只手/手套同时触达至少两个目标并持续这么多帧，防止开局静态重叠误通过。"
            "设为 0 会关闭这道动作确认门槛，空间相交只按‘自定义连续帧’判定。"
        )
        self.spin_action_confirm_frames.valueChanged.connect(self.save_current_step)
        self.action_confirm_layout.addWidget(self.spin_action_confirm_frames)
        self.action_confirm_layout.addWidget(QLabel("（看到分开→相交时可直接确认）"))
        self.action_confirm_layout.addStretch()
        action_rule_layout.addLayout(self.action_confirm_layout)

        dependency_group = QGroupBox("④ 🔗 步骤顺序与前置依赖")
        dependency_group.setStyleSheet("QGroupBox { font-weight:bold; color:#7a4f00; }")
        dependency_layout = QVBoxLayout()

        prereq_layout = QHBoxLayout()
        prereq_layout.addWidget(QLabel("前置步骤:"))
        self.input_prerequisite_steps = QLineEdit()
        self.input_prerequisite_steps.setPlaceholderText("如 2,3；这些步骤完成前，本步骤/跳步都不允许")
        self.input_prerequisite_steps.setToolTip(
            "当前步骤的硬性依赖，填写必须先完成的步骤号，如 2,3。"
            "前置步骤始终会阻止本步骤正常完成；是否越过全局跳步范围主动监测，由旁边开关决定。"
        )
        self.input_prerequisite_steps.textChanged.connect(self.save_current_step)
        prereq_layout.addWidget(self.input_prerequisite_steps)
        self.chk_prerequisite_hard_monitor = VisibleCheckBox("硬监测（不受跳步范围限制）")
        self.chk_prerequisite_hard_monitor.setToolTip(
            "勾选：只要前置未完成，本步骤无论离当前多远都会被监测；"
            "不勾选：本步骤只在全局‘跳步监控范围’覆盖到时才监测。"
        )
        self.chk_prerequisite_hard_monitor.stateChanged.connect(self.save_current_step)
        prereq_layout.addWidget(self.chk_prerequisite_hard_monitor)
        dependency_layout.addLayout(prereq_layout)

        self.lbl_prerequisite_monitor_note = QLabel(
            "备注：这个开关只控制是否进行超范围硬监测，不会取消前置依赖本身。"
            "未勾选时，如果步骤仍位于‘后1步/后2步’范围内，提前执行照样会报警。"
        )
        self.lbl_prerequisite_monitor_note.setWordWrap(True)
        self.lbl_prerequisite_monitor_note.setStyleSheet("color:#6c4d00; padding:3px;")
        dependency_layout.addWidget(self.lbl_prerequisite_monitor_note)

        prereq_mode_layout = QHBoxLayout()
        prereq_mode_layout.addWidget(QLabel("仅前置依赖违规的处理方式:"))
        self.combo_prerequisite_mode = QComboBox()
        self.combo_prerequisite_mode.addItem("阻塞并报警", "block_and_alarm")
        self.combo_prerequisite_mode.addItem("只报警（默认）", "alarm_only")
        self.combo_prerequisite_mode.addItem("只阻塞", "block_only")
        self.combo_prerequisite_mode.setToolTip(
            "阻塞并报警：报警后仍停在原步骤；只报警：记录违规并按跳步流程继续；"
            "只阻塞：不报警，也不允许本步骤通过。"
        )
        self.combo_prerequisite_mode.currentIndexChanged.connect(self.save_current_step)
        prereq_mode_layout.addWidget(self.combo_prerequisite_mode)
        prereq_mode_layout.addStretch()
        dependency_layout.addLayout(prereq_mode_layout)

        self.lbl_prerequisite_mode_note = QLabel(
            "备注：这里只影响“前置步骤未完成却执行本步骤”的情况，不改变普通跳步报警。"
            "“只报警”仍会记录前置违规/NG，只是不把流程永久卡在原步骤。"
        )
        self.lbl_prerequisite_mode_note.setWordWrap(True)
        self.lbl_prerequisite_mode_note.setStyleSheet("color:#8a5a00; padding:3px;")
        dependency_layout.addWidget(self.lbl_prerequisite_mode_note)

        self.prewarning_group = QGroupBox("⚠️ 前置未完成时的提前预警")
        prewarning_layout = QVBoxLayout()
        self.chk_prewarning_enabled = VisibleCheckBox("启用：前置步骤未完成时，监测手接近/触达指定目标")
        self.chk_prewarning_enabled.stateChanged.connect(self.on_step_option_changed)
        prewarning_layout.addWidget(self.chk_prewarning_enabled)

        prewarning_target_layout = QHBoxLayout()
        prewarning_target_layout.addWidget(QLabel("预警目标:"))
        self.input_prewarning_target = MappingLineEdit()
        self.input_prewarning_target.setPlaceholderText("如 正确、上电按钮；留空则使用本步骤描述中的目标")
        self.input_prewarning_target.set_mapping_tooltip(
            "手接近哪个检测目标时发出预警。可填写映射中的中文名或英文名；"
            "留空时自动使用本步骤指令描述解析出的目标。"
        )
        self.input_prewarning_target.textChanged.connect(self.save_current_step)
        prewarning_target_layout.addWidget(self.input_prewarning_target)
        prewarning_layout.addLayout(prewarning_target_layout)

        prewarning_param_layout = QHBoxLayout()
        prewarning_param_layout.addWidget(QLabel("目标区外扩:"))
        self.spin_prewarning_padding = QDoubleSpinBox()
        self.spin_prewarning_padding.setRange(0.0, 2.0)
        self.spin_prewarning_padding.setSingleStep(0.05)
        self.spin_prewarning_padding.setDecimals(2)
        self.spin_prewarning_padding.setValue(DEFAULT_PREWARNING_PADDING_RATIO)
        self.spin_prewarning_padding.valueChanged.connect(self.save_current_step)
        prewarning_param_layout.addWidget(self.spin_prewarning_padding)
        prewarning_param_layout.addWidget(QLabel("   命中帧:"))
        self.spin_prewarning_hits = QSpinBox()
        self.spin_prewarning_hits.setRange(1, 120)
        self.spin_prewarning_hits.setValue(DEFAULT_PREWARNING_HIT_FRAMES)
        self.spin_prewarning_hits.valueChanged.connect(self.on_prewarning_frames_changed)
        prewarning_param_layout.addWidget(self.spin_prewarning_hits)
        prewarning_param_layout.addWidget(QLabel("/ 最近"))
        self.spin_prewarning_window = QSpinBox()
        self.spin_prewarning_window.setRange(1, 180)
        self.spin_prewarning_window.setValue(DEFAULT_PREWARNING_WINDOW_FRAMES)
        self.spin_prewarning_window.valueChanged.connect(self.on_prewarning_frames_changed)
        prewarning_param_layout.addWidget(self.spin_prewarning_window)
        prewarning_param_layout.addWidget(QLabel("帧"))
        prewarning_param_layout.addStretch()
        prewarning_layout.addLayout(prewarning_param_layout)

        self.lbl_prewarning_note = QLabel(
            "备注：只有本步骤填写的“前置步骤”尚未全部完成时才生效。外扩越大越早提醒；"
            "“3/最近5帧”表示最近5帧中至少命中3帧。这里只预警和声光提示，不推进工序、不记NG。"
            "检测模型必须包含并启用手/手套类别。"
        )
        self.lbl_prewarning_note.setWordWrap(True)
        self.lbl_prewarning_note.setStyleSheet("color:#8a5a00; background:#fff8e1; padding:5px;")
        prewarning_layout.addWidget(self.lbl_prewarning_note)
        self.prewarning_group.setLayout(prewarning_layout)
        dependency_layout.addWidget(self.prewarning_group)
        dependency_group.setLayout(dependency_layout)

        self.hand_release_group = QGroupBox("🖐️ 空间装配离手确认")
        hand_release_layout = QVBoxLayout()
        self.chk_require_hand_release = VisibleCheckBox("必须离手后才能完成（关系满足时进度停在99%）")
        self.chk_require_hand_release.stateChanged.connect(self.on_step_option_changed)
        hand_release_layout.addWidget(self.chk_require_hand_release)

        hand_release_param_layout = QHBoxLayout()
        hand_release_param_layout.addWidget(QLabel("操作区外扩:"))
        self.spin_hand_release_padding = QDoubleSpinBox()
        self.spin_hand_release_padding.setRange(0.0, 2.0)
        self.spin_hand_release_padding.setSingleStep(0.05)
        self.spin_hand_release_padding.setDecimals(2)
        self.spin_hand_release_padding.setValue(DEFAULT_HAND_RELEASE_PADDING)
        self.spin_hand_release_padding.valueChanged.connect(self.save_current_step)
        hand_release_param_layout.addWidget(self.spin_hand_release_padding)
        hand_release_param_layout.addWidget(QLabel("   离手确认:"))
        self.spin_hand_release_frames = QSpinBox()
        self.spin_hand_release_frames.setRange(1, 300)
        self.spin_hand_release_frames.setValue(DEFAULT_HAND_RELEASE_FRAMES)
        self.spin_hand_release_frames.setSuffix(" 帧")
        self.spin_hand_release_frames.valueChanged.connect(self.save_current_step)
        hand_release_param_layout.addWidget(self.spin_hand_release_frames)
        hand_release_param_layout.addStretch()
        hand_release_layout.addLayout(hand_release_param_layout)

        self.lbl_hand_release_note = QLabel(
            "备注：操作区是当前步骤所有目标检测框的局部区域；外扩用于容纳手在目标周围的操作。"
            "空间关系从一开始正常累计，达到阈值后手还在区域内则停在99%；"
            "所有手离开并持续达到“离手确认帧”后才完成，不会重新累计空间关系。"
            "检测模型必须包含并启用手/手套类别。"
        )
        self.lbl_hand_release_note.setWordWrap(True)
        self.lbl_hand_release_note.setStyleSheet("color:#174a8b; background:#eef5ff; padding:5px;")
        hand_release_layout.addWidget(self.lbl_hand_release_note)
        self.hand_release_group.setLayout(hand_release_layout)
        action_rule_layout.addWidget(self.hand_release_group)

        self.wrong_pair_group = QGroupBox("❌ 空间装配错误配对报警")
        wrong_pair_layout = QVBoxLayout()
        self.chk_wrong_pair_enabled = VisibleCheckBox("启用错误零件与目标的装配报警")
        self.chk_wrong_pair_enabled.stateChanged.connect(self.on_step_option_changed)
        wrong_pair_layout.addWidget(self.chk_wrong_pair_enabled)

        wrong_pair_target_layout = QHBoxLayout()
        wrong_pair_target_layout.addWidget(QLabel("错误零件:"))
        self.input_wrong_pair_item = MappingLineEdit()
        self.input_wrong_pair_item.setPlaceholderText("如 黑插头、红表笔；可填写一个或多个映射名称")
        self.input_wrong_pair_item.set_mapping_tooltip(
            "只填写模型标签映射中的物品名；多个名称用逗号或顿号分隔。"
        )
        self.input_wrong_pair_item.textChanged.connect(self.save_current_step)
        wrong_pair_target_layout.addWidget(self.input_wrong_pair_item)
        wrong_pair_target_layout.addWidget(QLabel("错误装配目标:"))
        self.input_wrong_pair_target = MappingLineEdit()
        self.input_wrong_pair_target.setPlaceholderText("如 箱子1接口；留空=正确装配的第2个目标")
        self.input_wrong_pair_target.set_mapping_tooltip(
            "填写模型标签映射中的装配目标名；留空时自动使用正确装配的第 2 个目标。"
        )
        self.input_wrong_pair_target.textChanged.connect(self.save_current_step)
        wrong_pair_target_layout.addWidget(self.input_wrong_pair_target)
        wrong_pair_layout.addLayout(wrong_pair_target_layout)

        wrong_pair_param_layout = QHBoxLayout()
        wrong_pair_param_layout.addWidget(QLabel("错误确认帧:"))
        self.spin_wrong_pair_frames = QSpinBox()
        self.spin_wrong_pair_frames.setRange(1, 300)
        self.spin_wrong_pair_frames.setValue(DEFAULT_WRONG_PAIR_CONFIRM_FRAMES)
        self.spin_wrong_pair_frames.setSuffix(" 帧")
        self.spin_wrong_pair_frames.valueChanged.connect(self.save_current_step)
        wrong_pair_param_layout.addWidget(self.spin_wrong_pair_frames)
        wrong_pair_param_layout.addWidget(QLabel("   错误框外扩:"))
        self.spin_wrong_pair_padding = QDoubleSpinBox()
        self.spin_wrong_pair_padding.setRange(0.0, 2.0)
        self.spin_wrong_pair_padding.setSingleStep(0.05)
        self.spin_wrong_pair_padding.setDecimals(2)
        self.spin_wrong_pair_padding.setValue(DEFAULT_WRONG_PAIR_PADDING_RATIO)
        self.spin_wrong_pair_padding.valueChanged.connect(self.save_current_step)
        wrong_pair_param_layout.addWidget(self.spin_wrong_pair_padding)
        wrong_pair_param_layout.addStretch()
        wrong_pair_layout.addLayout(wrong_pair_param_layout)

        self.lbl_wrong_pair_note = QLabel(
            "备注：只用于空间相交/装配。错误零件与错误装配目标连续相交达到确认帧后，"
            "仅触发红灯、蜂鸣器和界面报警；不提前预警、不阻塞、不记录NG。"
            "外扩比例会同时扩大错误零件框和目标框。"
        )
        self.lbl_wrong_pair_note.setWordWrap(True)
        self.lbl_wrong_pair_note.setStyleSheet("color:#9c1c1c; background:#fff0f0; padding:5px;")
        wrong_pair_layout.addWidget(self.lbl_wrong_pair_note)
        self.wrong_pair_group.setLayout(wrong_pair_layout)
        action_rule_layout.addWidget(self.wrong_pair_group)

        self.detach_layout = QHBoxLayout()
        self.detach_layout.addWidget(QLabel("拆除物(被拿走):"))
        self.input_detach_removed = MappingLineEdit()
        self.input_detach_removed.setPlaceholderText("拆开时离开基准物的目标，如 手机")
        self.input_detach_removed.set_mapping_tooltip(
            "填写模型标签映射中会被拿走的物品名。"
        )
        self.input_detach_removed.textChanged.connect(self.save_current_step)
        self.detach_layout.addWidget(self.input_detach_removed)
        self.detach_layout.addWidget(QLabel("基准物(保留参照):"))
        self.input_detach_base = MappingLineEdit()
        self.input_detach_base.setPlaceholderText("拆除物原本贴合/安装的目标，如 耳机")
        self.input_detach_base.set_mapping_tooltip(
            "填写模型标签映射中作为保留参照的物品名。"
        )
        self.input_detach_base.textChanged.connect(self.save_current_step)
        self.detach_layout.addWidget(self.input_detach_base)
        action_rule_layout.addLayout(self.detach_layout)

        self.step_tuning_layout = QHBoxLayout()
        self.step_tuning_layout.addWidget(QLabel("自定义连续帧:"))
        self.spin_stable_frames = QSpinBox()
        self.spin_stable_frames.setRange(0, 600)
        self.spin_stable_frames.setSpecialValueText("跟随难度")
        self.spin_stable_frames.setValue(0)
        self.spin_stable_frames.setToolTip(
            "当前工序的关系需要连续成立多少帧才通过。0 表示跟随难度："
            "简单=15、中等=40、困难=90。"
        )
        self.spin_stable_frames.valueChanged.connect(self.save_current_step)
        self.step_tuning_layout.addWidget(self.spin_stable_frames)

        self.step_tuning_layout.addWidget(QLabel("   框外扩比例 (每边):"))
        self.spin_padding_ratio = QDoubleSpinBox()
        self.spin_padding_ratio.setRange(-1.0, 2.0)
        self.spin_padding_ratio.setSingleStep(0.01)
        self.spin_padding_ratio.setDecimals(2)
        self.spin_padding_ratio.setSpecialValueText("跟随难度")
        self.spin_padding_ratio.setValue(-1.0)
        self.spin_padding_ratio.setToolTip(
            "按检测框宽高向四周外扩的比例。-1 表示跟随难度：简单=0.20、中等=0.08、困难=0。"
            "例如 0.50 表示每边增加原宽/高的 50%，最终宽高为原来的 2 倍；"
            "1.00 最终为 3 倍，2.00 最终为 5 倍。"
            "空间相交会外扩所有目标物；手触达只按此值外扩目标物，手框固定轻微外扩 0.05；"
            "拆除分离会外扩两个物品，数值越大判定越严格。"
        )
        self.spin_padding_ratio.valueChanged.connect(self.save_current_step)
        self.step_tuning_layout.addWidget(self.spin_padding_ratio)
        self.step_tuning_layout.addStretch()
        action_rule_layout.addLayout(self.step_tuning_layout)
        self.action_rule_group.setLayout(action_rule_layout)

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

        step_group.setLayout(s_layout)
        right_layout.addWidget(step_group)
        right_layout.addWidget(self.action_rule_group)
        right_layout.addWidget(dependency_group)

        right_layout.addWidget(global_group)

        forbidden_group = QGroupBox("⑥ 🚫 画面绝对禁止出现")
        forbidden_group.setStyleSheet("QGroupBox { font-weight:bold; color:#b3261e; }")
        forbidden_layout = QVBoxLayout()
        forbidden_hint = QLabel("这是当前方案的全局底线：不论执行到哪一步，画面中都不允许出现。")
        forbidden_hint.setStyleSheet("color:#7f4945;")
        forbidden_layout.addWidget(forbidden_hint)
        self.forbidden_input = QTextEdit()
        self.forbidden_input.setFixedHeight(54)
        self.forbidden_input.setPlaceholderText("输入映射中的物品名，多个用逗号分隔")
        self.forbidden_input.setStyleSheet(
            "border:1px solid #e2aaa5; border-radius:5px; padding:4px; background:#fff8f7;"
        )
        self.forbidden_input.textChanged.connect(self.save_current_step)
        self.forbidden_highlighter = KeywordHighlighter(
            self.forbidden_input.document(), color_hex="#d93025"
        )
        forbidden_layout.addWidget(self.forbidden_input)
        forbidden_group.setLayout(forbidden_layout)
        right_layout.addWidget(forbidden_group)

        self.mapping_name_inputs = (
            self.input_prewarning_target,
            self.input_wrong_pair_item,
            self.input_wrong_pair_target,
            self.input_detach_removed,
            self.input_detach_base,
        )

        self.btn_save_all = QPushButton("💾 保存所有方案配置")
        self.btn_save_all.setMinimumHeight(45)
        self.btn_save_all.setStyleSheet("background-color: #1a73e8; color: white; font-weight: bold; font-size: 14px;")
        self.btn_save_all.clicked.connect(self.save_to_json)
        right_layout.addWidget(self.btn_save_all)

        right_panel = QWidget()
        right_panel.setLayout(right_layout)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_panel)
        right_scroll.setMinimumWidth(620)
        main_layout.addWidget(right_scroll, stretch=5)
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
            for mapping_input in self.mapping_name_inputs:
                mapping_input.set_mapping_dict(self.lookup_dict)

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
            self.config_data["profiles"][name] = {
                "process_steps": [],
                "forbidden_items": "",
                "step_timeout": DEFAULT_STEP_TIMEOUT,
                "jump_monitor_scope": DEFAULT_JUMP_MONITOR_SCOPE,
                "jump_strong_action_enabled": DEFAULT_JUMP_STRONG_ACTION_ENABLED,
                "jump_strong_action_frames": DEFAULT_JUMP_STRONG_ACTION_FRAMES,
                "jump_ignore_static_intersection": DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION,
            }
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

        scope = profile_data.get("jump_monitor_scope", DEFAULT_JUMP_MONITOR_SCOPE)
        scope_idx = self.combo_jump_scope.findData(scope)
        if scope_idx < 0:
            scope_idx = self.combo_jump_scope.findData(DEFAULT_JUMP_MONITOR_SCOPE)
        self.combo_jump_scope.blockSignals(True)
        self.combo_jump_scope.setCurrentIndex(scope_idx)
        self.combo_jump_scope.blockSignals(False)

        self.chk_jump_strong_action.blockSignals(True)
        self.chk_jump_strong_action.setChecked(bool(profile_data.get(
            "jump_strong_action_enabled", DEFAULT_JUMP_STRONG_ACTION_ENABLED
        )))
        self.chk_jump_strong_action.blockSignals(False)

        self.spin_jump_strong_frames.blockSignals(True)
        self.spin_jump_strong_frames.setValue(int(profile_data.get(
            "jump_strong_action_frames", DEFAULT_JUMP_STRONG_ACTION_FRAMES
        )))
        self.spin_jump_strong_frames.blockSignals(False)

        self.chk_jump_ignore_static.blockSignals(True)
        self.chk_jump_ignore_static.setChecked(bool(profile_data.get(
            "jump_ignore_static_intersection", DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
        )))
        self.chk_jump_ignore_static.blockSignals(False)

        for step_data in profile_data.get("process_steps", []):
            if isinstance(step_data, str): step_data = {"text": step_data}
            if "difficulty" not in step_data: step_data["difficulty"] = "中等 (标准) 🟡"
            if "count" not in step_data: step_data["count"] = 1
            if "multi_strategy" not in step_data: step_data["multi_strategy"] = "lock"
            if "cooldown" not in step_data: step_data["cooldown"] = 1.5
            if "action_type" not in step_data: step_data["action_type"] = "spatial"
            if "action_confirm_frames" not in step_data:
                step_data["action_confirm_frames"] = DEFAULT_ACTION_CONFIRM_FRAMES
            if "order_group" not in step_data: step_data["order_group"] = ""
            if "prerequisite_steps" not in step_data: step_data["prerequisite_steps"] = ""
            if "prerequisite_hard_monitor" not in step_data:
                step_data["prerequisite_hard_monitor"] = DEFAULT_PREREQUISITE_HARD_MONITOR
            if "prerequisite_mode" not in step_data:
                step_data["prerequisite_mode"] = DEFAULT_PREREQUISITE_MODE
            if "prewarning_enabled" not in step_data: step_data["prewarning_enabled"] = False
            if "prewarning_target" not in step_data: step_data["prewarning_target"] = ""
            if "prewarning_padding_ratio" not in step_data:
                step_data["prewarning_padding_ratio"] = DEFAULT_PREWARNING_PADDING_RATIO
            if "prewarning_hit_frames" not in step_data:
                step_data["prewarning_hit_frames"] = DEFAULT_PREWARNING_HIT_FRAMES
            if "prewarning_window_frames" not in step_data:
                step_data["prewarning_window_frames"] = DEFAULT_PREWARNING_WINDOW_FRAMES
            if "require_hand_release" not in step_data: step_data["require_hand_release"] = False
            if "hand_release_padding" not in step_data:
                step_data["hand_release_padding"] = DEFAULT_HAND_RELEASE_PADDING
            if "hand_release_frames" not in step_data:
                step_data["hand_release_frames"] = DEFAULT_HAND_RELEASE_FRAMES
            if "wrong_pair_enabled" not in step_data: step_data["wrong_pair_enabled"] = False
            if "wrong_pair_item" not in step_data: step_data["wrong_pair_item"] = ""
            if "wrong_pair_target" not in step_data: step_data["wrong_pair_target"] = ""
            if "wrong_pair_confirm_frames" not in step_data:
                step_data["wrong_pair_confirm_frames"] = DEFAULT_WRONG_PAIR_CONFIRM_FRAMES
            if "wrong_pair_padding_ratio" not in step_data:
                step_data["wrong_pair_padding_ratio"] = DEFAULT_WRONG_PAIR_PADDING_RATIO
            if "detach_removed" not in step_data: step_data["detach_removed"] = ""
            if "detach_base" not in step_data: step_data["detach_base"] = ""
            if "stable_frames" not in step_data:
                step_data["stable_frames"] = step_data.get("detach_stable_frames", 0)
            if "padding_ratio" not in step_data:
                step_data["padding_ratio"] = step_data.get("detach_padding_ratio", -1)

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
        self.spin_action_confirm_frames.blockSignals(True)
        self.input_order_group.blockSignals(True)
        self.input_prerequisite_steps.blockSignals(True)
        self.chk_prerequisite_hard_monitor.blockSignals(True)
        self.combo_prerequisite_mode.blockSignals(True)
        self.chk_prewarning_enabled.blockSignals(True)
        self.input_prewarning_target.blockSignals(True)
        self.spin_prewarning_padding.blockSignals(True)
        self.spin_prewarning_hits.blockSignals(True)
        self.spin_prewarning_window.blockSignals(True)
        self.chk_require_hand_release.blockSignals(True)
        self.spin_hand_release_padding.blockSignals(True)
        self.spin_hand_release_frames.blockSignals(True)
        self.chk_wrong_pair_enabled.blockSignals(True)
        self.input_wrong_pair_item.blockSignals(True)
        self.input_wrong_pair_target.blockSignals(True)
        self.spin_wrong_pair_frames.blockSignals(True)
        self.spin_wrong_pair_padding.blockSignals(True)
        self.input_detach_removed.blockSignals(True)
        self.input_detach_base.blockSignals(True)
        self.spin_stable_frames.blockSignals(True)
        self.spin_padding_ratio.blockSignals(True)
        self.text_editor.clear()
        self.spin_count.setValue(1)
        self.combo_difficulty.setCurrentIndex(1)
        self.combo_strategy.setCurrentIndex(0)
        self.spin_cooldown.setValue(1.5)
        self.combo_action_type.setCurrentIndex(0)
        self.spin_action_confirm_frames.setValue(DEFAULT_ACTION_CONFIRM_FRAMES)
        self.input_order_group.clear()
        self.input_prerequisite_steps.clear()
        self.chk_prerequisite_hard_monitor.setChecked(DEFAULT_PREREQUISITE_HARD_MONITOR)
        self.combo_prerequisite_mode.setCurrentIndex(
            self.combo_prerequisite_mode.findData(DEFAULT_PREREQUISITE_MODE)
        )
        self.chk_prewarning_enabled.setChecked(False)
        self.input_prewarning_target.clear()
        self.spin_prewarning_padding.setValue(DEFAULT_PREWARNING_PADDING_RATIO)
        self.spin_prewarning_hits.setValue(DEFAULT_PREWARNING_HIT_FRAMES)
        self.spin_prewarning_window.setValue(DEFAULT_PREWARNING_WINDOW_FRAMES)
        self.chk_require_hand_release.setChecked(False)
        self.spin_hand_release_padding.setValue(DEFAULT_HAND_RELEASE_PADDING)
        self.spin_hand_release_frames.setValue(DEFAULT_HAND_RELEASE_FRAMES)
        self.chk_wrong_pair_enabled.setChecked(False)
        self.input_wrong_pair_item.clear()
        self.input_wrong_pair_target.clear()
        self.spin_wrong_pair_frames.setValue(DEFAULT_WRONG_PAIR_CONFIRM_FRAMES)
        self.spin_wrong_pair_padding.setValue(DEFAULT_WRONG_PAIR_PADDING_RATIO)
        self.input_detach_removed.clear()
        self.input_detach_base.clear()
        self.spin_stable_frames.setValue(0)
        self.spin_padding_ratio.setValue(-1.0)
        self.text_editor.blockSignals(False)
        self.spin_count.blockSignals(False)
        self.combo_difficulty.blockSignals(False)
        self.combo_strategy.blockSignals(False)
        self.spin_cooldown.blockSignals(False)
        self.combo_action_type.blockSignals(False)
        self.spin_action_confirm_frames.blockSignals(False)
        self.input_order_group.blockSignals(False)
        self.input_prerequisite_steps.blockSignals(False)
        self.chk_prerequisite_hard_monitor.blockSignals(False)
        self.combo_prerequisite_mode.blockSignals(False)
        self.chk_prewarning_enabled.blockSignals(False)
        self.input_prewarning_target.blockSignals(False)
        self.spin_prewarning_padding.blockSignals(False)
        self.spin_prewarning_hits.blockSignals(False)
        self.spin_prewarning_window.blockSignals(False)
        self.chk_require_hand_release.blockSignals(False)
        self.spin_hand_release_padding.blockSignals(False)
        self.spin_hand_release_frames.blockSignals(False)
        self.chk_wrong_pair_enabled.blockSignals(False)
        self.input_wrong_pair_item.blockSignals(False)
        self.input_wrong_pair_target.blockSignals(False)
        self.spin_wrong_pair_frames.blockSignals(False)
        self.spin_wrong_pair_padding.blockSignals(False)
        self.input_detach_removed.blockSignals(False)
        self.input_detach_base.blockSignals(False)
        self.spin_stable_frames.blockSignals(False)
        self.spin_padding_ratio.blockSignals(False)
        self._refresh_mapping_input_states()
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
        profile["jump_monitor_scope"] = self.combo_jump_scope.currentData() or DEFAULT_JUMP_MONITOR_SCOPE
        profile["jump_strong_action_enabled"] = self.chk_jump_strong_action.isChecked()
        profile["jump_strong_action_frames"] = self.spin_jump_strong_frames.value()
        profile["jump_ignore_static_intersection"] = self.chk_jump_ignore_static.isChecked()

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

    def _refresh_mapping_input_states(self):
        for mapping_input in self.mapping_name_inputs:
            mapping_input.refresh_mapping_state()

    def _update_detach_ui_visibility(self):
        action_index = self.combo_action_type.currentIndex()
        is_detach = action_index == 2
        is_spatial = action_index == 0
        action_titles = {
            0: "③ 🎯 空间相交 / 装配专用设置",
            1: "③ 👆 手 / 手套触达专用设置",
            2: "③ 🧩 拆除 / 分离专用设置",
        }
        self.action_rule_group.setTitle(action_titles.get(action_index, action_titles[0]))
        self.target_validation_hint.setVisible(action_index != 1)
        self._set_layout_visible(self.detach_layout, is_detach)
        self._set_layout_visible(self.action_confirm_layout, is_spatial)
        self.hand_release_group.setVisible(is_spatial)
        self.wrong_pair_group.setVisible(is_spatial)
        self._update_step_option_controls()

    def _update_step_option_controls(self):
        prewarning_enabled = self.chk_prewarning_enabled.isChecked()
        for widget in (
                self.input_prewarning_target, self.spin_prewarning_padding,
                self.spin_prewarning_hits, self.spin_prewarning_window):
            widget.setEnabled(prewarning_enabled)

        hand_release_enabled = (
            self.combo_action_type.currentIndex() == 0
            and self.chk_require_hand_release.isChecked()
        )
        self.spin_hand_release_padding.setEnabled(hand_release_enabled)
        self.spin_hand_release_frames.setEnabled(hand_release_enabled)

        wrong_pair_enabled = (
            self.combo_action_type.currentIndex() == 0
            and self.chk_wrong_pair_enabled.isChecked()
        )
        for widget in (
                self.input_wrong_pair_item, self.input_wrong_pair_target,
                self.spin_wrong_pair_frames, self.spin_wrong_pair_padding):
            widget.setEnabled(wrong_pair_enabled)

    def on_step_option_changed(self):
        self._update_step_option_controls()
        self.save_current_step()

    def on_prewarning_frames_changed(self):
        if self.spin_prewarning_window.value() < self.spin_prewarning_hits.value():
            self.spin_prewarning_window.blockSignals(True)
            self.spin_prewarning_window.setValue(self.spin_prewarning_hits.value())
            self.spin_prewarning_window.blockSignals(False)
        self.save_current_step()

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
            self.spin_action_confirm_frames.blockSignals(True)
            self.input_order_group.blockSignals(True)
            self.input_prerequisite_steps.blockSignals(True)
            self.chk_prerequisite_hard_monitor.blockSignals(True)
            self.combo_prerequisite_mode.blockSignals(True)
            self.chk_prewarning_enabled.blockSignals(True)
            self.input_prewarning_target.blockSignals(True)
            self.spin_prewarning_padding.blockSignals(True)
            self.spin_prewarning_hits.blockSignals(True)
            self.spin_prewarning_window.blockSignals(True)
            self.chk_require_hand_release.blockSignals(True)
            self.spin_hand_release_padding.blockSignals(True)
            self.spin_hand_release_frames.blockSignals(True)
            self.chk_wrong_pair_enabled.blockSignals(True)
            self.input_wrong_pair_item.blockSignals(True)
            self.input_wrong_pair_target.blockSignals(True)
            self.spin_wrong_pair_frames.blockSignals(True)
            self.spin_wrong_pair_padding.blockSignals(True)
            self.input_detach_removed.blockSignals(True)
            self.input_detach_base.blockSignals(True)
            self.spin_stable_frames.blockSignals(True)
            self.spin_padding_ratio.blockSignals(True)

            self.text_editor.setPlainText(step_data.get("text", ""))
            self.combo_difficulty.setCurrentText(step_data.get("difficulty", "中等 (标准) 🟡"))
            self.spin_count.setValue(step_data.get("count", 1))
            action_type = step_data.get("action_type", "spatial")
            action_idx = 2 if action_type == "detach" else (1 if action_type == "hand_touch" else 0)
            self.combo_action_type.setCurrentIndex(action_idx)
            action_confirm_frames = step_data.get(
                "action_confirm_frames", DEFAULT_ACTION_CONFIRM_FRAMES
            )
            if action_confirm_frames in (None, ""):
                action_confirm_frames = DEFAULT_ACTION_CONFIRM_FRAMES
            self.spin_action_confirm_frames.setValue(int(action_confirm_frames))
            self.input_order_group.setText(step_data.get("order_group", ""))
            self.input_prerequisite_steps.setText(str(step_data.get("prerequisite_steps", "")))
            self.chk_prerequisite_hard_monitor.setChecked(bool(step_data.get(
                "prerequisite_hard_monitor", DEFAULT_PREREQUISITE_HARD_MONITOR
            )))
            prerequisite_mode = str(step_data.get(
                "prerequisite_mode", DEFAULT_PREREQUISITE_MODE
            ))
            prerequisite_mode_idx = self.combo_prerequisite_mode.findData(prerequisite_mode)
            if prerequisite_mode_idx < 0:
                prerequisite_mode_idx = self.combo_prerequisite_mode.findData(DEFAULT_PREREQUISITE_MODE)
            self.combo_prerequisite_mode.setCurrentIndex(prerequisite_mode_idx)
            self.chk_prewarning_enabled.setChecked(bool(step_data.get("prewarning_enabled", False)))
            self.input_prewarning_target.setText(str(step_data.get("prewarning_target", "")))
            self.spin_prewarning_padding.setValue(float(step_data.get(
                "prewarning_padding_ratio", DEFAULT_PREWARNING_PADDING_RATIO
            )))
            self.spin_prewarning_hits.setValue(int(step_data.get(
                "prewarning_hit_frames", DEFAULT_PREWARNING_HIT_FRAMES
            )))
            self.spin_prewarning_window.setValue(int(step_data.get(
                "prewarning_window_frames", DEFAULT_PREWARNING_WINDOW_FRAMES
            )))
            self.chk_require_hand_release.setChecked(bool(step_data.get("require_hand_release", False)))
            self.spin_hand_release_padding.setValue(float(step_data.get(
                "hand_release_padding", DEFAULT_HAND_RELEASE_PADDING
            )))
            self.spin_hand_release_frames.setValue(int(step_data.get(
                "hand_release_frames", DEFAULT_HAND_RELEASE_FRAMES
            )))
            self.chk_wrong_pair_enabled.setChecked(bool(step_data.get("wrong_pair_enabled", False)))
            self.input_wrong_pair_item.setText(str(step_data.get("wrong_pair_item", "")))
            self.input_wrong_pair_target.setText(str(step_data.get("wrong_pair_target", "")))
            self.spin_wrong_pair_frames.setValue(int(step_data.get(
                "wrong_pair_confirm_frames", DEFAULT_WRONG_PAIR_CONFIRM_FRAMES
            )))
            self.spin_wrong_pair_padding.setValue(float(step_data.get(
                "wrong_pair_padding_ratio", DEFAULT_WRONG_PAIR_PADDING_RATIO
            )))
            self.input_detach_removed.setText(step_data.get("detach_removed", ""))
            self.input_detach_base.setText(step_data.get("detach_base", ""))
            stable_frames = step_data.get("stable_frames")
            if stable_frames in (None, "") and action_type == "detach":
                stable_frames = step_data.get("detach_stable_frames", 0)
            padding_ratio = step_data.get("padding_ratio")
            if padding_ratio in (None, "") and action_type == "detach":
                padding_ratio = step_data.get("detach_padding_ratio", -1)
            self.spin_stable_frames.setValue(int(stable_frames or 0))
            self.spin_padding_ratio.setValue(float(
                -1 if padding_ratio in (None, "") else padding_ratio
            ))

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
            self.spin_action_confirm_frames.blockSignals(False)
            self.input_order_group.blockSignals(False)
            self.input_prerequisite_steps.blockSignals(False)
            self.chk_prerequisite_hard_monitor.blockSignals(False)
            self.combo_prerequisite_mode.blockSignals(False)
            self.chk_prewarning_enabled.blockSignals(False)
            self.input_prewarning_target.blockSignals(False)
            self.spin_prewarning_padding.blockSignals(False)
            self.spin_prewarning_hits.blockSignals(False)
            self.spin_prewarning_window.blockSignals(False)
            self.chk_require_hand_release.blockSignals(False)
            self.spin_hand_release_padding.blockSignals(False)
            self.spin_hand_release_frames.blockSignals(False)
            self.chk_wrong_pair_enabled.blockSignals(False)
            self.input_wrong_pair_item.blockSignals(False)
            self.input_wrong_pair_target.blockSignals(False)
            self.spin_wrong_pair_frames.blockSignals(False)
            self.spin_wrong_pair_padding.blockSignals(False)
            self.input_detach_removed.blockSignals(False)
            self.input_detach_base.blockSignals(False)
            self.spin_stable_frames.blockSignals(False)
            self.spin_padding_ratio.blockSignals(False)

            self._refresh_mapping_input_states()
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
                "action_confirm_frames": self.spin_action_confirm_frames.value(),
                "order_group": self.input_order_group.text().strip(),
                "prerequisite_steps": self.input_prerequisite_steps.text().strip(),
                "prerequisite_hard_monitor": self.chk_prerequisite_hard_monitor.isChecked(),
                "prerequisite_mode": (
                    self.combo_prerequisite_mode.currentData() or DEFAULT_PREREQUISITE_MODE
                ),
                "prewarning_enabled": self.chk_prewarning_enabled.isChecked(),
                "prewarning_target": self.input_prewarning_target.text().strip(),
                "prewarning_padding_ratio": self.spin_prewarning_padding.value(),
                "prewarning_hit_frames": self.spin_prewarning_hits.value(),
                "prewarning_window_frames": self.spin_prewarning_window.value(),
                "require_hand_release": self.chk_require_hand_release.isChecked(),
                "hand_release_padding": self.spin_hand_release_padding.value(),
                "hand_release_frames": self.spin_hand_release_frames.value(),
                "wrong_pair_enabled": self.chk_wrong_pair_enabled.isChecked(),
                "wrong_pair_item": self.input_wrong_pair_item.text().strip(),
                "wrong_pair_target": self.input_wrong_pair_target.text().strip(),
                "wrong_pair_confirm_frames": self.spin_wrong_pair_frames.value(),
                "wrong_pair_padding_ratio": self.spin_wrong_pair_padding.value(),
                "detach_removed": self.input_detach_removed.text().strip(),
                "detach_base": self.input_detach_base.text().strip(),
                "stable_frames": self.spin_stable_frames.value(),
                "padding_ratio": self.spin_padding_ratio.value(),
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
                                                                "action_confirm_frames": DEFAULT_ACTION_CONFIRM_FRAMES,
                                                                "prerequisite_steps": "",
                                                                "prerequisite_hard_monitor": DEFAULT_PREREQUISITE_HARD_MONITOR,
                                                                "prerequisite_mode": DEFAULT_PREREQUISITE_MODE,
                                                                "prewarning_enabled": False,
                                                                "prewarning_target": "",
                                                                "prewarning_padding_ratio": DEFAULT_PREWARNING_PADDING_RATIO,
                                                                "prewarning_hit_frames": DEFAULT_PREWARNING_HIT_FRAMES,
                                                                "prewarning_window_frames": DEFAULT_PREWARNING_WINDOW_FRAMES,
                                                                "require_hand_release": False,
                                                                "hand_release_padding": DEFAULT_HAND_RELEASE_PADDING,
                                                                "hand_release_frames": DEFAULT_HAND_RELEASE_FRAMES,
                                                                "wrong_pair_enabled": False,
                                                                "wrong_pair_item": "",
                                                                "wrong_pair_target": "",
                                                                "wrong_pair_confirm_frames": DEFAULT_WRONG_PAIR_CONFIRM_FRAMES,
                                                                "wrong_pair_padding_ratio": DEFAULT_WRONG_PAIR_PADDING_RATIO,
                                                                "detach_removed": "", "detach_base": "",
                                                                "stable_frames": 0,
                                                                "padding_ratio": -1})
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
        self.config_data["jump_monitor_scope"] = profile.get("jump_monitor_scope", DEFAULT_JUMP_MONITOR_SCOPE)
        self.config_data["jump_strong_action_enabled"] = profile.get(
            "jump_strong_action_enabled", DEFAULT_JUMP_STRONG_ACTION_ENABLED
        )
        self.config_data["jump_strong_action_frames"] = profile.get(
            "jump_strong_action_frames", DEFAULT_JUMP_STRONG_ACTION_FRAMES
        )
        self.config_data["jump_ignore_static_intersection"] = profile.get(
            "jump_ignore_static_intersection", DEFAULT_JUMP_IGNORE_STATIC_INTERSECTION
        )

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=4)

        QMessageBox.information(self, "成功", f"方案【{self.active_profile_name}】及所有配置已成功保存！")
        self.accept()
