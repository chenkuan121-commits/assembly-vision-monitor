import os
import shutil
import json
from ultralytics import YOLO
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                               QLabel, QLineEdit, QPushButton, QMessageBox, QHBoxLayout)

# 确保必要的文件夹存在
os.makedirs("models", exist_ok=True)
os.makedirs("configs", exist_ok=True)


class ModelMappingDialog(QDialog):
    """负责中英文映射配置与修改的弹窗"""

    def __init__(self, model_path, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ 配置模型识别目标 (中英文映射)")
        self.setMinimumWidth(400)

        self.model_path = model_path
        self.config_path = config_path
        self.input_fields = {}  # 存放 {class_id: QLineEdit}

        # 1. 读取 YOLO 原始标签
        try:
            model = YOLO(self.model_path)
            self.original_names = model.names  # 格式如 {0: 'lemo', 1: 'shell'}
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取模型失败: {e}")
            self.reject()
            return

        # 2. 尝试读取已有的配置文件（实现“修改”记忆功能）
        self.existing_mapping = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.existing_mapping = config_data.get("mapping", {})

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        info = QLabel("请为底层标签输入中文名。\n留空则界面上默认显示英文原名。")
        info.setStyleSheet("color: #555; margin-bottom: 10px;")
        main_layout.addWidget(info)

        form_layout = QFormLayout()

        # 动态生成输入框，并填入历史数据
        for class_id, eng_name in self.original_names.items():
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"中文别名 (原名: {eng_name})")

            # 如果之前保存过这个 class_id 的中文名，就把它填进去
            str_id = str(class_id)  # JSON 的 key 是字符串
            if str_id in self.existing_mapping:
                line_edit.setText(self.existing_mapping[str_id]["zh_name"])

            form_layout.addRow(QLabel(f"[{class_id}] {eng_name}:"), line_edit)
            self.input_fields[class_id] = line_edit

        main_layout.addLayout(form_layout)

        # 按钮区
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("💾 保存配置")
        btn_cancel = QPushButton("取消")
        btn_save.clicked.connect(self.save_config)
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def save_config(self):
        mapping_data = {}
        for class_id, line_edit in self.input_fields.items():
            zh_name = line_edit.text().strip()
            eng_name = self.original_names[class_id]
            # 无论用户填没填中文，我们都把关系存下来
            mapping_data[str(class_id)] = {
                "eng_name": eng_name,
                "zh_name": zh_name if zh_name else eng_name  # 如果没填，中文名就用英文顶替
            }

        config_data = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception:
                config_data = {}

        config_data["model_path"] = self.model_path
        config_data["mapping"] = mapping_data

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)

        QMessageBox.information(self, "成功", "模型配置已保存！")
        self.accept()
