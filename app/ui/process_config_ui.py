"""
@file process_config_ui.py
@brief 流程配置/步骤编辑界面
@author AI Assistant
@date 2026-03-10
@note 美观的可视化流程编辑器，支持步骤的增删改查和拖拽排序
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QGridLayout, QDoubleSpinBox, QCheckBox, QScrollArea,
    QFrame, QFileDialog, QLineEdit, QTextEdit, QSpinBox, QListWidget,
    QListWidgetItem, QSplitter, QMessageBox, QSizePolicy, QDialog,
    QDialogButtonBox, QFormLayout, QTabWidget
)
from PyQt5.QtGui import QColor, QFont, QIcon
from PyQt5.QtCore import Qt, pyqtSignal
import os
import json

from app.core.rule_engine import RuleEngine
from app.core.action_recognizer import ActionType


# ======================== 样式 ========================
STYLE_SHEET = """
    QGroupBox {
        font-size: 13px;
        font-weight: bold;
        border: 2px solid #9b59b6;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        background-color: #fdf2ff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #8e44ad;
    }
    QPushButton {
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
        border: none;
        min-height: 30px;
    }
    QPushButton:disabled {
        background-color: #bdc3c7;
        color: #7f8c8d;
    }
    QLabel { font-size: 12px; }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
        padding: 6px;
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 12px;
        min-height: 26px;
    }
    QListWidget {
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 12px;
        alternate-background-color: #f5eef8;
    }
    QListWidget::item { padding: 8px; border-bottom: 1px solid #e8daef; }
    QListWidget::item:selected { background-color: #9b59b6; color: white; }
    QTextEdit {
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 12px;
    }
"""

BTN_PRIMARY = "background-color: #9b59b6; color: white;"
BTN_SUCCESS = "background-color: #2ecc71; color: white;"
BTN_DANGER = "background-color: #e74c3c; color: white;"
BTN_WARNING = "background-color: #f39c12; color: white;"
BTN_INFO = "background-color: #1abc9c; color: white;"
BTN_DARK = "background-color: #34495e; color: white;"


# 可用动作类型列表
ACTION_TYPES = [at.value for at in ActionType]
ACTION_TYPES_EN = [at.name for at in ActionType]


class ProcessConfigWidget(QWidget):
    """
    @class ProcessConfigWidget
    @brief 流程配置编辑界面

    功能：
    - 创建/编辑/删除操作流程
    - 添加/编辑/删除/排序步骤
    - 配置时间约束和告警规则
    - 导入/导出流程配置
    """

    process_changed = pyqtSignal()  # 流程变更信号

    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLE_SHEET)
        self.rule_engine = RuleEngine()
        self.current_step_idx = -1
        self.init_ui()
        self.refresh_process_list()

    def init_ui(self):
        """构建界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- 顶部工具栏 ---
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("📦 选择流程:"))
        self.process_combo = QComboBox()
        self.process_combo.setMinimumWidth(280)
        self.process_combo.currentIndexChanged.connect(self.on_process_selected)
        toolbar.addWidget(self.process_combo)

        self.btn_new = QPushButton("+ 新建")
        self.btn_new.setStyleSheet(BTN_SUCCESS)
        self.btn_new.clicked.connect(self.create_new_process)
        toolbar.addWidget(self.btn_new)

        self.btn_copy = QPushButton("📋 复制")
        self.btn_copy.setStyleSheet(BTN_INFO)
        self.btn_copy.clicked.connect(self.copy_process)
        toolbar.addWidget(self.btn_copy)

        self.btn_delete = QPushButton("🗑 删除")
        self.btn_delete.setStyleSheet(BTN_DANGER)
        self.btn_delete.clicked.connect(self.delete_process)
        toolbar.addWidget(self.btn_delete)

        toolbar.addStretch()

        self.btn_import = QPushButton("📥 导入")
        self.btn_import.setStyleSheet(BTN_DARK)
        self.btn_import.clicked.connect(self.import_process)
        toolbar.addWidget(self.btn_import)

        self.btn_export = QPushButton("📤 导出")
        self.btn_export.setStyleSheet(BTN_DARK)
        self.btn_export.clicked.connect(self.export_process)
        toolbar.addWidget(self.btn_export)

        main_layout.addLayout(toolbar)

        # --- 主内容：左右分栏 ---
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：流程信息 + 步骤列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 流程基本信息
        info_group = QGroupBox("📝 流程信息")
        info_layout = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("如: PCB元件安装流程")
        info_layout.addRow("名称:", self.name_edit)

        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("流程描述...")
        info_layout.addRow("描述:", self.desc_edit)

        self.version_edit = QLineEdit("1.0")
        self.version_edit.setMaximumWidth(100)
        info_layout.addRow("版本:", self.version_edit)

        self.category_combo = QComboBox()
        self.category_combo.addItems(["general", "assembly", "welding", "inspection", "packaging"])
        info_layout.addRow("类别:", self.category_combo)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # 步骤列表
        steps_group = QGroupBox("📋 步骤列表 (拖拽排序)")
        steps_layout = QVBoxLayout()

        self.steps_list = QListWidget()
        self.steps_list.setAlternatingRowColors(True)
        self.steps_list.setDragDropMode(QListWidget.InternalMove)
        self.steps_list.currentRowChanged.connect(self.on_step_selected)
        self.steps_list.model().rowsMoved.connect(self.on_steps_reordered)
        steps_layout.addWidget(self.steps_list)

        step_btns = QHBoxLayout()
        self.btn_add_step = QPushButton("+ 添加步骤")
        self.btn_add_step.setStyleSheet(BTN_SUCCESS)
        self.btn_add_step.clicked.connect(self.add_step)
        step_btns.addWidget(self.btn_add_step)

        self.btn_del_step = QPushButton("- 删除步骤")
        self.btn_del_step.setStyleSheet(BTN_DANGER)
        self.btn_del_step.clicked.connect(self.delete_step)
        step_btns.addWidget(self.btn_del_step)

        self.btn_move_up = QPushButton("↑ 上移")
        self.btn_move_up.setStyleSheet(BTN_WARNING)
        self.btn_move_up.clicked.connect(self.move_step_up)
        step_btns.addWidget(self.btn_move_up)

        self.btn_move_down = QPushButton("↓ 下移")
        self.btn_move_down.setStyleSheet(BTN_WARNING)
        self.btn_move_down.clicked.connect(self.move_step_down)
        step_btns.addWidget(self.btn_move_down)

        steps_layout.addLayout(step_btns)
        steps_group.setLayout(steps_layout)
        left_layout.addWidget(steps_group, 1)

        splitter.addWidget(left_widget)

        # 右侧：步骤配置详情
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 步骤编辑
        step_group = QGroupBox("⚙️ 步骤配置")
        step_layout = QFormLayout()

        self.step_name_edit = QLineEdit()
        self.step_name_edit.setPlaceholderText("步骤名称...")
        step_layout.addRow("名称:", self.step_name_edit)

        self.step_desc_edit = QLineEdit()
        self.step_desc_edit.setPlaceholderText("步骤描述...")
        step_layout.addRow("描述:", self.step_desc_edit)

        # 预期动作（多选）
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.action_checks = {}
        for i, (cn, en) in enumerate(zip(ACTION_TYPES, ACTION_TYPES_EN)):
            cb = QCheckBox(f"{cn} ({en})")
            self.action_checks[en] = cb
            actions_layout.addWidget(cb)
        step_layout.addRow("预期动作:", actions_widget)

        # 时间约束
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 300)
        self.min_duration_spin.setValue(0.5)
        self.min_duration_spin.setSuffix(" s")
        step_layout.addRow("最短时间:", self.min_duration_spin)

        self.max_duration_spin = QDoubleSpinBox()
        self.max_duration_spin.setRange(0.5, 600)
        self.max_duration_spin.setValue(30.0)
        self.max_duration_spin.setSuffix(" s")
        step_layout.addRow("最长时间:", self.max_duration_spin)

        self.required_check = QCheckBox("必须执行")
        self.required_check.setChecked(True)
        step_layout.addRow("必需步骤:", self.required_check)

        self.skip_check = QCheckBox("允许跳过")
        step_layout.addRow("可跳过:", self.skip_check)

        # 告警设置
        self.alarm_timeout_check = QCheckBox("超时告警")
        self.alarm_timeout_check.setChecked(True)
        step_layout.addRow("", self.alarm_timeout_check)

        self.alarm_skip_check = QCheckBox("跳过告警")
        self.alarm_skip_check.setChecked(True)
        step_layout.addRow("", self.alarm_skip_check)

        self.alarm_error_check = QCheckBox("错误告警")
        self.alarm_error_check.setChecked(True)
        step_layout.addRow("", self.alarm_error_check)

        step_group.setLayout(step_layout)
        right_layout.addWidget(step_group, 1)

        # 全局约束
        constraint_group = QGroupBox("🔒 全局约束")
        constraint_layout = QFormLayout()

        self.strict_order_check = QCheckBox("严格顺序")
        self.strict_order_check.setChecked(True)
        constraint_layout.addRow("顺序模式:", self.strict_order_check)

        self.allow_reverse_check = QCheckBox("允许逆序")
        constraint_layout.addRow("", self.allow_reverse_check)

        self.max_total_time_spin = QDoubleSpinBox()
        self.max_total_time_spin.setRange(10, 3600)
        self.max_total_time_spin.setValue(300)
        self.max_total_time_spin.setSuffix(" s")
        constraint_layout.addRow("最大总时间:", self.max_total_time_spin)

        constraint_group.setLayout(constraint_layout)
        right_layout.addWidget(constraint_group)

        # 保存按钮
        save_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 保存流程")
        self.btn_save.setStyleSheet(BTN_SUCCESS + "font-size: 14px; padding: 12px;")
        self.btn_save.clicked.connect(self.save_process)
        save_layout.addWidget(self.btn_save)

        self.btn_apply_step = QPushButton("✅ 应用步骤修改")
        self.btn_apply_step.setStyleSheet(BTN_PRIMARY + "font-size: 14px; padding: 12px;")
        self.btn_apply_step.clicked.connect(self.apply_step_changes)
        save_layout.addWidget(self.btn_apply_step)

        right_layout.addLayout(save_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter, 1)

    # ================================================================
    #                       流程管理
    # ================================================================

    def refresh_process_list(self):
        """刷新流程列表"""
        self.process_combo.blockSignals(True)
        self.process_combo.clear()
        self.process_combo.addItem("-- 选择流程 --", None)
        for p in self.rule_engine.list_processes():
            display = f"{p['name']} v{p['version']} ({p['step_count']}步)"
            self.process_combo.addItem(display, p['process_id'])
        self.process_combo.blockSignals(False)

    def on_process_selected(self, index):
        """选择流程"""
        process_id = self.process_combo.currentData()
        if not process_id:
            self.steps_list.clear()
            return
        config = self.rule_engine.load_process(process_id)
        if config:
            self._load_config_to_ui(config)

    def _load_config_to_ui(self, config: dict):
        """将配置加载到UI"""
        self.name_edit.setText(config.get("name", ""))
        self.desc_edit.setText(config.get("description", ""))
        self.version_edit.setText(config.get("version", "1.0"))

        category = config.get("category", "general")
        idx = self.category_combo.findText(category)
        if idx >= 0:
            self.category_combo.setCurrentIndex(idx)

        constraints = config.get("constraints", {})
        self.strict_order_check.setChecked(constraints.get("strict_order", True))
        self.allow_reverse_check.setChecked(constraints.get("allow_reverse", False))
        self.max_total_time_spin.setValue(constraints.get("max_total_time", 300))

        # 加载步骤
        self.steps_list.clear()
        for step in config.get("steps", []):
            name = step.get("step_name", step.get("name", ""))
            actions = ", ".join(step.get("expected_actions", []))
            text = f"{step['step_id']}. {name}  [{actions}]"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, step)
            self.steps_list.addItem(item)

        self.current_step_idx = -1

    def create_new_process(self):
        """新建流程"""
        name, ok = self._input_dialog("新建流程", "流程名称:")
        if ok and name:
            self.rule_engine.create_process(name)
            self.refresh_process_list()
            self.process_combo.setCurrentIndex(self.process_combo.count() - 1)
            self.process_changed.emit()

    def copy_process(self):
        """复制流程"""
        process_id = self.process_combo.currentData()
        if not process_id:
            QMessageBox.warning(self, "提示", "请先选择要复制的流程")
            return
        self.rule_engine.copy_process(process_id)
        self.refresh_process_list()
        self.process_combo.setCurrentIndex(self.process_combo.count() - 1)
        self.process_changed.emit()

    def delete_process(self):
        """删除流程"""
        process_id = self.process_combo.currentData()
        if not process_id:
            return
        reply = QMessageBox.question(self, "确认", "确定要删除此流程吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.rule_engine.delete_process(process_id)
            self.refresh_process_list()
            self.steps_list.clear()
            self.process_changed.emit()

    def import_process(self):
        """导入流程"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "导入流程配置", "", "JSON文件 (*.json)"
        )
        if filepath:
            config = self.rule_engine.import_process(filepath)
            if config:
                self.refresh_process_list()
                self.process_combo.setCurrentIndex(self.process_combo.count() - 1)
                QMessageBox.information(self, "成功", "流程导入成功")
                self.process_changed.emit()
            else:
                QMessageBox.warning(self, "失败", "导入失败，请检查文件格式")

    def export_process(self):
        """导出流程"""
        process_id = self.process_combo.currentData()
        if not process_id:
            QMessageBox.warning(self, "提示", "请先选择流程")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "导出流程配置", f"{process_id}.json", "JSON文件 (*.json)"
        )
        if filepath:
            if self.rule_engine.export_process(process_id, filepath):
                QMessageBox.information(self, "成功", f"已导出到 {filepath}")
            else:
                QMessageBox.warning(self, "失败", "导出失败")

    # ================================================================
    #                       步骤管理
    # ================================================================

    def on_step_selected(self, row):
        """选择步骤时更新右侧编辑区"""
        self.current_step_idx = row
        if row < 0:
            return
        item = self.steps_list.item(row)
        if not item:
            return
        step_data = item.data(Qt.UserRole)
        if not step_data:
            return
        self._load_step_to_editor(step_data)

    def _load_step_to_editor(self, step: dict):
        """加载步骤到编辑器"""
        self.step_name_edit.setText(step.get("step_name", step.get("name", "")))
        self.step_desc_edit.setText(step.get("description", ""))
        self.min_duration_spin.setValue(step.get("min_duration", 0.5))
        self.max_duration_spin.setValue(step.get("max_duration", 30.0))
        self.required_check.setChecked(step.get("required", True))
        self.skip_check.setChecked(step.get("allow_skip", False))
        self.alarm_timeout_check.setChecked(step.get("alarm_on_timeout", True))
        self.alarm_skip_check.setChecked(step.get("alarm_on_skip", True))
        self.alarm_error_check.setChecked(step.get("alarm_on_error", True))

        actions = step.get("expected_actions", [])
        for key, cb in self.action_checks.items():
            cb.setChecked(key in actions)

    def _get_step_from_editor(self) -> dict:
        """从编辑器获取步骤数据"""
        actions = [key for key, cb in self.action_checks.items() if cb.isChecked()]
        return {
            "step_name": self.step_name_edit.text().strip(),
            "description": self.step_desc_edit.text().strip(),
            "expected_actions": actions,
            "min_duration": self.min_duration_spin.value(),
            "max_duration": self.max_duration_spin.value(),
            "required": self.required_check.isChecked(),
            "allow_skip": self.skip_check.isChecked(),
            "alarm_on_timeout": self.alarm_timeout_check.isChecked(),
            "alarm_on_skip": self.alarm_skip_check.isChecked(),
            "alarm_on_error": self.alarm_error_check.isChecked(),
        }

    def add_step(self):
        """添加步骤"""
        step_id = self.steps_list.count() + 1
        step_data = {
            "step_id": step_id,
            "step_name": f"新步骤 {step_id}",
            "expected_actions": ["REACHING"],
            "description": "",
            "min_duration": 0.5,
            "max_duration": 30.0,
            "required": True,
        }
        text = f"{step_id}. {step_data['step_name']}  [{', '.join(step_data['expected_actions'])}]"
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, step_data)
        self.steps_list.addItem(item)
        self.steps_list.setCurrentRow(self.steps_list.count() - 1)

    def delete_step(self):
        """删除步骤"""
        row = self.steps_list.currentRow()
        if row >= 0:
            self.steps_list.takeItem(row)
            self._renumber_steps()

    def move_step_up(self):
        """上移步骤"""
        row = self.steps_list.currentRow()
        if row > 0:
            item = self.steps_list.takeItem(row)
            self.steps_list.insertItem(row - 1, item)
            self.steps_list.setCurrentRow(row - 1)
            self._renumber_steps()

    def move_step_down(self):
        """下移步骤"""
        row = self.steps_list.currentRow()
        if row < self.steps_list.count() - 1:
            item = self.steps_list.takeItem(row)
            self.steps_list.insertItem(row + 1, item)
            self.steps_list.setCurrentRow(row + 1)
            self._renumber_steps()

    def on_steps_reordered(self):
        """拖拽排序后更新编号"""
        self._renumber_steps()

    def _renumber_steps(self):
        """重新编号步骤"""
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            step_data = item.data(Qt.UserRole)
            if step_data:
                step_data["step_id"] = i + 1
                name = step_data.get("step_name", step_data.get("name", ""))
                actions = ", ".join(step_data.get("expected_actions", []))
                item.setText(f"{i + 1}. {name}  [{actions}]")
                item.setData(Qt.UserRole, step_data)

    def apply_step_changes(self):
        """应用步骤编辑器的修改"""
        row = self.steps_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选择一个步骤")
            return
        step_data = self._get_step_from_editor()
        step_data["step_id"] = row + 1

        item = self.steps_list.item(row)
        actions_str = ", ".join(step_data.get("expected_actions", []))
        item.setText(f"{row + 1}. {step_data['step_name']}  [{actions_str}]")
        item.setData(Qt.UserRole, step_data)

    # ================================================================
    #                       保存流程
    # ================================================================

    def save_process(self):
        """保存当前流程"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入流程名称")
            return

        process_id = self.process_combo.currentData()
        if not process_id:
            # 新建
            config = self.rule_engine.create_process(name)
            process_id = config["process_id"]
        else:
            config = self.rule_engine.load_process(process_id) or {}

        # 更新基本信息
        config["name"] = name
        config["description"] = self.desc_edit.text().strip()
        config["version"] = self.version_edit.text().strip() or "1.0"
        config["category"] = self.category_combo.currentText()

        # 收集步骤
        steps = []
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            step_data = item.data(Qt.UserRole)
            if step_data:
                step_data["step_id"] = i + 1
                steps.append(step_data)
        config["steps"] = steps

        # 约束
        config["constraints"] = {
            "strict_order": self.strict_order_check.isChecked(),
            "allow_reverse": self.allow_reverse_check.isChecked(),
            "max_total_time": self.max_total_time_spin.value(),
            "allow_skip_steps": [],
        }

        if self.rule_engine.save_process(config):
            QMessageBox.information(self, "成功",
                                    f"流程 '{name}' 已保存，共 {len(steps)} 个步骤")
            self.refresh_process_list()
            self.process_changed.emit()
        else:
            QMessageBox.warning(self, "失败", "保存失败")

    # ================================================================
    #                       辅助
    # ================================================================

    def _input_dialog(self, title, label, default=""):
        """简单输入对话框"""
        from PyQt5.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, title, label, text=default)
        return text, ok
