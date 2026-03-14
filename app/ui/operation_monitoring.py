"""
@file operation_monitoring.py
@brief 操作监控界面 - 包含学习模式和监控模式
@author AI Assistant
@date 2026-03-10
@note 美观的现代化界面，支持视频学习录制、实时操作步骤验证
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QGroupBox, QGridLayout, QDoubleSpinBox, QCheckBox, QScrollArea,
    QFrame, QFileDialog, QLineEdit, QTextEdit, QSpinBox, QTabWidget,
    QListWidget, QListWidgetItem, QProgressBar, QSplitter, QMessageBox,
    QSlider, QStackedWidget, QSizePolicy, QSpacerItem, QToolBar, QAction
)
from typing import Dict, List, Optional
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QIcon, QPainter, QPen, QBrush
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QSize
import cv2
import numpy as np
import os
import json
from datetime import datetime

from app.core.realtime_monitor import RealTimeMonitor
from app.core.action_recognizer import ActionRecognizer, ActionType
from app.core.sequence_validator import SequenceValidator, StepDefinition, ValidationResult, MonitoringState
from app.core.rule_engine import RuleEngine
from app.core.alarm_manager import AlarmManager, AlarmType


# ======================== 样式常量 ========================
STYLE_SHEET = """
    QGroupBox {
        font-size: 13px;
        font-weight: bold;
        border: 2px solid #3498db;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        background-color: #f8f9fa;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #2c3e50;
    }
    QPushButton {
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: bold;
        border: none;
        min-height: 32px;
    }
    QPushButton:hover {
        opacity: 0.9;
    }
    QPushButton:disabled {
        background-color: #bdc3c7;
        color: #7f8c8d;
    }
    QLabel {
        font-size: 12px;
    }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
        padding: 6px;
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 12px;
        min-height: 28px;
    }
    QListWidget {
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 12px;
        alternate-background-color: #ecf0f1;
    }
    QListWidget::item {
        padding: 6px;
        border-bottom: 1px solid #ecf0f1;
    }
    QListWidget::item:selected {
        background-color: #3498db;
        color: white;
    }
    QProgressBar {
        border: 1px solid #bdc3c7;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        min-height: 22px;
    }
    QProgressBar::chunk {
        background-color: #2ecc71;
        border-radius: 5px;
    }
    QTextEdit {
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        font-size: 11px;
    }
"""

BTN_PRIMARY = "background-color: #3498db; color: white;"
BTN_SUCCESS = "background-color: #2ecc71; color: white;"
BTN_DANGER = "background-color: #e74c3c; color: white;"
BTN_WARNING = "background-color: #f39c12; color: white;"
BTN_INFO = "background-color: #1abc9c; color: white;"
BTN_DARK = "background-color: #34495e; color: white;"


class OperationMonitoringWidget(QWidget):
    """
    @class OperationMonitoringWidget
    @brief 操作监控主界面

    包含两大模式：
    1. 学习模式 - 录制标准操作流程、标记步骤、保存模板
    2. 监控模式 - 实时检测操作步骤、验证正确性、告警
    """

    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLE_SHEET)

        # 核心引擎
        self.monitor = RealTimeMonitor()
        self.action_recognizer = ActionRecognizer()
        self.sequence_validator = SequenceValidator()
        self.rule_engine = RuleEngine()
        self.alarm_manager = AlarmManager()

        # 摄像头与录制状态
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.current_mode = "learning"  # learning / monitoring

        # 学习模式状态
        self.learning_video_writer = None
        self.learning_is_recording = False
        self.learning_start_time = None
        self.learning_frames = []
        self.learning_step_markers = []
        self.learning_current_step = 0
        self.step_actions_buffer = []          # 当前步骤期间收集的动作
        self.last_step_mark_time = None        # 上一次标记步骤的时间

        # 监控模式状态
        self.monitoring_active = False

        # 输出目录
        self.output_dir = os.path.join("data", "recordings")
        self.learning_dir = os.path.join(self.output_dir, "learning")
        self.monitoring_dir = os.path.join(self.output_dir, "monitoring")
        for d in [self.output_dir, self.learning_dir, self.monitoring_dir]:
            os.makedirs(d, exist_ok=True)

        # 告警回调
        self.alarm_manager.on_alarm = self._on_alarm_triggered

        self.init_ui()
        self.load_available_cameras()
        self.refresh_process_list()

    # ================================================================
    #                         UI 构建
    # ================================================================

    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- 顶部模式切换栏 ---
        mode_bar = QHBoxLayout()
        mode_label = QLabel("操作模式:")
        mode_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        mode_bar.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("📚 学习模式", "learning")
        self.mode_combo.addItem("🔍 监控模式", "monitoring")
        self.mode_combo.setMinimumWidth(200)
        self.mode_combo.setStyleSheet("font-size: 14px; padding: 8px;")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_bar.addWidget(self.mode_combo)

        mode_bar.addStretch()

        # 流程选择
        mode_bar.addWidget(QLabel("当前流程:"))
        self.process_combo = QComboBox()
        self.process_combo.setMinimumWidth(250)
        self.process_combo.currentIndexChanged.connect(self.on_process_changed)
        mode_bar.addWidget(self.process_combo)

        self.refresh_process_btn = QPushButton("刷新")
        self.refresh_process_btn.setStyleSheet(BTN_INFO)
        self.refresh_process_btn.setFixedWidth(60)
        self.refresh_process_btn.clicked.connect(self.refresh_process_list)
        mode_bar.addWidget(self.refresh_process_btn)

        main_layout.addLayout(mode_bar)

        # --- 主内容区 (使用QSplitter实现可调大小) ---
        self.main_splitter = QSplitter(Qt.Horizontal)

        # 左侧：视频 + 控制
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 视频显示区
        video_group = QGroupBox("视频画面")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "border: 2px solid #2c3e50; border-radius: 8px; "
            "background-color: #1a1a2e;"
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("📷 等待开始...")
        self.video_label.setStyleSheet(
            self.video_label.styleSheet() + "color: #7f8c8d; font-size: 18px;"
        )
        video_layout.addWidget(self.video_label)

        # 视频信息栏
        info_bar = QHBoxLayout()
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 13px;")
        info_bar.addWidget(self.fps_label)
        self.detect_time_label = QLabel("检测: 0ms")
        self.detect_time_label.setStyleSheet("color: #3498db; font-weight: bold; font-size: 13px;")
        info_bar.addWidget(self.detect_time_label)
        self.action_label = QLabel("动作: --")
        self.action_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 13px;")
        info_bar.addWidget(self.action_label)
        self.confidence_label = QLabel("置信度: 0%")
        self.confidence_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 13px;")
        info_bar.addWidget(self.confidence_label)
        info_bar.addStretch()
        video_layout.addLayout(info_bar)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)

        # 控制按钮区
        ctrl_layout = QHBoxLayout()

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        ctrl_layout.addWidget(QLabel("摄像头:"))
        ctrl_layout.addWidget(self.camera_combo)

        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setStyleSheet(BTN_SUCCESS)
        self.start_btn.clicked.connect(self.start_operation)
        ctrl_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setStyleSheet(BTN_DANGER)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_operation)
        ctrl_layout.addWidget(self.stop_btn)

        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setStyleSheet(BTN_WARNING)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_operation)
        ctrl_layout.addWidget(self.pause_btn)

        ctrl_layout.addStretch()
        left_layout.addLayout(ctrl_layout)

        self.main_splitter.addWidget(left_widget)

        # 右侧：模式特定面板（堆叠）
        self.right_stack = QStackedWidget()

        # 学习模式右侧面板
        self.learning_panel = self._build_learning_panel()
        self.right_stack.addWidget(self.learning_panel)

        # 监控模式右侧面板
        self.monitoring_panel = self._build_monitoring_panel()
        self.right_stack.addWidget(self.monitoring_panel)

        self.main_splitter.addWidget(self.right_stack)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.main_splitter, 1)

        # --- 底部告警栏 ---
        alarm_group = QGroupBox("告警信息")
        alarm_group.setStyleSheet(
            alarm_group.styleSheet() + "border-color: #e74c3c;"
        )
        alarm_layout = QVBoxLayout()
        self.alarm_list = QListWidget()
        self.alarm_list.setMaximumHeight(120)
        self.alarm_list.setAlternatingRowColors(True)
        alarm_layout.addWidget(self.alarm_list)

        alarm_btn_layout = QHBoxLayout()
        self.clear_alarm_btn = QPushButton("清除全部告警")
        self.clear_alarm_btn.setStyleSheet(BTN_DARK)
        self.clear_alarm_btn.clicked.connect(self.clear_alarms)
        alarm_btn_layout.addWidget(self.clear_alarm_btn)
        alarm_btn_layout.addStretch()
        self.alarm_count_label = QLabel("告警: 0")
        self.alarm_count_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")
        alarm_btn_layout.addWidget(self.alarm_count_label)
        alarm_layout.addLayout(alarm_btn_layout)

        alarm_group.setLayout(alarm_layout)
        main_layout.addWidget(alarm_group)

    def _build_learning_panel(self) -> QWidget:
        """构建学习模式面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # 流程信息
        info_group = QGroupBox("📝 流程信息")
        info_layout = QGridLayout()

        info_layout.addWidget(QLabel("流程名称:"), 0, 0)
        self.learn_name_edit = QLineEdit()
        self.learn_name_edit.setPlaceholderText("输入流程名称...")
        info_layout.addWidget(self.learn_name_edit, 0, 1)

        info_layout.addWidget(QLabel("描述:"), 1, 0)
        self.learn_desc_edit = QLineEdit()
        self.learn_desc_edit.setPlaceholderText("流程描述...")
        info_layout.addWidget(self.learn_desc_edit, 1, 1)

        info_layout.addWidget(QLabel("版本:"), 2, 0)
        self.learn_version_edit = QLineEdit("1.0")
        self.learn_version_edit.setMaximumWidth(80)
        info_layout.addWidget(self.learn_version_edit, 2, 1)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # 录制控制
        record_group = QGroupBox("🎥 录制控制")
        record_layout = QVBoxLayout()

        record_btn_layout = QHBoxLayout()
        self.learn_record_btn = QPushButton("● 开始录制")
        self.learn_record_btn.setStyleSheet(BTN_DANGER + "font-size: 14px;")
        self.learn_record_btn.clicked.connect(self.toggle_learning_record)
        record_btn_layout.addWidget(self.learn_record_btn)

        self.learn_mark_step_btn = QPushButton("📌 标记步骤")
        self.learn_mark_step_btn.setStyleSheet(BTN_PRIMARY)
        self.learn_mark_step_btn.setEnabled(False)
        self.learn_mark_step_btn.clicked.connect(self.mark_learning_step)
        record_btn_layout.addWidget(self.learn_mark_step_btn)
        record_layout.addLayout(record_btn_layout)

        self.learn_status_label = QLabel("状态: 就绪")
        self.learn_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #7f8c8d;")
        record_layout.addWidget(self.learn_status_label)

        self.learn_timer_label = QLabel("⏱ 00:00:00")
        self.learn_timer_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        self.learn_timer_label.setAlignment(Qt.AlignCenter)
        record_layout.addWidget(self.learn_timer_label)

        record_group.setLayout(record_layout)
        layout.addWidget(record_group)

        # 已识别步骤列表
        steps_group = QGroupBox("📋 已标记的步骤")
        steps_layout = QVBoxLayout()

        self.learn_steps_list = QListWidget()
        self.learn_steps_list.setAlternatingRowColors(True)
        steps_layout.addWidget(self.learn_steps_list)

        step_edit_layout = QHBoxLayout()
        self.learn_step_name_edit = QLineEdit()
        self.learn_step_name_edit.setPlaceholderText("步骤名称...")
        step_edit_layout.addWidget(self.learn_step_name_edit)

        self.learn_del_step_btn = QPushButton("删除")
        self.learn_del_step_btn.setStyleSheet(BTN_DANGER)
        self.learn_del_step_btn.clicked.connect(self.delete_learning_step)
        step_edit_layout.addWidget(self.learn_del_step_btn)
        steps_layout.addLayout(step_edit_layout)

        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group, 1)

        # 保存
        save_layout = QHBoxLayout()
        self.learn_save_btn = QPushButton("💾 保存流程")
        self.learn_save_btn.setStyleSheet(BTN_SUCCESS + "font-size: 14px; padding: 10px;")
        self.learn_save_btn.clicked.connect(self.save_learning_process)
        save_layout.addWidget(self.learn_save_btn)

        self.learn_load_video_btn = QPushButton("📂 加载视频学习")
        self.learn_load_video_btn.setStyleSheet(BTN_INFO + "font-size: 14px; padding: 10px;")
        self.learn_load_video_btn.clicked.connect(self.load_video_for_learning)
        save_layout.addWidget(self.learn_load_video_btn)

        layout.addLayout(save_layout)

        return panel

    def _build_monitoring_panel(self) -> QWidget:
        """构建监控模式面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # 监控状态
        status_group = QGroupBox("📊 监控状态")
        status_layout = QVBoxLayout()

        self.monitor_state_label = QLabel("状态: 未启动")
        self.monitor_state_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #7f8c8d; "
            "padding: 8px; background-color: #ecf0f1; border-radius: 6px;"
        )
        self.monitor_state_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.monitor_state_label)

        self.monitor_progress = QProgressBar()
        self.monitor_progress.setRange(0, 100)
        self.monitor_progress.setValue(0)
        self.monitor_progress.setFormat("%v% 已完成")
        status_layout.addWidget(self.monitor_progress)

        stat_grid = QGridLayout()
        self.monitor_step_label = QLabel("当前步骤: --")
        self.monitor_step_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        stat_grid.addWidget(self.monitor_step_label, 0, 0)

        self.monitor_remaining_label = QLabel("剩余时间: --")
        self.monitor_remaining_label.setStyleSheet("font-size: 13px;")
        stat_grid.addWidget(self.monitor_remaining_label, 0, 1)

        self.monitor_error_label = QLabel("错误: 0")
        self.monitor_error_label.setStyleSheet("font-size: 13px; color: #e74c3c;")
        stat_grid.addWidget(self.monitor_error_label, 1, 0)

        self.monitor_warning_label = QLabel("警告: 0")
        self.monitor_warning_label.setStyleSheet("font-size: 13px; color: #f39c12;")
        stat_grid.addWidget(self.monitor_warning_label, 1, 1)

        status_layout.addLayout(stat_grid)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 步骤进度条列表
        progress_group = QGroupBox("📋 步骤进度")
        progress_layout = QVBoxLayout()
        self.step_progress_list = QListWidget()
        self.step_progress_list.setAlternatingRowColors(True)
        self.step_progress_list.setMinimumHeight(200)
        progress_layout.addWidget(self.step_progress_list)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group, 1)

        # 姿态信息
        pose_group = QGroupBox("🦴 姿态信息")
        pose_layout = QGridLayout()

        self.monitor_person_label = QLabel("检测人数: 0")
        self.monitor_person_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        pose_layout.addWidget(self.monitor_person_label, 0, 0, 1, 2)

        self.monitor_angle_labels = {}
        joint_names = ["左肘", "右肘", "左肩", "右肩", "左髋", "右髋", "左膝", "右膝"]
        for i, joint in enumerate(joint_names):
            lbl = QLabel(f"{joint}: --°")
            lbl.setStyleSheet("font-size: 11px; padding: 2px;")
            self.monitor_angle_labels[joint] = lbl
            pose_layout.addWidget(lbl, 1 + i // 2, i % 2)

        pose_group.setLayout(pose_layout)
        layout.addWidget(pose_group)

        # 监控控制
        monitor_ctrl = QHBoxLayout()
        self.monitor_start_btn = QPushButton("▶ 开始监控")
        self.monitor_start_btn.setStyleSheet(BTN_SUCCESS + "font-size: 14px; padding: 10px;")
        self.monitor_start_btn.clicked.connect(self.start_monitoring_process)
        monitor_ctrl.addWidget(self.monitor_start_btn)

        self.monitor_reset_btn = QPushButton("🔄 重置")
        self.monitor_reset_btn.setStyleSheet(BTN_WARNING + "font-size: 14px; padding: 10px;")
        self.monitor_reset_btn.clicked.connect(self.reset_monitoring)
        monitor_ctrl.addWidget(self.monitor_reset_btn)

        layout.addLayout(monitor_ctrl)

        return panel

    # ================================================================
    #                      摄像头 / 视频控制
    # ================================================================

    def load_available_cameras(self):
        """检测可用摄像头"""
        self.camera_combo.clear()
        import sys
        from io import StringIO
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            found = False
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        self.camera_combo.addItem(f"摄像头 {i}", i)
                        found = True
                        cap.release()
                except Exception:
                    continue
            if not found:
                self.camera_combo.addItem("未检测到摄像头", -1)
        finally:
            sys.stderr = original_stderr

    def on_mode_changed(self, index):
        """模式切换"""
        mode = self.mode_combo.currentData()
        self.current_mode = mode
        self.right_stack.setCurrentIndex(0 if mode == "learning" else 1)

    def on_process_changed(self, index):
        """流程选择变化"""
        process_id = self.process_combo.currentData()
        if process_id:
            config = self.rule_engine.load_process(process_id)
            if config:
                self._update_step_progress_list(config)
                self.learn_name_edit.setText(config.get("name", ""))
                self.learn_desc_edit.setText(config.get("description", ""))
                self.learn_version_edit.setText(config.get("version", "1.0"))

    def refresh_process_list(self):
        """刷新流程列表"""
        self.process_combo.clear()
        self.process_combo.addItem("-- 选择流程 --", None)
        processes = self.rule_engine.list_processes()
        for p in processes:
            display = f"{p['name']} v{p['version']} ({p['step_count']}步)"
            self.process_combo.addItem(display, p['process_id'])

    def start_operation(self):
        """开始操作（学习 / 监控共用摄像头）"""
        camera_data = self.camera_combo.currentData()
        if camera_data is None or camera_data < 0:
            QMessageBox.warning(self, "提示", "请选择有效的摄像头")
            return

        # 设置姿态估计模式
        self.monitor.set_monitor_mode("pose_analysis")
        self.monitor.pose_conf_threshold = 0.5

        import sys
        from io import StringIO
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            self.cap = cv2.VideoCapture(camera_data)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", f"无法打开摄像头 {camera_data}")
                return
        finally:
            sys.stderr = old_stderr

        self.is_running = True
        self.timer.start(33)  # ~30fps
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)

    def stop_operation(self):
        """停止操作"""
        if self.learning_is_recording:
            self.stop_learning_record()
        self.is_running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.video_label.setText("📷 已停止")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.monitoring_active = False

    def pause_operation(self):
        """暂停/恢复"""
        if self.timer.isActive():
            self.timer.stop()
            self.pause_btn.setText("▶ 继续")
            self.pause_btn.setStyleSheet(BTN_SUCCESS)
            if self.monitoring_active:
                self.sequence_validator.pause()
        else:
            self.timer.start(33)
            self.pause_btn.setText("⏸ 暂停")
            self.pause_btn.setStyleSheet(BTN_WARNING)
            if self.monitoring_active:
                self.sequence_validator.resume()

    # ================================================================
    #                         帧更新
    # ================================================================

    def update_frame(self):
        """更新帧（核心循环）"""
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            # 如果是视频文件播放完毕
            if hasattr(self, '_video_file_mode') and self._video_file_mode:
                self.stop_operation()
                self.learn_status_label.setText("状态: 视频播放完毕")
            return

        # 运行姿态估计
        annotated_frame, detection_time, pose_results = self.monitor.process_pose_frame(frame)

        # 更新FPS/检测时间
        fps = int(1000 / max(detection_time, 1))
        self.fps_label.setText(f"FPS: {fps}")
        self.detect_time_label.setText(f"检测: {detection_time:.1f}ms")

        # 获取关键点并识别动作
        action = ActionType.IDLE
        confidence = 0.0
        keypoints_data = None

        if pose_results and len(pose_results) > 0:
            first_person = pose_results[0]
            kps = first_person.get("keypoints", None)
            angles = first_person.get("angles", {})

            if kps is not None:
                # keypoints可能是list（来自tolist()），转为numpy数组
                if isinstance(kps, list):
                    kps = np.array(kps)
                keypoints_data = kps
                action, confidence = self.action_recognizer.recognize_action(kps)

            # 更新姿态角度
            if self.current_mode == "monitoring":
                self.monitor_person_label.setText(f"检测人数: {len(pose_results)}")
                for joint_name, angle in angles.items():
                    if joint_name in self.monitor_angle_labels:
                        self.monitor_angle_labels[joint_name].setText(f"{joint_name}: {angle}°")

        # 更新动作信息
        self.action_label.setText(f"动作: {action.value}")
        self.confidence_label.setText(f"置信度: {confidence:.0%}")

        # 模式特定处理
        if self.current_mode == "learning":
            self._process_learning_frame(frame, annotated_frame, action, confidence,
                                          keypoints_data, detection_time)
        elif self.current_mode == "monitoring" and self.monitoring_active:
            self._process_monitoring_frame(frame, annotated_frame, action, confidence,
                                            keypoints_data, detection_time)

        # 在视频上叠加动作信息
        self._draw_action_overlay(annotated_frame, action, confidence)

        # 显示帧
        self._display_frame(annotated_frame)

        # 录制帧
        if self.learning_is_recording and self.learning_video_writer is not None:
            self.learning_video_writer.write(frame)

    def _draw_action_overlay(self, frame, action: ActionType, confidence: float):
        """在帧上绘制动作信息叠加层"""
        h, w = frame.shape[:2]
        # 动作标签背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 60), (300, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 动作文本
        color = (0, 255, 0) if action != ActionType.IDLE else (200, 200, 200)
        text = f"Action: {action.value} ({confidence:.0%})"
        cv2.putText(frame, text, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 模式标签
        mode_text = "LEARNING" if self.current_mode == "learning" else "MONITORING"
        mode_color = (0, 200, 255) if self.current_mode == "learning" else (0, 255, 100)
        cv2.putText(frame, mode_text, (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

        # 录制指示器
        if self.learning_is_recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _display_frame(self, frame):
        """将OpenCV帧显示到QLabel"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    # ================================================================
    #                      学习模式逻辑
    # ================================================================

    def toggle_learning_record(self):
        """切换学习录制状态"""
        if self.learning_is_recording:
            self.stop_learning_record()
        else:
            self.start_learning_record()

    def start_learning_record(self):
        """开始学习录制"""
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "提示", "请先点击「开始」打开摄像头")
            return

        name = self.learn_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入流程名称")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learn_{name}_{timestamp}.mp4"
        filepath = os.path.join(self.learning_dir, filename)

        fps = 30.0
        fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.learning_video_writer = cv2.VideoWriter(filepath, fourcc, fps, (fw, fh))

        if not self.learning_video_writer.isOpened():
            QMessageBox.warning(self, "错误", "无法创建视频文件")
            self.learning_video_writer = None
            return

        self.learning_is_recording = True
        self.learning_start_time = datetime.now()
        self.learning_frames = []
        self.learning_step_markers = []
        self.learning_current_step = 0
        self.step_actions_buffer = []
        self.last_step_mark_time = datetime.now()
        self.learn_steps_list.clear()

        self.learn_record_btn.setText("⏹ 停止录制")
        self.learn_record_btn.setStyleSheet(BTN_DARK + "font-size: 14px;")
        self.learn_mark_step_btn.setEnabled(True)
        self.learn_status_label.setText("状态: 🔴 录制中")
        self.learn_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #e74c3c;")

    def stop_learning_record(self):
        """停止学习录制"""
        if self.learning_video_writer:
            self.learning_video_writer.release()
            self.learning_video_writer = None

        self.learning_is_recording = False
        duration = ""
        if self.learning_start_time:
            d = datetime.now() - self.learning_start_time
            duration = str(d).split('.')[0]
            self.learning_start_time = None

        self.learn_record_btn.setText("● 开始录制")
        self.learn_record_btn.setStyleSheet(BTN_DANGER + "font-size: 14px;")
        self.learn_mark_step_btn.setEnabled(False)
        self.learn_status_label.setText(f"状态: ✅ 录制完成 (时长: {duration})")
        self.learn_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #2ecc71;")

    def mark_learning_step(self):
        """标记学习步骤 - 分析步骤期间收集的所有动作"""
        step_name = self.learn_step_name_edit.text().strip()
        if not step_name:
            step_name = f"步骤 {self.learning_current_step + 1}"

        self.learning_current_step += 1
        now = datetime.now()
        elapsed = ""
        if self.learning_start_time:
            d = now - self.learning_start_time
            elapsed = str(d).split('.')[0]

        # ===== 分析步骤期间的动作分布 =====
        # 统计buffer中所有非空闲动作
        action_counts = {}
        for frame in self.step_actions_buffer:
            act = frame["action"]
            if act != "空闲":  # 过滤掉空闲帧
                action_counts[act] = action_counts.get(act, 0) + 1

        total_frames = len(self.step_actions_buffer)
        # 如果全是空闲，也记录空闲
        if not action_counts and total_frames > 0:
            action_counts["空闲"] = total_frames

        # 取频次最高的动作作为主要动作，所有检测到的动作都作为expected
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        detected_actions = [a for a, _ in sorted_actions]
        primary_action = detected_actions[0] if detected_actions else "未知"

        # 计算这段步骤的实际持续时间
        step_duration = 0.0
        if self.last_step_mark_time:
            step_duration = (now - self.last_step_mark_time).total_seconds()
        elif self.learning_start_time:
            step_duration = (now - self.learning_start_time).total_seconds()

        # 构建动作分布描述
        dist_str = ", ".join([f"{a}({c})" for a, c in sorted_actions[:3]])

        marker = {
            "step_id": self.learning_current_step,
            "step_name": step_name,
            "timestamp": elapsed,
            "detected_actions": detected_actions,     # 所有检测到的动作列表
            "primary_action": primary_action,          # 主要动作
            "action_distribution": dict(sorted_actions),  # 动作分布
            "total_frames": total_frames,
            "step_duration": step_duration,
        }
        self.learning_step_markers.append(marker)

        item_text = (f"✓ {self.learning_current_step}. {step_name}  [{elapsed}]  "
                     f"主动作: {primary_action}  分布: {dist_str}  时长: {step_duration:.1f}s")
        item = QListWidgetItem(item_text)
        item.setForeground(QColor("#2ecc71"))
        self.learn_steps_list.addItem(item)
        self.learn_step_name_edit.clear()

        # 重置buffer，开始收集下一步骤的动作
        self.step_actions_buffer = []
        self.last_step_mark_time = now

    def delete_learning_step(self):
        """删除选中的学习步骤"""
        row = self.learn_steps_list.currentRow()
        if row >= 0:
            self.learn_steps_list.takeItem(row)
            if row < len(self.learning_step_markers):
                self.learning_step_markers.pop(row)

    def save_learning_process(self):
        """保存学习到的流程 - 使用学习期间实际采集的动作数据"""
        name = self.learn_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入流程名称")
            return
        if not self.learning_step_markers:
            QMessageBox.warning(self, "提示", "请先录制并标记至少一个步骤")
            return

        desc = self.learn_desc_edit.text().strip()
        version = self.learn_version_edit.text().strip() or "1.0"

        # 创建流程
        config = self.rule_engine.create_process(name, desc, version)

        # 添加步骤 - 使用学习期间实际采集的动作
        for marker in self.learning_step_markers:
            # expected_actions: 使用学习期间检测到的真实动作名称
            learned_actions = marker.get("detected_actions", [])
            if not learned_actions:
                # 兼容旧格式
                old_action = marker.get("detected_action", "空闲")
                learned_actions = [old_action]

            # 计算合理的时间约束
            step_dur = marker.get("step_duration", 3.0)
            min_dur = max(0.5, step_dur * 0.3)   # 最少为实际时长的30%
            max_dur = max(10.0, step_dur * 3.0)   # 最多为实际时长的3倍

            # 构建动作分布描述
            dist = marker.get("action_distribution", {})
            dist_desc = ", ".join([f"{a}:{c}帧" for a, c in dist.items()]) if dist else ""

            step_data = {
                "step_id": marker["step_id"],
                "step_name": marker["step_name"],
                "expected_actions": learned_actions,
                "description": f"学习录制 | 主动作: {marker.get('primary_action', '未知')} | 分布: {dist_desc}",
                "min_duration": round(min_dur, 1),
                "max_duration": round(max_dur, 1),
                "required": True,
            }
            self.rule_engine.add_step(step_data)

        # 设置全局约束
        config["constraints"] = {
            "strict_order": True,
            "allow_reverse": False,
            "match_ratio_threshold": 0.3,  # 动作匹配率阈值
        }
        self.rule_engine.save_process(config)

        QMessageBox.information(self, "成功",
                                f"流程 '{name}' 已保存，共 {len(self.learning_step_markers)} 个步骤\n"
                                f"每个步骤的动作已从学习录制中自动提取")
        self.refresh_process_list()

    def load_video_for_learning(self):
        """加载视频文件进行学习"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", self.output_dir,
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if not filepath:
            return

        # 停止当前摄像头
        if self.is_running:
            self.stop_operation()

        # 打开视频文件
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return

        self._video_file_mode = True
        self.monitor.set_monitor_mode("pose_analysis")
        self.monitor.pose_conf_threshold = 0.5

        self.is_running = True
        self.timer.start(33)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.learn_mark_step_btn.setEnabled(True)
        self.learn_status_label.setText(f"状态: 📹 播放视频 - {os.path.basename(filepath)}")
        self.learn_status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #3498db;")
        self.learning_start_time = datetime.now()

    def _process_learning_frame(self, raw_frame, annotated_frame, action, confidence,
                                 keypoints_data, detection_time):
        """处理学习模式帧 - 收集每帧的动作用于步骤学习"""
        # 更新计时
        if self.learning_start_time:
            elapsed = datetime.now() - self.learning_start_time
            elapsed_str = str(elapsed).split('.')[0]
            self.learn_timer_label.setText(f"⏱ {elapsed_str}")

        # 记录帧数据（用于后续特征提取）
        if self.learning_is_recording and keypoints_data is not None:
            now_ts = datetime.now().timestamp()
            self.learning_frames.append({
                "timestamp": now_ts,
                "action": action.value,
                "confidence": confidence,
            })
            # 持续收集当前步骤期间的动作（排除空闲，保留有效动作）
            self.step_actions_buffer.append({
                "timestamp": now_ts,
                "action": action.value,
                "confidence": confidence,
            })

    # ================================================================
    #                      监控模式逻辑
    # ================================================================

    def start_monitoring_process(self):
        """开始监控流程"""
        process_id = self.process_combo.currentData()
        if not process_id:
            QMessageBox.warning(self, "提示", "请先选择要监控的流程")
            return

        config = self.rule_engine.load_process(process_id)
        if not config:
            QMessageBox.warning(self, "错误", "无法加载流程配置")
            return

        # 加载步骤到验证器（load_from_config 内部会调用 load_steps）
        self.sequence_validator.load_from_config(config)
        self.sequence_validator.start()

        # 加载告警配置
        self.alarm_manager.load_config(config.get("alarms", {}))
        self.alarm_manager.clear_history()

        self.monitoring_active = True
        self._update_step_progress_list(config)
        self.monitor_state_label.setText("状态: 🟢 监控中")
        self.monitor_state_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #2ecc71; "
            "padding: 8px; background-color: #eafaf1; border-radius: 6px;"
        )
        self.monitor_start_btn.setEnabled(False)

    def reset_monitoring(self):
        """重置监控"""
        self.monitoring_active = False
        self.sequence_validator.reset()
        self.alarm_manager.clear_history()
        self.alarm_list.clear()
        self.alarm_count_label.setText("告警: 0")

        self.monitor_state_label.setText("状态: 未启动")
        self.monitor_state_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #7f8c8d; "
            "padding: 8px; background-color: #ecf0f1; border-radius: 6px;"
        )
        self.monitor_progress.setValue(0)
        self.monitor_step_label.setText("当前步骤: --")
        self.monitor_remaining_label.setText("剩余时间: --")
        self.monitor_error_label.setText("错误: 0")
        self.monitor_warning_label.setText("警告: 0")
        self.step_progress_list.clear()
        self.monitor_start_btn.setEnabled(True)

    def _process_monitoring_frame(self, raw_frame, annotated_frame, action, confidence,
                                   keypoints_data, detection_time):
        """处理监控模式帧"""
        # 验证当前动作
        result, message = self.sequence_validator.validate(action.value)

        # 更新进度
        progress = self.sequence_validator.get_progress()
        self.monitor_progress.setValue(int(progress["progress_percent"]))
        self.monitor_step_label.setText(f"当前步骤: {progress['current_step_name'] or '--'}")

        remaining = progress.get("remaining_time")
        if remaining is not None:
            self.monitor_remaining_label.setText(f"剩余时间: {remaining:.1f}s")
        else:
            self.monitor_remaining_label.setText("剩余时间: --")

        self.monitor_error_label.setText(f"错误: {progress['error_count']}")
        self.monitor_warning_label.setText(f"警告: {progress['warning_count']}")

        # 更新步骤列表
        step_details = self.sequence_validator.get_step_details()
        self.step_progress_list.clear()
        for sd in step_details:
            status_icon = {
                "待执行": "○",
                "进行中": "●",
                "已完成": "✓",
                "超时": "⏱",
                "失败": "✗",
            }.get(sd["status"], "?")

            color = {
                "待执行": "#7f8c8d",
                "进行中": "#3498db",
                "已完成": "#2ecc71",
                "超时": "#f39c12",
                "失败": "#e74c3c",
            }.get(sd["status"], "#7f8c8d")

            duration_str = f"  {sd['duration']:.1f}s" if sd["duration"] > 0 else ""
            item = QListWidgetItem(
                f"  {status_icon}  {sd['step_id']}. {sd['step_name']}"
                f"  [{sd['status']}]{duration_str}"
            )
            item.setForeground(QColor(color))
            font = QFont()
            font.setBold(sd["status"] == "进行中")
            font.setPointSize(11)
            item.setFont(font)
            self.step_progress_list.addItem(item)

        # 处理告警
        if result in (ValidationResult.INVALID_ORDER, ValidationResult.TIMEOUT,
                       ValidationResult.INVALID_ACTION, ValidationResult.MISSING_STEP):
            self.alarm_manager.trigger_from_validation(
                result.value, message,
                step_id=progress.get("current_step"),
                frame=raw_frame,
                detected_action=action.value
            )

        # 更新状态
        state = progress["state"]
        state_styles = {
            "空闲": ("未启动", "#7f8c8d", "#ecf0f1"),
            "等待动作": ("等待动作", "#f39c12", "#fef9e7"),
            "进行中": ("检测中", "#3498db", "#ebf5fb"),
            "已完成": ("✅ 全部完成", "#2ecc71", "#eafaf1"),
            "错误": ("❌ 错误", "#e74c3c", "#fdedec"),
            "暂停": ("⏸ 已暂停", "#7f8c8d", "#ecf0f1"),
        }
        text, color, bg = state_styles.get(state, ("未知", "#7f8c8d", "#ecf0f1"))
        self.monitor_state_label.setText(f"状态: {text}")
        self.monitor_state_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {color}; "
            f"padding: 8px; background-color: {bg}; border-radius: 6px;"
        )

    def _update_step_progress_list(self, config: Dict):
        """从配置更新步骤进度列表（初始化）"""
        self.step_progress_list.clear()
        for step in config.get("steps", []):
            name = step.get("step_name", step.get("name", ""))
            item = QListWidgetItem(f"  ○  {step['step_id']}. {name}  [待执行]")
            item.setForeground(QColor("#7f8c8d"))
            self.step_progress_list.addItem(item)

    # ================================================================
    #                         告警处理
    # ================================================================

    def _on_alarm_triggered(self, alarm):
        """告警触发回调"""
        level_colors = {
            "提示": "#1abc9c",
            "警告": "#f39c12",
            "错误": "#e74c3c",
            "严重": "#8e44ad",
        }
        color = level_colors.get(alarm.level.value, "#7f8c8d")
        text = f"[{alarm.level.value}] {alarm.timestamp[:19]} - {alarm.message}"
        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        font = QFont()
        font.setBold(alarm.level.value in ("错误", "严重"))
        item.setFont(font)
        self.alarm_list.insertItem(0, item)
        self.alarm_list.scrollToTop()

        total = len(self.alarm_manager.alarm_history)
        self.alarm_count_label.setText(f"告警: {total}")

    def clear_alarms(self):
        """清除告警"""
        self.alarm_manager.acknowledge_all()
        self.alarm_list.clear()
        self.alarm_count_label.setText("告警: 0")

    # ================================================================
    #                         清理
    # ================================================================

    def closeEvent(self, event):
        """关闭事件"""
        self.stop_operation()
        super().closeEvent(event)
