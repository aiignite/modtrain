"""
@file realtime_monitoring.py
@brief 实时监控UI模块，提供摄像头监控、目标检测、人体姿态分析和录屏功能
@author AI Assistant
@date 2026-03-09
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QSlider, QGroupBox, QGridLayout,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea,
                             QFrame, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
import numpy as np
import os
from datetime import datetime

from app.core.realtime_monitor import RealTimeMonitor


class RealtimeMonitoringWidget(QWidget):
    """
    @class RealtimeMonitoringWidget
    @brief 实时监控UI组件
    
    支持功能：
    - 缺陷检测模式：使用自定义模型进行目标检测
    - 人体姿态分析模式：使用姿态估计模型进行人体关节分析
    - 录屏功能：录制监控视频并保存
    """

    def __init__(self):
        """
        @brief 初始化实时监控UI组件
        """
        super().__init__()
        self.init_ui()

        self.monitor = RealTimeMonitor()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.video_writer = None
        self.is_recording = False
        self.recording_start_time = None
        self.output_dir = os.path.join("data", "recordings")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.load_available_models()

    def init_ui(self):
        """
        @brief 初始化UI界面
        """
        main_layout = QVBoxLayout(self)

        video_group = QGroupBox("视频流")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)

        control_group = QGroupBox("控制")
        control_layout = QGridLayout()

        control_layout.addWidget(QLabel("摄像头:"), 0, 0)
        self.camera_combo = QComboBox()
        self.detect_cameras()
        control_layout.addWidget(self.camera_combo, 0, 1)

        control_layout.addWidget(QLabel("监控模式:"), 1, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("缺陷检测", "detection")
        self.mode_combo.addItem("人体姿态分析", "pose_analysis")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        control_layout.addWidget(self.mode_combo, 1, 1)

        control_layout.addWidget(QLabel("模型:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem("选择模型...")
        control_layout.addWidget(self.model_combo, 2, 1)
        
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        control_layout.addWidget(self.browse_model_btn, 2, 2)

        self.multi_model_check = QCheckBox("多模型检测(同时加载COCO)")
        self.multi_model_check.setChecked(False)
        control_layout.addWidget(self.multi_model_check, 3, 0, 1, 3)

        control_layout.addWidget(QLabel("置信度阈值:"), 4, 0)
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.1, 0.99)
        self.conf_threshold.setValue(0.5)
        self.conf_threshold.setSingleStep(0.05)
        control_layout.addWidget(self.conf_threshold, 4, 1)

        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout, 5, 0, 1, 2)
        
        self.record_button = QPushButton("开始录屏")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        self.record_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_layout.addWidget(self.record_button, 6, 0, 1, 2)
        
        self.recording_status_label = QLabel("")
        self.recording_status_label.setStyleSheet("color: red; font-weight: bold;")
        control_layout.addWidget(self.recording_status_label, 7, 0, 1, 3)

        control_group.setLayout(control_layout)

        self.stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()

        self.fps_label = QLabel("FPS: 0")
        self.detection_time_label = QLabel("检测时间: 0 ms")
        self.defect_count_label = QLabel("缺陷数量: 0")
        self.defect_rate_label = QLabel("不良率: 0%")
        
        self.class_stats_label = QLabel("检测类别:")
        self.class_stats_label.setWordWrap(True)

        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.detection_time_label)
        stats_layout.addWidget(self.defect_count_label)
        stats_layout.addWidget(self.defect_rate_label)
        stats_layout.addWidget(self.class_stats_label)

        self.stats_group.setLayout(stats_layout)

        self.pose_stats_group = QGroupBox("姿态分析")
        pose_stats_layout = QVBoxLayout()
        
        self.person_count_label = QLabel("检测人数: 0")
        self.person_count_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        pose_stats_layout.addWidget(self.person_count_label)
        
        self.pose_angles_widget = QWidget()
        self.pose_angles_layout = QVBoxLayout(self.pose_angles_widget)
        self.pose_angles_layout.setContentsMargins(0, 0, 0, 0)
        
        self.angle_labels = {}
        joint_names = ["左肘", "右肘", "左肩", "右肩", "左髋", "右髋", "左膝", "右膝"]
        for joint in joint_names:
            label = QLabel(f"{joint}: --°")
            label.setStyleSheet("font-size: 12px; padding: 2px;")
            self.angle_labels[joint] = label
            self.pose_angles_layout.addWidget(label)
        
        self.pose_angles_layout.addStretch()
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.pose_angles_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setMinimumHeight(200)
        pose_stats_layout.addWidget(scroll_area)
        
        self.pose_stats_group.setLayout(pose_stats_layout)
        self.pose_stats_group.setVisible(False)

        right_layout = QVBoxLayout()
        right_layout.addWidget(control_group)
        right_layout.addWidget(self.stats_group)
        right_layout.addWidget(self.pose_stats_group)
        right_layout.addStretch()

        top_layout = QHBoxLayout()
        top_layout.addWidget(video_group, 2)
        top_layout.addLayout(right_layout, 1)

        main_layout.addLayout(top_layout)
    
    def on_mode_changed(self, index):
        """
        @brief 监控模式切换回调
        @param index: 选择的索引
        """
        mode = self.mode_combo.currentData()
        
        if mode == "pose_analysis":
            self.multi_model_check.setEnabled(False)
            self.multi_model_check.setChecked(False)
            self.defect_count_label.setVisible(False)
            self.defect_rate_label.setVisible(False)
            self.class_stats_label.setVisible(False)
            self.stats_group.setVisible(False)
            self.pose_stats_group.setVisible(True)
            
            self.model_combo.setEnabled(False)
            self.browse_model_btn.setEnabled(False)
        else:
            self.multi_model_check.setEnabled(True)
            self.defect_count_label.setVisible(True)
            self.defect_rate_label.setVisible(True)
            self.class_stats_label.setVisible(True)
            self.stats_group.setVisible(True)
            self.pose_stats_group.setVisible(False)
            
            self.model_combo.setEnabled(True)
            self.browse_model_btn.setEnabled(True)
    
    def load_available_models(self):
        """加载可用的模型"""
        for i in range(self.model_combo.count() - 1, 0, -1):
            self.model_combo.removeItem(i)
        
        train_runs_dir = os.path.join("runs", "train")
        if os.path.exists(train_runs_dir):
            for exp_dir in os.listdir(train_runs_dir):
                exp_path = os.path.join(train_runs_dir, exp_dir)
                if os.path.isdir(exp_path):
                    weights_dir = os.path.join(exp_path, "weights")
                    if os.path.exists(weights_dir):
                        for weight_file in os.listdir(weights_dir):
                            if weight_file.endswith('.pt'):
                                model_path = os.path.join(weights_dir, weight_file)
                                display_name = f"{exp_dir}/{weight_file}"
                                self.model_combo.addItem(display_name, model_path)
        
        yolo_dir = "yolo"
        if os.path.exists(yolo_dir):
            for weight_file in os.listdir(yolo_dir):
                if weight_file.endswith('.weights'):
                    model_path = os.path.join(yolo_dir, weight_file)
                    display_name = f"yolo/{weight_file}"
                    self.model_combo.addItem(display_name, model_path)
    
    def browse_model(self):
        """浏览选择模型文件"""
        from PyQt5.QtWidgets import QFileDialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", 
            "Model Files (*.pt *.weights *.onnx);;All Files (*)",
            options=options
        )
        if file_path:
            exists = False
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == file_path:
                    self.model_combo.setCurrentIndex(i)
                    exists = True
                    break
            
            if not exists:
                self.model_combo.addItem(os.path.basename(file_path), file_path)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def detect_cameras(self):
        """检测可用摄像头"""
        import sys
        from io import StringIO

        original_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            cameras_found = False
            for i in range(10):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        self.camera_combo.addItem(f"摄像头 {i}")
                        cameras_found = True
                        cap.release()
                except Exception:
                    continue

            if not cameras_found:
                self.camera_combo.addItem("未检测到摄像头")
        finally:
            sys.stderr = original_stderr

    def start_monitoring(self):
        """开始监控"""
        mode = self.mode_combo.currentData()
        
        if mode == "detection":
            if self.model_combo.currentIndex() == 0:
                print("错误: 请先选择模型")
                return
            
            model_path = self.model_combo.currentData()
            if not model_path:
                print("错误: 模型路径无效")
                return
            
            if not self.monitor.load_model(model_path):
                print(f"错误: 无法加载模型 {model_path}")
                return
            
            if self.multi_model_check.isChecked():
                self.monitor.load_coco_model()
            
            self.monitor.conf_threshold = self.conf_threshold.value()
        else:
            self.monitor.set_monitor_mode("pose_analysis")
            self.monitor.pose_conf_threshold = self.conf_threshold.value()

        camera_index = self.camera_combo.currentIndex()

        import os
        os.environ['OPENCV_VIDEOIO_PRIORITY_OBSENSOR'] = '0'
        
        if camera_index < 0 or self.camera_combo.currentText() == "未检测到摄像头":
            print("错误: 未选择有效的摄像头")
            return

        camera_id = camera_index

        import sys
        from io import StringIO
        original_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"错误: 无法打开摄像头 {camera_id}")
                return

            self.timer.start(30)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.mode_combo.setEnabled(False)
            self.record_button.setEnabled(True)
            
            print(f"开始监控: 模式={mode}, 置信度阈值={self.conf_threshold.value()}")
        finally:
            sys.stderr = original_stderr

    def stop_monitoring(self):
        """停止监控"""
        if self.is_recording:
            self.stop_recording()
        
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.record_button.setEnabled(False)
        
        self.monitor.set_monitor_mode("detection")
    
    def toggle_recording(self):
        """
        @brief 切换录屏状态（开始/停止）
        """
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """
        @brief 开始录屏
        """
        if self.cap is None or not self.cap.isOpened():
            print("错误: 请先开始监控")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.mode_combo.currentData()
        mode_name = "pose" if mode == "pose_analysis" else "detect"
        filename = f"recording_{mode_name}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        fps = 30.0
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
        
        if not self.video_writer.isOpened():
            print(f"错误: 无法创建视频文件 {filepath}")
            self.video_writer = None
            return
        
        self.is_recording = True
        self.recording_start_time = datetime.now()
        
        self.record_button.setText("停止录屏")
        self.record_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.recording_status_label.setText(f"录屏中: {filename}")
        
        print(f"开始录屏: {filepath}")
    
    def stop_recording(self):
        """
        @brief 停止录屏
        """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        
        if self.recording_start_time:
            duration = datetime.now() - self.recording_start_time
            duration_str = str(duration).split('.')[0]
            self.recording_status_label.setText(f"录屏已保存 (时长: {duration_str})")
            self.recording_start_time = None
        
        self.record_button.setText("开始录屏")
        self.record_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        print("录屏已停止")

    def update_frame(self):
        """更新帧显示"""
        ret, frame = self.cap.read()
        if not ret:
            return

        mode = self.mode_combo.currentData()
        
        if mode == "pose_analysis":
            annotated_frame, detection_time, pose_results = self.monitor.process_pose_frame(frame)
            self.update_pose_stats(detection_time, pose_results)
        else:
            annotated_frame, detection_time, defects = self.monitor.process_frame(frame)
            self.update_detection_stats(detection_time, defects)
        
        if self.is_recording and self.video_writer is not None:
            frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
        
        frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_detection_stats(self, detection_time, defects):
        """
        @brief 更新检测模式统计信息
        @param detection_time: 检测时间(ms)
        @param defects: 检测到的缺陷列表
        """
        custom_count = len([d for d in defects if d.get('source') == 'custom'])
        coco_count = len([d for d in defects if d.get('source') == 'coco'])
        
        class_counts = {}
        for d in defects:
            cls_name = d.get('class', 'unknown')
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        class_text = "检测类别: "
        if class_counts:
            items = [f"{k}:{v}" for k, v in sorted(class_counts.items(), key=lambda x: -x[1])[:5]]
            class_text += ", ".join(items)
        else:
            class_text += "无"
        
        self.fps_label.setText(f"FPS: {int(1000 / max(detection_time, 1))}")
        self.detection_time_label.setText(f"检测时间: {detection_time:.2f} ms")
        if self.multi_model_check.isChecked():
            self.defect_count_label.setText(f"自定义: {custom_count}  COCO: {coco_count}")
        else:
            self.defect_count_label.setText(f"缺陷数量: {len(defects)}")
        
        self.class_stats_label.setText(class_text)
        
        if defects:
            self.defect_rate_label.setText("不良率: 100%")
        else:
            self.defect_rate_label.setText("不良率: 0%")
        
        if self.monitor.detect_anomalies(defects):
            self.monitor.trigger_alert(True)
    
    def update_pose_stats(self, detection_time, pose_results):
        """
        @brief 更新姿态分析模式统计信息
        @param detection_time: 检测时间(ms)
        @param pose_results: 姿态分析结果列表
        """
        person_count = len(pose_results)
        
        self.fps_label.setText(f"FPS: {int(1000 / max(detection_time, 1))}")
        self.detection_time_label.setText(f"检测时间: {detection_time:.2f} ms")
        self.person_count_label.setText(f"检测人数: {person_count}")
        
        for joint_name in self.angle_labels:
            self.angle_labels[joint_name].setText(f"{joint_name}: --°")
        
        if pose_results and len(pose_results) > 0:
            first_person = pose_results[0]
            angles = first_person.get("angles", {})
            
            for joint_name, angle in angles.items():
                if joint_name in self.angle_labels:
                    self.angle_labels[joint_name].setText(f"{joint_name}: {angle}°")
