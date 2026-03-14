from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QGridLayout, QComboBox,
                             QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import os
import cv2
import numpy as np

from app.core.model_evaluator import ModelEvaluator


class ModelEvaluationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.evaluator = ModelEvaluator()
        self.init_ui()
        self.load_available_models()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 模型选择区
        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout()

        # 模型路径
        model_layout.addWidget(QLabel("模型:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItem("选择模型...")
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # 浏览模型按钮
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.browse_model_btn, 0, 2)

        # 测试数据集
        model_layout.addWidget(QLabel("测试数据集:"), 1, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("选择数据集...")
        model_layout.addWidget(self.dataset_combo, 1, 1)
        
        # 浏览数据集按钮
        self.browse_dataset_btn = QPushButton("浏览...")
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        model_layout.addWidget(self.browse_dataset_btn, 1, 2)

        # 开始评估按钮
        self.evaluate_btn = QPushButton("开始评估")
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        model_layout.addWidget(self.evaluate_btn, 2, 0, 1, 3)

        model_group.setLayout(model_layout)

        # 评估结果区
        result_group = QGroupBox("评估结果")
        result_layout = QVBoxLayout()

        # 性能指标表格
        self.metrics_table = QTableWidget(4, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        metrics = [["mAP@0.5", "0.0"], ["mAP@0.5:0.95", "0.0"], ["精确率", "0.0"], ["召回率", "0.0"]]
        for row, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(value))
        result_layout.addWidget(self.metrics_table)

        # 混淆矩阵
        self.confusion_matrix_label = QLabel("混淆矩阵将显示在这里")
        self.confusion_matrix_label.setFixedSize(400, 300)
        self.confusion_matrix_label.setStyleSheet("border: 1px solid gray;")
        self.confusion_matrix_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.confusion_matrix_label)

        result_group.setLayout(result_layout)

        # 错误分析区
        error_group = QGroupBox("错误分析")
        error_layout = QVBoxLayout()
        self.error_analysis = QTextEdit()
        self.error_analysis.setReadOnly(True)
        self.error_analysis.setPlainText("错误分析将显示在这里...")
        error_layout.addWidget(self.error_analysis)
        error_group.setLayout(error_layout)

        # 布局组合
        top_layout = QHBoxLayout()
        top_layout.addWidget(model_group, 1)
        top_layout.addWidget(result_group, 2)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(error_group)
    
    def load_available_models(self):
        """加载可用的模型"""
        latest_model = None
        latest_time = 0
        
        # 扫描训练结果目录
        train_runs_dir = os.path.join(os.getcwd(), "runs", "train")
        if os.path.exists(train_runs_dir):
            exp_dirs = []
            for exp_dir in os.listdir(train_runs_dir):
                exp_path = os.path.join(train_runs_dir, exp_dir)
                if os.path.isdir(exp_path):
                    exp_dirs.append((exp_dir, os.path.getmtime(exp_path)))
            
            # 按时间排序，最新的在前面
            exp_dirs.sort(key=lambda x: x[1], reverse=True)
            
            for exp_dir, _ in exp_dirs:
                exp_path = os.path.join(train_runs_dir, exp_dir)
                weights_dir = os.path.join(exp_path, "weights")
                if os.path.exists(weights_dir):
                    # 优先显示best.pt
                    if os.path.exists(os.path.join(weights_dir, "best.pt")):
                        weight_file = "best.pt"
                        model_path = os.path.join(weights_dir, weight_file)
                        display_name = f"{exp_dir}/{weight_file}"
                        self.model_combo.addItem(display_name, model_path)
                        if latest_model is None:
                            latest_model = model_path
                    # 然后是last.pt
                    if os.path.exists(os.path.join(weights_dir, "last.pt")):
                        weight_file = "last.pt"
                        model_path = os.path.join(weights_dir, weight_file)
                        display_name = f"{exp_dir}/{weight_file}"
                        self.model_combo.addItem(display_name, model_path)
        
        # 扫描yolo目录
        yolo_dir = os.path.join(os.getcwd(), "yolo")
        if os.path.exists(yolo_dir):
            for weight_file in os.listdir(yolo_dir):
                if weight_file.endswith('.pt'):
                    model_path = os.path.join(yolo_dir, weight_file)
                    display_name = f"yolo/{weight_file}"
                    self.model_combo.addItem(display_name, model_path)
        
        # 自动选择最新模型
        if latest_model:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == latest_model:
                    self.model_combo.setCurrentIndex(i)
                    break
        
        # 自动加载数据集
        dataset_yaml = os.path.join(os.getcwd(), "data", "dataset", "data.yaml")
        if os.path.exists(dataset_yaml):
            dataset_dir = os.path.dirname(dataset_yaml)
            self.dataset_combo.addItem(os.path.basename(dataset_dir), dataset_yaml)
            self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
    
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
            # 检查是否已存在
            exists = False
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == file_path:
                    self.model_combo.setCurrentIndex(i)
                    exists = True
                    break
            
            if not exists:
                self.model_combo.addItem(os.path.basename(file_path), file_path)
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
    
    def browse_dataset(self):
        """浏览选择数据集目录"""
        from PyQt5.QtWidgets import QFileDialog
        options = QFileDialog.Options()
        dataset_dir = QFileDialog.getExistingDirectory(
            self, "选择数据集目录", "", options=options
        )
        if dataset_dir:
            # 检查是否已存在
            exists = False
            for i in range(self.dataset_combo.count()):
                if self.dataset_combo.itemData(i) == dataset_dir:
                    self.dataset_combo.setCurrentIndex(i)
                    exists = True
                    break
            
            if not exists:
                self.dataset_combo.addItem(os.path.basename(dataset_dir), dataset_dir)
                self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)

    def evaluate_model(self):
        """评估模型"""
        model_path = self.model_combo.itemData(self.model_combo.currentIndex())
        dataset_path = self.dataset_combo.itemData(self.dataset_combo.currentIndex())
        
        if not model_path or model_path == "选择模型...":
            self.error_analysis.setPlainText("错误: 请先选择模型!")
            return
        
        if not dataset_path or dataset_path == "选择数据集...":
            if os.path.exists("data/dataset/data.yaml"):
                dataset_path = os.path.abspath("data/dataset/data.yaml")
            else:
                self.error_analysis.setPlainText("错误: 请选择数据集或确保data/dataset/data.yaml存在!")
                return
        
        self.error_analysis.setPlainText("正在评估模型，请稍候...\n这可能需要几分钟时间...")
        self.repaint()
        
        if os.path.isdir(dataset_path):
            yaml_path = os.path.join(dataset_path, "data.yaml")
            if os.path.exists(yaml_path):
                dataset_path = yaml_path
        
        metrics = self.evaluator.evaluate(model_path, dataset_path)
        
        if metrics:
            self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics.get('mAP50', 0):.4f}"))
            self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics.get('mAP50_95', 0):.4f}"))
            self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
            self.metrics_table.setItem(3, 1, QTableWidgetItem(f"{metrics.get('recall', 0):.4f}"))
            
            # 显示混淆矩阵（从评估结果目录加载）
            self.show_confusion_matrix()
            
            analysis = "="*40 + "\n"
            analysis += "模型评估结果\n"
            analysis += "="*40 + "\n\n"
            analysis += f"模型: {os.path.basename(model_path)}\n"
            analysis += f"数据集: {os.path.dirname(dataset_path)}\n\n"
            analysis += f"mAP@0.5: {metrics.get('mAP50', 0):.4f}\n"
            analysis += f"mAP@0.5:0.95: {metrics.get('mAP50_95', 0):.4f}\n"
            analysis += f"精确率: {metrics.get('precision', 0):.4f}\n"
            analysis += f"召回率: {metrics.get('recall', 0):.4f}\n\n"
            
            map50 = metrics.get('mAP50', 0)
            analysis += "评估建议:\n"
            if map50 > 0.9:
                analysis += "✓ 模型表现优秀！\n"
            elif map50 > 0.7:
                analysis += "○ 模型表现良好，可以继续优化\n"
            elif map50 > 0.5:
                analysis += "△ 需要更多训练数据或调整参数\n"
            else:
                analysis += "✗ 模型效果较差，检查数据和标注\n"
            
            self.error_analysis.setPlainText(analysis)
        else:
            self.error_analysis.setPlainText("评估失败，请检查模型和数据集路径")
    
    def show_confusion_matrix(self):
        """显示混淆矩阵图片"""
        cm_path = os.path.join(os.getcwd(), "runs", "detect", "val", "confusion_matrix_normalized.png")
        if not os.path.exists(cm_path):
            cm_path = os.path.join(os.getcwd(), "runs", "detect", "val", "confusion_matrix.png")
        if not os.path.exists(cm_path):
            runs_dir = os.path.join(os.getcwd(), "runs", "detect")
            if os.path.exists(runs_dir):
                for root, dirs, files in os.walk(runs_dir):
                    for f in files:
                        if "confusion_matrix" in f and f.endswith('.png'):
                            cm_path = os.path.join(root, f)
                            break
        
        if os.path.exists(cm_path):
            pixmap = QPixmap(cm_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(self.confusion_matrix_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.confusion_matrix_label.setPixmap(pixmap)
            else:
                self.confusion_matrix_label.setText("无法加载混淆矩阵图片")
        else:
            self.confusion_matrix_label.setText("未找到混淆矩阵图片\n评估后会生成")