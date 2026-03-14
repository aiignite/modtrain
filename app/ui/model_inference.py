from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QGridLayout, QComboBox,
                             QTextEdit, QFileDialog, QScrollArea, QFrame, QSplitter,
                             QDoubleSpinBox, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO


class ModelInferenceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_coco = None
        self.test_images = []
        self.current_image_index = 0
        self.results = {}
        self.conf_threshold = 0.25
        self.coco_model_path = 'yolov8n.pt'
        self.load_config()
        self.init_ui()
        self.load_available_models()
    
    def load_config(self):
        """加载用户配置文件"""
        config_file = os.path.join("config", "user_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_config = config.get("model", {})
                    self.coco_model_path = model_config.get("default_path", "yolov8n.pt")
            except Exception as e:
                print(f"加载配置文件失败: {e}")

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()

        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("自定义模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("选择模型...")
        self.model_combo.setMinimumWidth(150)
        row1.addWidget(self.model_combo, 1)
        self.browse_model_btn = QPushButton("浏览...")
        self.browse_model_btn.setFixedWidth(70)
        self.browse_model_btn.clicked.connect(self.browse_model)
        row1.addWidget(self.browse_model_btn)
        model_layout.addLayout(row1)

        self.multi_model_check = QCheckBox("启用多模型检测 (同时加载COCO模型检测80类物体)")
        self.multi_model_check.setChecked(False)
        self.multi_model_check.setToolTip(f"同时使用自定义模型和{self.coco_model_path}进行检测")
        model_layout.addWidget(self.multi_model_check)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        model_layout.addWidget(self.load_model_btn)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        data_group = QGroupBox("测试数据与参数")
        data_layout = QVBoxLayout()

        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        self.conf_thresh = QDoubleSpinBox()
        self.conf_thresh.setRange(0.01, 0.99)
        self.conf_thresh.setValue(0.25)
        self.conf_thresh.setSingleStep(0.05)
        self.conf_thresh.setDecimals(2)
        self.conf_thresh.setFixedWidth(100)
        param_layout.addWidget(self.conf_thresh, 0, 1)
        param_layout.addWidget(QLabel("(越低检测越多)"), 0, 2)
        data_layout.addLayout(param_layout)

        self.slice_check = QCheckBox("大图切片检测 (适合检测小目标)")
        self.slice_check.setChecked(False)
        data_layout.addWidget(self.slice_check)

        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("  切片大小:"))
        self.slice_size = QComboBox()
        self.slice_size.addItems(["640", "800", "1024", "1280"])
        self.slice_size.setCurrentIndex(0)
        self.slice_size.setFixedWidth(100)
        slice_layout.addWidget(self.slice_size)
        
        slice_layout.addWidget(QLabel("  NMS阈值:"))
        self.nms_thresh = QDoubleSpinBox()
        self.nms_thresh.setRange(0.1, 0.9)
        self.nms_thresh.setValue(0.3)
        self.nms_thresh.setSingleStep(0.1)
        self.nms_thresh.setFixedWidth(80)
        self.nms_thresh.setToolTip("越小去重越严格，误判越少")
        slice_layout.addWidget(self.nms_thresh)
        slice_layout.addStretch()
        data_layout.addLayout(slice_layout)

        self.select_images_btn = QPushButton("选择测试图片")
        self.select_images_btn.clicked.connect(self.select_test_images)
        data_layout.addWidget(self.select_images_btn)

        self.select_dir_btn = QPushButton("选择测试目录")
        self.select_dir_btn.clicked.connect(self.select_test_directory)
        data_layout.addWidget(self.select_dir_btn)

        self.start_infer_btn = QPushButton("开始推理")
        self.start_infer_btn.clicked.connect(self.run_inference)
        self.start_infer_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        data_layout.addWidget(self.start_infer_btn)

        self.open_result_dir_btn = QPushButton("查看结果目录")
        self.open_result_dir_btn.clicked.connect(self.open_result_directory)
        self.open_result_dir_btn.setEnabled(False)
        data_layout.addWidget(self.open_result_dir_btn)

        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        log_group = QGroupBox("推理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlainText("准备就绪...")
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)

        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setMinimumWidth(380)

        right_layout = QVBoxLayout()

        result_group = QGroupBox("推理结果预览")
        result_layout = QVBoxLayout()

        self.nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< 上一张")
        self.prev_btn.clicked.connect(self.show_prev)
        self.prev_btn.setEnabled(False)
        self.nav_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("0/0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.nav_layout.addWidget(self.page_label)

        self.next_btn = QPushButton("下一张 >")
        self.next_btn.clicked.connect(self.show_next)
        self.next_btn.setEnabled(False)
        self.nav_layout.addWidget(self.next_btn)

        result_layout.addLayout(self.nav_layout)

        self.result_label = QLabel("推理结果将显示在这里")
        self.result_label.setMinimumSize(600, 400)
        self.result_label.setStyleSheet("border: 1px solid gray;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)

        self.result_info = QLabel("")
        self.result_info.setWordWrap(True)
        result_layout.addWidget(self.result_info)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([400, 800])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

    def load_available_models(self):
        train_runs_dir = os.path.join(os.getcwd(), "runs", "train")
        if os.path.exists(train_runs_dir):
            exp_dirs = []
            for exp_dir in os.listdir(train_runs_dir):
                exp_path = os.path.join(train_runs_dir, exp_dir)
                if os.path.isdir(exp_path):
                    exp_dirs.append((exp_dir, os.path.getmtime(exp_path)))

            exp_dirs.sort(key=lambda x: x[1], reverse=True)

            for exp_dir, _ in exp_dirs:
                exp_path = os.path.join(train_runs_dir, exp_dir)
                weights_dir = os.path.join(exp_path, "weights")
                if os.path.exists(weights_dir):
                    if os.path.exists(os.path.join(weights_dir, "best.pt")):
                        model_path = os.path.join(weights_dir, "best.pt")
                        self.model_combo.addItem(f"{exp_dir}/best.pt", model_path)
                    if os.path.exists(os.path.join(weights_dir, "last.pt")):
                        model_path = os.path.join(weights_dir, "last.pt")
                        self.model_combo.addItem(f"{exp_dir}/last.pt", model_path)

        yolo_dir = os.path.join(os.getcwd(), "yolo")
        if os.path.exists(yolo_dir):
            for f in os.listdir(yolo_dir):
                if f.endswith('.pt'):
                    model_path = os.path.join(yolo_dir, f)
                    self.model_combo.addItem(f"yolo/{f}", model_path)

        if self.model_combo.count() > 1:
            self.model_combo.setCurrentIndex(1)

    def browse_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "",
            "Model Files (*.pt *.onnx);;All Files (*)",
            options=options
        )
        if file_path:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == file_path:
                    self.model_combo.setCurrentIndex(i)
                    return
            self.model_combo.addItem(os.path.basename(file_path), file_path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def load_model(self):
        model_path = self.model_combo.itemData(self.model_combo.currentIndex())
        if not model_path or model_path == "选择模型...":
            self.log_text.append("错误: 请先选择模型!")
            return

        try:
            self.log_text.append(f"加载自定义模型: {model_path}")
            self.repaint()
            self.model = YOLO(model_path)
            self.log_text.append(f"✓ 自定义模型加载成功! 可检测 {len(self.model.names)} 类: {list(self.model.names.values())}")
        except Exception as e:
            self.log_text.append(f"✗ 自定义模型加载失败: {e}")
            self.model = None
            return

        if self.multi_model_check.isChecked():
            try:
                self.log_text.append(f"\n加载COCO模型 ({self.coco_model_path})...")
                self.repaint()
                self.model_coco = YOLO(self.coco_model_path)
                self.log_text.append(f"✓ COCO模型加载成功! 可检测80类通用物体")
            except Exception as e:
                self.log_text.append(f"⚠ COCO模型加载失败: {e}")
                self.log_text.append("  将仅使用自定义模型检测")
                self.model_coco = None
        else:
            self.model_coco = None

    def select_test_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择测试图片", "",
            "Images (*.bmp *.jpg *.jpeg *.png *.png);;All Files (*)",
            options=options
        )
        if files:
            self.test_images = list(files)
            self.current_image_index = 0
            self.log_text.append(f"选择了 {len(self.test_images)} 张图片")

    def select_test_directory(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self, "选择测试目录", "", options=options
        )
        if directory:
            self.test_images = []
            for f in os.listdir(directory):
                if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    self.test_images.append(os.path.join(directory, f))

            self.current_image_index = 0
            self.log_text.append(f"从目录加载了 {len(self.test_images)} 张图片")

    def run_inference(self):
        if not self.model:
            self.log_text.append("错误: 请先加载模型!")
            return

        if not self.test_images:
            self.log_text.append("错误: 请先选择测试图片!")
            return

        conf_thresh = self.conf_thresh.value()
        self.log_text.append(f"\n置信度阈值: {conf_thresh}")

        result_dir = os.path.join(os.getcwd(), "data", "inference_results")
        os.makedirs(result_dir, exist_ok=True)

        self.results = {}
        success_count = 0

        self.log_text.append(f"\n开始推理 {len(self.test_images)} 张图片...")
        self.repaint()

        for i, img_path in enumerate(self.test_images):
            try:
                import time
                start_time = time.time()
                
                img = cv2.imread(img_path)
                if img is None:
                    self.log_text.append(f"✗ 无法读取: {os.path.basename(img_path)}")
                    continue

                boxes = []
                colors_custom = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                colors_coco = [(128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0), (255, 0, 128)]

                use_slice = self.slice_check.isChecked()
                slice_sz = int(self.slice_size.currentText())
                nms_t = self.nms_thresh.value()
                
                inference_count = 0
                if use_slice:
                    h, w = img.shape[:2]
                    if max(w, h) > slice_sz * 1.5:
                        stride = slice_sz - 150
                        cols = (w + stride - 1) // stride
                        rows = (h + stride - 1) // stride
                        inference_count = cols * rows
                    else:
                        inference_count = 1
                    all_boxes = self.detect_with_slice(img, conf_thresh, slice_sz, nms_t)
                else:
                    inference_count = 1
                    all_boxes = self.detect_single_image(img, conf_thresh)

                inference_time = (time.time() - start_time) * 1000

                for box in all_boxes:
                    boxes.append(box)
                    is_custom = box.get('source') == 'custom'
                    colors = colors_custom if is_custom else colors_coco
                    color = colors[box.get('cls', 0) % len(colors)]
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    prefix = "" if is_custom else "COCO:"
                    label = f"{prefix}{box['class']} {box['conf']:.2f}"
                    y_pos = y1 - 10 if is_custom else y2 + 15
                    cv2.putText(img, label, (x1, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4 if is_custom else 0.35, color, 1)

                custom_count = len([b for b in boxes if b.get('source') == 'custom'])
                coco_count = len([b for b in boxes if b.get('source') == 'coco'])

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                result_img_path = os.path.join(result_dir, f"{base_name}_result.jpg")
                cv2.imwrite(result_img_path, img)

                self.results[img_path] = {
                    'result_path': result_img_path,
                    'boxes': boxes,
                    'custom_count': custom_count,
                    'coco_count': coco_count,
                    'count': len(boxes),
                    'inference_time': inference_time,
                    'inference_count': inference_count
                }

                success_count += 1
                if self.model_coco:
                    self.log_text.append(f"[{i+1}/{len(self.test_images)}] {os.path.basename(img_path)} - 自定义:{custom_count} COCO:{coco_count} [{inference_count}次, {inference_time:.1f}ms]")
                else:
                    self.log_text.append(f"[{i+1}/{len(self.test_images)}] {os.path.basename(img_path)} - 检测到 {len(boxes)} 个目标 [{inference_count}次, {inference_time:.1f}ms]")

            except Exception as e:
                import traceback
                self.log_text.append(f"✗ 处理失败 {os.path.basename(img_path)}: {e}")
                self.log_text.append(traceback.format_exc())

        self.log_text.append(f"\n推理完成! 成功: {success_count}/{len(self.test_images)}")
        self.log_text.append(f"结果保存在: {result_dir}")

        self.open_result_dir_btn.setEnabled(True)

        if self.results:
            self.current_image_index = 0
            self.update_nav_buttons()
            self.display_result_image()

    def detect_single_image(self, img, conf_thresh):
        """单图检测"""
        boxes = []
        if self.model:
            results = self.model(img, conf=conf_thresh, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = self.model.names[cls]
                    boxes.append({
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'conf': float(conf), 'class': cls_name, 'cls': cls, 'source': 'custom'
                    })
        if self.model_coco:
            results = self.model_coco(img, conf=conf_thresh, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    cls_name = self.model_coco.names[cls]
                    boxes.append({
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'conf': float(conf), 'class': cls_name, 'cls': cls, 'source': 'coco'
                    })
        return boxes

    def detect_with_slice(self, img, conf_thresh, slice_size=640, nms_thresh=0.3, overlap=150):
        """切片检测：将大图切成重叠区域分别检测"""
        h, w = img.shape[:2]
        all_boxes = []
        
        if max(w, h) <= slice_size * 1.5:
            return self.detect_single_image(img, conf_thresh)
        
        stride = slice_size - overlap
        regions = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + slice_size, w)
                y2 = min(y + slice_size, h)
                x1 = max(0, x2 - slice_size)
                y1 = max(0, y2 - slice_size)
                regions.append((x1, y1, x2, y2))
        
        margin = 20
        for (x1, y1, x2, y2) in regions:
            patch = img[y1:y2, x1:x2].copy()
            patch_h, patch_w = patch.shape[:2]
            patch_boxes = self.detect_single_image(patch, conf_thresh)
            
            for b in patch_boxes:
                bx1, by1, bx2, by2 = b['x1'], b['y1'], b['x2'], b['y2']
                if x1 > 0 and bx1 < margin:
                    continue
                if y1 > 0 and by1 < margin:
                    continue
                if x2 < w and bx2 > patch_w - margin:
                    continue
                if y2 < h and by2 > patch_h - margin:
                    continue
                
                b['x1'] += x1
                b['y1'] += y1
                b['x2'] += x1
                b['y2'] += y1
                all_boxes.append(b)
        
        return self.nms_boxes(all_boxes, nms_thresh)

    def nms_boxes(self, boxes, iou_thresh=0.5):
        """非极大值抑制：去除重叠框"""
        if not boxes:
            return []
        
        custom_boxes = [b for b in boxes if b.get('source') == 'custom']
        coco_boxes = [b for b in boxes if b.get('source') == 'coco']
        
        def nms_single(box_list):
            if not box_list:
                return []
            box_list = sorted(box_list, key=lambda x: x['conf'], reverse=True)
            keep = []
            while box_list:
                best = box_list.pop(0)
                keep.append(best)
                box_list = [b for b in box_list if self.iou(best, b) < iou_thresh]
            return keep
        
        return nms_single(custom_boxes) + nms_single(coco_boxes)

    def iou(self, a, b):
        """计算IOU"""
        x1 = max(a['x1'], b['x1'])
        y1 = max(a['y1'], b['y1'])
        x2 = min(a['x2'], b['x2'])
        y2 = min(a['y2'], b['y2'])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
        area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
        return inter / (area_a + area_b - inter + 1e-6)

    def display_result_image(self):
        if not self.results:
            return

        img_path = self.test_images[self.current_image_index]
        if img_path not in self.results:
            return

        result = self.results[img_path]
        pixmap = QPixmap(result['result_path'])

        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.result_label.setPixmap(pixmap)

            info_text = f"文件: {os.path.basename(img_path)}\n"
            info_text += f"推理时间: {result.get('inference_time', 0):.1f} ms\n"
            info_text += f"推理次数: {result.get('inference_count', 1)} 次\n"
            info_text += f"检测到 {result['count']} 个目标:\n"
            for i, box in enumerate(result['boxes']):
                info_text += f"  [{i+1}] {box['class']} (置信度: {box['conf']:.2f})\n"
            self.result_info.setText(info_text)

            self.page_label.setText(f"{self.current_image_index + 1}/{len(self.results)}")

    def show_prev(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_result_image()
            self.update_nav_buttons()

    def show_next(self):
        if self.current_image_index < len(self.test_images) - 1:
            self.current_image_index += 1
            self.display_result_image()
            self.update_nav_buttons()

    def update_nav_buttons(self):
        total = len(self.results)
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < total - 1)

    def open_result_directory(self):
        result_dir = os.path.join(os.getcwd(), "data", "inference_results")
        if os.path.exists(result_dir):
            os.startfile(result_dir)
