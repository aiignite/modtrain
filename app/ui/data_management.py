from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QGroupBox, QGridLayout,
                             QSpinBox, QComboBox, QCheckBox)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import os
import cv2
import shutil
import xml.etree.ElementTree as ET
import numpy as np

class DataManagementWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.data_type = None
        self.processed_images = []
        self.current_image_index = 0
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 数据导入区
        import_group = QGroupBox("数据导入")
        import_layout = QGridLayout()

        # 图片导入
        self.img_import_btn = QPushButton("导入图片")
        self.img_import_btn.clicked.connect(self.import_images)
        import_layout.addWidget(self.img_import_btn, 0, 0)

        # 视频导入
        self.video_import_btn = QPushButton("导入视频")
        self.video_import_btn.clicked.connect(self.import_videos)
        import_layout.addWidget(self.video_import_btn, 0, 1)

        # 导入路径显示
        import_layout.addWidget(QLabel("导入路径:"), 1, 0)
        self.import_path_label = QLabel("未选择")
        import_layout.addWidget(self.import_path_label, 1, 1)

        import_group.setLayout(import_layout)

        # 数据预览区
        preview_group = QGroupBox("数据预览")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("预览区域")
        self.preview_label.setFixedSize(400, 300)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_label)
        
        # 翻页控制
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一张")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.page_label = QLabel("0/0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.next_btn = QPushButton("下一张 ▶")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label)
        nav_layout.addWidget(self.next_btn)
        preview_layout.addLayout(nav_layout)
        
        preview_group.setLayout(preview_layout)

        # 数据处理区
        process_group = QGroupBox("数据处理")
        process_layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("帧提取间隔:"))
        self.frame_interval = QSpinBox()
        self.frame_interval.setValue(1)
        self.frame_interval.setFixedWidth(80)
        top_layout.addWidget(self.frame_interval)
        top_layout.addStretch()
        process_layout.addLayout(top_layout)

        aug_group = QGroupBox("数据增强选项 (可多选)")
        aug_layout = QVBoxLayout()

        aug_layout.addWidget(QLabel("✓ 自动转换标注！先标注原图再增强"))

        flip_layout = QHBoxLayout()
        self.flip_h = QCheckBox("水平翻转 (推荐)")
        self.flip_h.setChecked(True)
        flip_layout.addWidget(self.flip_h)
        self.flip_v = QCheckBox("垂直翻转 (推荐)")
        self.flip_v.setChecked(True)
        flip_layout.addWidget(self.flip_v)
        flip_layout.addStretch()
        aug_layout.addLayout(flip_layout)

        rot_layout = QHBoxLayout()
        self.rotate_check = QCheckBox("旋转 ⚠")
        self.rotate_check.setChecked(False)
        self.rotate_check.setToolTip("注意：旋转后需重新在LabelImg中标注！")
        rot_layout.addWidget(self.rotate_check)
        rot_layout.addWidget(QLabel("角度:"))
        self.rotate_angle = QSpinBox()
        self.rotate_angle.setRange(5, 45)
        self.rotate_angle.setValue(10)
        self.rotate_angle.setFixedWidth(60)
        self.rotate_angle.setSuffix("°")
        rot_layout.addWidget(self.rotate_angle)
        rot_layout.addStretch()
        aug_layout.addLayout(rot_layout)

        bright_layout = QHBoxLayout()
        self.bright_check = QCheckBox("亮度变化 (推荐)")
        self.bright_check.setChecked(True)
        bright_layout.addWidget(self.bright_check)
        bright_layout.addWidget(QLabel("变化值:"))
        self.bright_value = QSpinBox()
        self.bright_value.setRange(10, 100)
        self.bright_value.setValue(30)
        self.bright_value.setFixedWidth(60)
        bright_layout.addWidget(self.bright_value)
        bright_layout.addStretch()
        aug_layout.addLayout(bright_layout)

        aug_group.setLayout(aug_layout)
        process_layout.addWidget(aug_group)

        btn_layout = QHBoxLayout()
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_data)
        self.process_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        btn_layout.addWidget(self.process_btn)

        self.labeling_tool_btn = QPushButton("打开LabelImg")
        self.labeling_tool_btn.clicked.connect(self.open_labeling_tool)
        btn_layout.addWidget(self.labeling_tool_btn)
        process_layout.addLayout(btn_layout)

        self.view_annotations_btn = QPushButton("查看标注结果")
        self.view_annotations_btn.clicked.connect(self.view_annotations)
        process_layout.addWidget(self.view_annotations_btn)

        self.prepare_training_btn = QPushButton("准备训练数据")
        self.prepare_training_btn.clicked.connect(self.prepare_training_data)
        self.prepare_training_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        process_layout.addWidget(self.prepare_training_btn)

        process_group.setLayout(process_layout)

        # 数据集信息区
        info_group = QGroupBox("数据集信息")
        info_layout = QVBoxLayout()
        
        self.data_info_label = QLabel("样本数量: 0\n类别数量: 0")
        info_layout.addWidget(self.data_info_label)
        
        dir_btn_layout = QHBoxLayout()
        
        self.open_imported_btn = QPushButton("待标注 (imported)")
        self.open_imported_btn.clicked.connect(self.open_imported_directory)
        dir_btn_layout.addWidget(self.open_imported_btn)
        
        self.open_raw_btn = QPushButton("原图 (raw)")
        self.open_raw_btn.setEnabled(False)
        self.open_raw_btn.clicked.connect(self.open_raw_directory)
        dir_btn_layout.addWidget(self.open_raw_btn)
        
        self.open_processed_btn = QPushButton("预览 (processed)")
        self.open_processed_btn.setEnabled(False)
        self.open_processed_btn.clicked.connect(self.open_processed_directory)
        dir_btn_layout.addWidget(self.open_processed_btn)
        
        self.open_unlabeled_btn = QPushButton("未标注 (unlabeled)")
        self.open_unlabeled_btn.setEnabled(False)
        self.open_unlabeled_btn.clicked.connect(self.open_unlabeled_directory)
        dir_btn_layout.addWidget(self.open_unlabeled_btn)
        
        info_layout.addLayout(dir_btn_layout)
        info_group.setLayout(info_layout)

        # 布局组合
        top_layout = QHBoxLayout()
        top_layout.addWidget(import_group, 1)
        top_layout.addWidget(preview_group, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(process_group, 1)
        bottom_layout.addWidget(info_group, 1)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

    def import_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)",
                                                options=options)
        if files:
            imported_dir = os.path.join(os.getcwd(), "data", "imported")
            os.makedirs(imported_dir, exist_ok=True)
            
            self.selected_files = []
            for f in files:
                dst = os.path.join(imported_dir, os.path.basename(f))
                if os.path.abspath(f) != os.path.abspath(dst):
                    shutil.copy2(f, dst)
                self.selected_files.append(dst)
            
            self.data_type = "images"
            self.import_path_label.setText(f"已选择 {len(files)} 张图片 -> data/imported/")
            if self.selected_files:
                pixmap = QPixmap(self.selected_files[0])
                pixmap = pixmap.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
                self.preview_label.setPixmap(pixmap)

    def import_videos(self):
        # 打开文件对话框选择视频
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)", options=options)
        if file:
            self.selected_files = [file]
            self.data_type = "video"
            self.import_path_label.setText(file)
            # 预览视频第一帧
            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            if ret:
                # 转换为QPixmap
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                from PyQt5.QtGui import QImage
                qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                pixmap = pixmap.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
                self.preview_label.setPixmap(pixmap)
            cap.release()

    def process_data(self):
        """处理数据：整理标注结果，分离已标注/未标注"""
        import glob
        
        if not self.selected_files:
            self.data_info_label.setText("错误: 请先导入数据!")
            return
        
        imported_dir = os.path.join(os.getcwd(), "data", "imported")
        raw_dir = os.path.join(os.getcwd(), "data", "raw")
        annotations_dir = os.path.join(os.getcwd(), "data", "annotations")
        processed_dir = os.path.join(os.getcwd(), "data", "processed")
        unlabeled_dir = os.path.join(os.getcwd(), "data", "unlabeled")
        
        for d in [raw_dir, annotations_dir, processed_dir, unlabeled_dir]:
            os.makedirs(d, exist_ok=True)
        
        labeled_count = 0
        unlabeled_count = 0
        
        aug_options = {
            'flip_h': self.flip_h.isChecked(),
            'flip_v': self.flip_v.isChecked(),
            'rotate': self.rotate_check.isChecked(),
            'rotate_angle': self.rotate_angle.value(),
            'bright': self.bright_check.isChecked(),
            'bright_value': self.bright_value.value()
        }
        any_aug = aug_options['flip_h'] or aug_options['flip_v'] or aug_options['rotate'] or aug_options['bright']
        
        if self.data_type == "images":
            for file_path in self.selected_files:
                file_name = os.path.basename(file_path)
                base_name = os.path.splitext(file_name)[0]
                ext = os.path.splitext(file_name)[1]
                
                aug_suffixes = ['_hflip', '_vflip', '_rot', '_bright+', '_dark-']
                is_aug_file = any(s in base_name for s in aug_suffixes)
                if is_aug_file:
                    print(f"跳过增强文件: {file_name}")
                    continue
                
                local_xml = os.path.join(os.path.dirname(file_path), base_name + ".xml")
                annotations_xml = os.path.join(annotations_dir, base_name + ".xml")
                xml_path = local_xml if os.path.exists(local_xml) else (annotations_xml if os.path.exists(annotations_xml) else None)
                
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                if xml_path:
                    raw_dst = os.path.join(raw_dir, file_name)
                    xml_dst = os.path.join(annotations_dir, base_name + ".xml")
                    
                    cv2.imwrite(raw_dst, img)
                    if xml_path != xml_dst:
                        shutil.copy2(xml_path, xml_dst)
                    labeled_count += 1
                    print(f"[已标注] {file_name} -> raw/")
                    
                    self.save_annotated_preview(img, xml_path, processed_dir, file_name)
                    
                    if any_aug:
                        aug_images = self.generate_augmentations(img, aug_options)
                        for aug_name, aug_img in aug_images.items():
                            aug_file = f"{base_name}_{aug_name}{ext}"
                            aug_dst = os.path.join(raw_dir, aug_file)
                            cv2.imwrite(aug_dst, aug_img)
                            self.transform_and_save_xml(xml_dst, aug_name, img.shape[1], img.shape[0], aug_options)
                            labeled_count += 1
                            print(f"  增强: {aug_file}")
                else:
                    unlabeled_dst = os.path.join(unlabeled_dir, file_name)
                    cv2.imwrite(unlabeled_dst, img)
                    unlabeled_count += 1
                    print(f"[未标注] {file_name} -> unlabeled/")
        
        aug_count = len([k for k in ['flip_h','flip_v','rotate','bright'] if aug_options[k]]) if any_aug else 0
        aug_info = f"\n增强: {aug_count}种" if any_aug else ""
        self.data_info_label.setText(f"已标注: {labeled_count} 张\n未标注: {unlabeled_count} 张{aug_info}")
        self.raw_dir = raw_dir
        self.open_raw_btn.setEnabled(True)
        self.open_processed_btn.setEnabled(True)
        self.open_unlabeled_btn.setEnabled(unlabeled_count > 0)
        self.view_annotations_btn.setEnabled(True)
        
        print(f"\n处理完成: 已标注={labeled_count}, 未标注={unlabeled_count}")
    
    def save_annotated_preview(self, img, xml_path, output_dir, file_name):
        """保存带标注框的预览图到processed目录"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        preview_img = img.copy()
        box_index = 1
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[(box_index - 1) % len(colors)]
            cv2.rectangle(preview_img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(preview_img, f"{box_index}:{class_name}", (xmin, ymin-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            box_index += 1
        
        out_path = os.path.join(output_dir, file_name)
        cv2.imwrite(out_path, preview_img)
        print(f"    预览: {file_name} -> processed/")
    
    def generate_augmentations(self, img, options):
        """根据选项生成数据增强版本"""
        import numpy as np
        aug_images = {}
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        if options.get('flip_h'):
            aug_images['hflip'] = cv2.flip(img, 1)
        
        if options.get('flip_v'):
            aug_images['vflip'] = cv2.flip(img, 0)
        
        angle = options.get('rotate_angle', 15)
        if options.get('rotate'):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_images[f'rot{angle}'] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            aug_images[f'rot-{angle}'] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        bright_val = options.get('bright_value', 30)
        if options.get('bright'):
            img_float = img.astype(np.float32)
            bright = np.clip(img_float + bright_val, 0, 255).astype(np.uint8)
            aug_images[f'bright+{bright_val}'] = bright
            
            dark = np.clip(img_float - bright_val, 0, 255).astype(np.uint8)
            aug_images[f'dark-{bright_val}'] = dark
        
        return aug_images
    
    def transform_and_save_xml(self, orig_xml, aug_name, img_w, img_h, options):
        """根据增强类型自动转换标注框坐标并保存"""
        import math
        
        tree = ET.parse(orig_xml)
        root = tree.getroot()
        
        base_name = os.path.splitext(os.path.basename(orig_xml))[0]
        new_xml_name = f"{base_name}_{aug_name}.xml"
        new_xml_path = os.path.join(os.path.dirname(orig_xml), new_xml_name)
        
        root.find('filename').text = f"{base_name}_{aug_name}.jpg"
        
        for obj in root.iter('object'):
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            bw, bh = xmax - xmin, ymax - ymin
            
            if aug_name == 'hflip':
                nxmin = img_w - xmax
                nxmax = img_w - xmin
                bndbox.find('xmin').text = str(int(nxmin))
                bndbox.find('xmax').text = str(int(nxmax))
            
            elif aug_name == 'vflip':
                nymin = img_h - ymax
                nymax = img_h - ymin
                bndbox.find('ymin').text = str(int(nymin))
                bndbox.find('ymax').text = str(int(nymax))
            
            elif 'rot' in aug_name:
                angle = options.get('rotate_angle', 15)
                if '-' in aug_name:
                    angle = -angle
                
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                cx2, cy2 = img_w/2, img_h/2
                
                corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                new_corners = []
                for (x, y) in corners:
                    dx, dy = x - cx2, y - cy2
                    nx = cx2 + dx * cos_a - dy * sin_a
                    ny = cy2 + dx * sin_a + dy * cos_a
                    new_corners.append((nx, ny))
                
                xs = [p[0] for p in new_corners]
                ys = [p[1] for p in new_corners]
                bndbox.find('xmin').text = str(int(max(0, min(xs))))
                bndbox.find('ymin').text = str(int(max(0, min(ys))))
                bndbox.find('xmax').text = str(int(min(img_w, max(xs))))
                bndbox.find('ymax').text = str(int(min(img_h, max(ys))))
        
        tree.write(new_xml_path)
        print(f"    标注: {new_xml_name}")
    
    def open_imported_directory(self):
        """打开待标注目录 (imported)"""
        imported_dir = os.path.join(os.getcwd(), "data", "imported")
        if os.path.exists(imported_dir):
            os.startfile(imported_dir)
        else:
            print("待标注目录不存在，请先导入图片")
    
    def open_raw_directory(self):
        """打开原始数据目录 (raw)"""
        if hasattr(self, 'raw_dir') and os.path.exists(self.raw_dir):
            os.startfile(self.raw_dir)
        else:
            raw_dir = os.path.join(os.getcwd(), "data", "raw")
            if os.path.exists(raw_dir):
                os.startfile(raw_dir)
            else:
                print("原始数据目录不存在")
    
    def open_processed_directory(self):
        """打开标注结果目录 (processed)"""
        processed_dir = os.path.join(os.getcwd(), "data", "processed")
        if os.path.exists(processed_dir):
            os.startfile(processed_dir)
        else:
            print("标注目录还没有标注结果")
    
    def open_unlabeled_directory(self):
        """打开未标注目录 (unlabeled)"""
        unlabeled_dir = os.path.join(os.getcwd(), "data", "unlabeled")
        if os.path.exists(unlabeled_dir):
            os.startfile(unlabeled_dir)
        else:
            print("未标注目录不存在")
    
    def open_labeling_tool(self):
        """打开标注工具，自动打开imported目录和annotations目录"""
        import subprocess
        
        imported_dir = os.path.join(os.getcwd(), "data", "imported")
        annotations_dir = os.path.join(os.getcwd(), "data", "annotations")
        os.makedirs(imported_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # 尝试用Python启动labelImg（通过labelImg包或直接命令）
        try:
            cmd = ["labelImg", imported_dir, annotations_dir]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            print(f"启动LabelImg: {cmd}")
            return
        except Exception as e:
            print(f"labelImg命令失败: {e}")
        
        # 尝试通过python -m启动
        try:
            cmd = ["python", "-m", "labelImg", imported_dir, annotations_dir]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            print(f"启动LabelImg: {cmd}")
            return
        except Exception as e:
            print(f"python -m labelImg失败: {e}")
        
        # 打开下载页面
        print("请先安装labelImg: pip install labelImg")
        try:
            import webbrowser
            webbrowser.open("https://pypi.org/project/labelImg/")
        except:
            pass
    
    def view_annotations(self):
        """查看标注结果：从raw目录读取图片，动态绘制标注框"""
        self.load_raw_images_with_annotations()
        
        if self.processed_images:
            self.display_current_image()
            print(f"加载了 {len(self.processed_images)} 张图片")
        else:
            print("未找到图片，请先导入数据并标注")
    
    def load_raw_images_with_annotations(self):
        """加载raw目录下有对应标注的图片"""
        raw_dir = os.path.join(os.getcwd(), "data", "raw")
        annotations_dir = os.path.join(os.getcwd(), "data", "annotations")
        self.processed_images = []
        
        if os.path.exists(raw_dir):
            for file_name in sorted(os.listdir(raw_dir)):
                if file_name.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(file_name)[0]
                    xml_path = os.path.join(annotations_dir, base_name + ".xml")
                    if os.path.exists(xml_path):
                        self.processed_images.append(os.path.join(raw_dir, file_name))
        
        self.current_image_index = 0
        self.update_nav_buttons()
    
    def show_prev_image(self):
        """显示上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
    
    def show_next_image(self):
        """显示下一张图片"""
        if self.current_image_index < len(self.processed_images) - 1:
            self.current_image_index += 1
            self.display_current_image()
    
    def display_current_image(self):
        """显示当前索引的图片（带标注框）"""
        import xml.etree.ElementTree as ET
        from PyQt5.QtGui import QImage
        
        processed_dir = os.path.join(os.getcwd(), "data", "processed")
        annotations_dir = os.path.join(os.getcwd(), "data", "annotations")
        os.makedirs(processed_dir, exist_ok=True)
        
        if 0 <= self.current_image_index < len(self.processed_images):
            img_path = self.processed_images[self.current_image_index]
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            
            # 读取原图
            img = cv2.imread(img_path)
            if img is not None:
                # 查找对应XML
                xml_path = os.path.join(annotations_dir, base_name + ".xml")
                
                annotation_info = []
                if os.path.exists(xml_path):
                    # 解析XML并绘制标注框
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    box_index = 1
                    for obj in root.iter('object'):
                        class_name = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        
                        # 记录标注信息
                        annotation_info.append(f"[{box_index}] {class_name}: ({xmin},{ymin})-({xmax},{ymax})")
                        
                        # 绘制标注框（不同颜色区分）
                        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                        color = colors[(box_index - 1) % len(colors)]
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                        # 绘制类别标签（带序号）
                        label = f"{box_index}:{class_name}"
                        cv2.putText(img, label, (xmin, ymin - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        box_index += 1
                    
                    # 保存带标注的图片到processed目录
                    processed_img_path = os.path.join(processed_dir, file_name)
                    cv2.imwrite(processed_img_path, img)
                
                # 更新信息显示区域
                info_text = f"当前文件: {file_name}\n"
                info_text += f"标注数量: {len(annotation_info)} 个\n"
                if annotation_info:
                    info_text += "\n标注详情:\n" + "\n".join(annotation_info)
                else:
                    info_text += "\n未找到标注信息"
                self.data_info_label.setText(info_text)
                
                # 转换为QPixmap显示
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img_rgb.shape
                bytes_per_line = channel * width
                qimg = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(self.preview_label.width(),
                                          self.preview_label.height(),
                                          Qt.KeepAspectRatio)
                    self.preview_label.setPixmap(pixmap)
        
        self.update_nav_buttons()
    
    def update_nav_buttons(self):
        """更新翻页按钮状态"""
        total = len(self.processed_images)
        current = self.current_image_index + 1 if total > 0 else 0
        self.page_label.setText(f"{current}/{total}")
        self.prev_btn.setEnabled(current > 1)
        self.next_btn.setEnabled(current < total and total > 0)
    
    def prepare_training_data(self):
        """准备训练数据：XML转YOLO格式 + 数据集分割"""
        from app.core.data_manager import DataManager
        
        raw_dir = os.path.join(os.getcwd(), "data", "raw")
        annotations_dir = os.path.join(os.getcwd(), "data", "annotations")
        output_dir = os.path.join(os.getcwd(), "data", "dataset")
        
        # 检查是否有数据
        if not os.path.exists(annotations_dir) or len([f for f in os.listdir(annotations_dir) if f.endswith('.xml')]) == 0:
            self.data_info_label.setText("错误: annotations目录为空!\n请先用LabelImg标注")
            return
        
        data_manager = DataManager()
        
        result = data_manager.prepare_training_data(
            raw_dir=raw_dir,
            annotations_dir=annotations_dir,
            output_dataset_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.2
        )
        
        if result["success"]:
            info = f"训练数据准备完成!\n"
            info += f"训练集: {result['train_count']} 张\n"
            info += f"验证集: {result['val_count']} 张\n"
            info += f"测试集: {result['test_count']} 张\n"
            info += f"类别: {', '.join(result['classes'])}\n"
            info += f"配置文件: {result['yaml_path']}"
            self.data_info_label.setText(info)
            
            print(f"训练数据准备完成!")
            print(f"  训练集: {result['train_count']}")
            print(f"  验证集: {result['val_count']}")
            print(f"  测试集: {result['test_count']}")
            print(f"  类别: {result['classes']}")
            print(f"  data.yaml: {result['yaml_path']}")
        else:
            self.data_info_label.setText(f"准备失败: {result['message']}")