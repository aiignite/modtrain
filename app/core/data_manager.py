import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import yaml
from sklearn.model_selection import train_test_split
import json


class DataManager:
    def __init__(self):
        self.data_dir = ""
        self.annotations_dir = ""

    def import_data(self, data_path, data_type):
        """
        导入数据

        Args:
            data_path: 数据路径
            data_type: 数据类型 (images, video)

        Returns:
            bool: 导入是否成功
        """
        try:
            if data_type == "images":
                # 复制图片到数据目录
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)

                if os.path.isdir(data_path):
                    for file in os.listdir(data_path):
                        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            src = os.path.join(data_path, file)
                            dst = os.path.join(self.data_dir, file)
                            shutil.copy2(src, dst)
                else:
                    # 单个文件
                    if data_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        dst = os.path.join(self.data_dir, os.path.basename(data_path))
                        shutil.copy2(data_path, dst)

            elif data_type == "video":
                # 从视频提取帧
                if not os.path.exists(self.data_dir):
                    os.makedirs(self.data_dir)

                self.extract_frames(data_path, self.data_dir)

            return True
        except Exception as e:
            print(f"导入数据失败: {e}")
            return False

    def extract_frames(self, video_path, output_dir, interval=1):
        """
        从视频提取帧

        Args:
            video_path: 视频路径
            output_dir: 输出目录
            interval: 帧提取间隔

        Returns:
            int: 提取的帧数
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                output_path = os.path.join(output_dir, f"frame_{extracted_count}.jpg")
                cv2.imwrite(output_path, frame)
                extracted_count += 1

            frame_count += 1

        cap.release()
        return extracted_count

    def split_dataset(self, data_dir, train_ratio=0.8, val_ratio=0.1):
        """
        分割数据集

        Args:
            data_dir: 数据目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例

        Returns:
            dict: 数据集分割结果
        """
        # 获取所有图片文件
        images = []
        for file in os.listdir(data_dir):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(file)

        # 分割数据集
        train_images, test_images = train_test_split(images, test_size=1 - train_ratio, random_state=42)
        val_images, test_images = train_test_split(test_images, test_size=val_ratio / (1 - train_ratio),
                                                   random_state=42)

        return {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

    def apply_augmentation(self, image, augmentations):
        """
        应用数据增强

        Args:
            image: 输入图像
            augmentations: 增强选项

        Returns:
            numpy.ndarray: 增强后的图像
        """
        # 随机翻转
        if "flip" in augmentations and np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # 水平翻转

        # 随机旋转
        if "rotate" in augmentations:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))

        # 随机缩放
        if "scale" in augmentations:
            scale = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            # 裁剪回原始大小
            if scale > 1:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = image[start_y:start_y + h, start_x:start_x + w]
            else:
                # 填充
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = cv2.copyMakeBorder(image, pad_y, h - new_h - pad_y,
                                           pad_x, w - new_w - pad_x,
                                           cv2.BORDER_CONSTANT, value=0)

        return image

    def convert_xml_to_yolo(self, xml_dir, output_dir, class_mapping):
        """
        将XML标注文件转换为YOLO格式的txt文件
        
        Args:
            xml_dir: XML标注文件目录
            output_dir: YOLO格式输出目录
            class_mapping: 类别名称到ID的映射字典
            
        Returns:
            tuple: (转换数量, 类别集合)
        """
        converted_count = 0
        classes_found = set()
        
        os.makedirs(output_dir, exist_ok=True)
        
        for xml_file in os.listdir(xml_dir):
            if not xml_file.endswith('.xml'):
                continue
                
            xml_path = os.path.join(xml_dir, xml_file)
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 获取图像尺寸
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                
                # 收集标注
                yolo_lines = []
                for obj in root.iter('object'):
                    class_name = obj.find('name').text
                    # 统一转换为小写，避免大小写不同被识别为不同类别
                    class_name_lower = class_name.lower()
                    classes_found.add(class_name_lower)
                    
                    # 获取类别ID
                    if class_name_lower not in class_mapping:
                        class_mapping[class_name_lower] = len(class_mapping)
                    class_id = class_mapping[class_name_lower]
                    
                    # 获取边界框
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # 转换为YOLO归一化格式
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height
                    
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                
                # 写入YOLO格式文件
                if yolo_lines:
                    txt_filename = os.path.splitext(xml_file)[0] + '.txt'
                    txt_path = os.path.join(output_dir, txt_filename)
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(yolo_lines))
                    converted_count += 1
                    
            except Exception as e:
                print(f"转换文件 {xml_file} 失败: {e}")
        
        return converted_count, classes_found

    def prepare_training_data(self, raw_dir, annotations_dir, output_dataset_dir, train_ratio=0.7, val_ratio=0.2):
        """
        准备训练数据：XML转YOLO + 数据集分割
        
        Args:
            raw_dir: 原始图片目录
            annotations_dir: XML标注文件目录
            output_dataset_dir: 输出数据集目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            dict: 准备结果信息
        """
        # 类别映射
        class_mapping = {}
        
        # 创建YOLO目录结构
        images_train_dir = os.path.join(output_dataset_dir, 'images', 'train')
        images_val_dir = os.path.join(output_dataset_dir, 'images', 'val')
        images_test_dir = os.path.join(output_dataset_dir, 'images', 'test')
        labels_train_dir = os.path.join(output_dataset_dir, 'labels', 'train')
        labels_val_dir = os.path.join(output_dataset_dir, 'labels', 'val')
        labels_test_dir = os.path.join(output_dataset_dir, 'labels', 'test')
        temp_labels_dir = os.path.join(output_dataset_dir, 'temp_labels')
        
        for d in [images_train_dir, images_val_dir, images_test_dir,
                  labels_train_dir, labels_val_dir, labels_test_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 第一步：将XML转换为临时YOLO格式
        self.convert_xml_to_yolo(annotations_dir, temp_labels_dir, class_mapping)
        
        # 第二步：收集有标注的图片
        image_files = []
        for f in os.listdir(raw_dir):
            if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(f)[0]
                txt_file = base_name + '.txt'
                if os.path.exists(os.path.join(temp_labels_dir, txt_file)):
                    image_files.append(f)
        
        if len(image_files) == 0:
            return {"success": False, "message": "没有找到带标注的图片"}
        
        # 第三步：分割数据集
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        n = len(image_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # 第四步：复制文件到对应目录
        def copy_files(files, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
            count = 0
            for f in files:
                base_name = os.path.splitext(f)[0]
                txt_file = base_name + '.txt'
                
                # 转换图片格式（BMP转JPG）
                src_img_path = os.path.join(src_img_dir, f)
                dst_img_name = base_name + '.jpg'
                dst_img_path = os.path.join(dst_img_dir, dst_img_name)
                
                img = cv2.imread(src_img_path)
                if img is not None:
                    cv2.imwrite(dst_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    # 复制标注
                    src_lbl_path = os.path.join(src_lbl_dir, txt_file)
                    dst_lbl_path = os.path.join(dst_lbl_dir, base_name + '.txt')
                    if os.path.exists(src_lbl_path):
                        shutil.copy2(src_lbl_path, dst_lbl_path)
                    count += 1
            return count
        
        train_count = copy_files(train_files, raw_dir, temp_labels_dir, images_train_dir, labels_train_dir)
        val_count = copy_files(val_files, raw_dir, temp_labels_dir, images_val_dir, labels_val_dir)
        test_count = copy_files(test_files, raw_dir, temp_labels_dir, images_test_dir, labels_test_dir)
        
        # 清理临时目录
        shutil.rmtree(temp_labels_dir, ignore_errors=True)
        
        # 第五步：生成data.yaml
        class_names = sorted(class_mapping.keys(), key=lambda k: class_mapping[k])
        data_config = {
            "path": os.path.abspath(output_dataset_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names
        }
        
        yaml_path = os.path.join(output_dataset_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        return {
            "success": True,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "classes": class_names,
            "yaml_path": yaml_path
        }