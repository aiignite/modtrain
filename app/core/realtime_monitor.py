"""
@file realtime_monitor.py
@brief 实时监控核心模块，支持目标检测、姿态估计、语义分割等功能
@author AI Assistant
@date 2026-03-09
"""
import cv2
import time
import numpy as np
import json
import os
import math
from PIL import ImageFont, ImageDraw, Image

from ultralytics import YOLO


class RealTimeMonitor:
    """
    @class RealTimeMonitor
    @brief 实时监控类，处理摄像头视频流并进行模型推理
    
    支持功能：
    - 目标检测
    - 姿态估计（人体关键点检测）
    - 语义分割
    - 多模型同时检测
    """
    
    COCO_KEYPOINTS = {
        0: "鼻子",
        1: "左眼",
        2: "右眼",
        3: "左耳",
        4: "右耳",
        5: "左肩",
        6: "右肩",
        7: "左肘",
        8: "右肘",
        9: "左腕",
        10: "右腕",
        11: "左髋",
        12: "右髋",
        13: "左膝",
        14: "右膝",
        15: "左踝",
        16: "右踝"
    }
    
    COCO_SKELETON = [
        [0, 1], [1, 3], [0, 2], [2, 4],
        [0, 5], [5, 7], [7, 9],
        [0, 6], [6, 8], [8, 10],
        [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16],
        [5, 6], [11, 12]
    ]
    
    SKELETON_COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0),
        (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255),
        (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85)
    ]

    def __init__(self):
        """
        @brief 初始化实时监控器
        """
        self.model = None
        self.model_coco = None
        self.use_multi_model = False
        self.conf_threshold = 0.5
        self.net = None
        self.classes = None
        self.is_darknet_model = False
        self.class_mapping = {}
        self.coco_model_path = 'yolov8n.pt'
        self.is_pose_model = False
        self.is_coco_pose_model = False
        self.is_seg_model = False
        self.is_coco_seg_model = False
        
        self.monitor_mode = "detection"
        self.pose_model = None
        self.pose_conf_threshold = 0.5
        
        self.load_config()
        self.load_class_mapping()
        
        self.available_camera = self.detect_available_camera()
        if self.available_camera is None:
            print("警告: 没有找到可用的摄像头!")
    
    def load_config(self):
        """加载用户配置文件"""
        config_file = os.path.join("config", "user_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_config = config.get("model", {})
                    self.coco_model_path = model_config.get("default_path", "yolov8n.pt")
                    print(f"加载配置: COCO模型路径 = {self.coco_model_path}")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
            
    # def load_model(self, model_path):
    #     """加载模型"""
    #     try:
    #         self.model = YOLO(model_path)
    #         return True
    #     except Exception as e:
    #         print(f"加载模型失败: {e}")
    #         return False
    def load_model(self, model_path):
        """加载模型"""
        try:
            # 检查是否为Darknet格式模型
            if model_path.endswith('.weights'):
                # 这是一个Darknet格式的模型
                self.is_darknet_model = True
                self.is_pose_model = False
                # 查找对应的配置文件
                config_path = model_path.replace('.weights', '.cfg')
                if not os.path.exists(config_path):
                    print(f"错误: 找不到配置文件 {config_path}")
                    return False
                
                # 加载类别文件
                names_path = os.path.join(os.path.dirname(model_path), 'coco.names')
                if os.path.exists(names_path):
                    with open(names_path, 'r') as f:
                        self.classes = [line.strip() for line in f.readlines()]
                else:
                    # 如果没有类别文件，使用默认类别
                    self.classes = [f'class_{i}' for i in range(80)]
                
                # 使用OpenCV DNN加载模型
                self.net = cv2.dnn.readNet(model_path, config_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"成功加载Darknet模型: {model_path}")
                return True
            else:
                # 使用Ultralytics加载其他格式的模型
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.is_darknet_model = False
                # 检测是否为姿态模型
                self.is_pose_model = self._check_pose_model(self.model)
                # 检测是否为分割模型
                self.is_seg_model = self._check_seg_model(self.model)
                if self.is_pose_model:
                    print(f"成功加载姿态模型: {model_path}")
                elif self.is_seg_model:
                    print(f"成功加载分割模型: {model_path}")
                else:
                    print(f"成功加载Ultralytics模型: {model_path}")
                return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False
    
    def _check_pose_model(self, model):
        """检测是否为姿态模型"""
        try:
            # 使用小图像测试
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_frame, verbose=False)
            return hasattr(results[0], 'keypoints') and results[0].keypoints is not None
        except Exception:
            return False
    
    def _check_seg_model(self, model):
        """检测是否为分割模型"""
        try:
            # 使用小图像测试
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(dummy_frame, verbose=False)
            return hasattr(results[0], 'masks') and results[0].masks is not None
        except Exception:
            return False
    
    def load_coco_model(self):
        """加载COCO模型"""
        try:
            from ultralytics import YOLO
            self.model_coco = YOLO(self.coco_model_path)
            self.use_multi_model = True
            # 检测是否为姿态模型
            self.is_coco_pose_model = self._check_pose_model(self.model_coco)
            # 检测是否为分割模型
            self.is_coco_seg_model = self._check_seg_model(self.model_coco)
            if self.is_coco_pose_model:
                print(f"成功加载COCO姿态模型: {self.coco_model_path}")
            elif self.is_coco_seg_model:
                print(f"成功加载COCO分割模型: {self.coco_model_path}")
            else:
                print(f"成功加载COCO模型: {self.coco_model_path}")
            return True
        except Exception as e:
            print(f"加载COCO模型失败: {e}")
            self.model_coco = None
            return False
                
    def detect_available_camera(self):
        """检测可用的摄像头"""
        for i in range(5):  # 尝试检测前5个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print(f"找到可用摄像头: {i}")
                return i
        return None
    
    def load_class_mapping(self):
        """加载类别映射配置文件"""
        mapping_file = os.path.join("config", "class_mapping.json")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并所有类别的映射
                    all_mappings = {}
                    for category, mappings in config.get("class_mapping", {}).items():
                        all_mappings.update(mappings)
                    self.class_mapping = all_mappings
                    print(f"成功加载类别映射，共 {len(all_mappings)} 个类别")
            except Exception as e:
                print(f"加载类别映射失败: {e}")
        else:
            print(f"类别映射文件不存在: {mapping_file}")
    
    def draw_chinese_text(self, img, text, position, font_size=12, color=(0, 255, 0)):
        """在图片上绘制中文"""
        # 加载中文字体
        font_path = os.path.join("config", "fonts", "simhei.ttf")
        if os.path.exists(font_path):
            try:
                # 使用PIL绘制中文
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(font_path, font_size)
                draw.text(position, text, font=font, fill=color)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"绘制中文失败: {e}")
                # 失败时使用默认字体
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # 如果没有中文字体，使用默认字体
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img
    
    def draw_pose(self, img, keypoints, color=(0, 255, 0), thickness=2):
        """绘制姿态关键点和连接线"""
        # COCO格式的关键点连接顺序
        skeleton = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # 右臂
            [0, 5], [5, 6], [6, 7],          # 左臂
            [0, 8], [8, 9], [9, 10],         # 右腿
            [8, 11], [11, 12], [12, 13],     # 左腿
            [0, 14], [14, 16],               # 右眼
            [0, 15], [15, 17]                # 左眼
        ]
        
        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # 置信度阈值
                cv2.circle(img, (int(x), int(y)), 3, color, -1)
        
        # 绘制连接线
        for start, end in skeleton:
            if start < len(keypoints) and end < len(keypoints):
                if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                    x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                    x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        return img
    
    def draw_segmentation(self, img, masks, color=(0, 255, 255), alpha=0.4):
        """绘制分割掩码"""
        if masks is None:
            return img
        
        # 获取掩码数据
        if hasattr(masks, 'data'):
            mask_data = masks.data.cpu().numpy()
        else:
            mask_data = masks
        
        # 为每个掩码创建彩色覆盖层
        overlay = img.copy()
        
        for i, mask in enumerate(mask_data):
            # 调整掩码大小到图像尺寸
            if mask.ndim == 2:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
            
            # 为不同的对象使用不同的颜色
            color_idx = i % 5
            colors = [
                (0, 255, 255),  # 青色
                (255, 0, 255),  # 洋红色
                (255, 255, 0),  # 黄色
                (0, 255, 0),    # 绿色
                (255, 0, 0)     # 蓝色
            ]
            current_color = colors[color_idx]
            
            # 应用掩码到覆盖层
            overlay[mask == 1] = current_color
        
        # 混合原始图像和覆盖层
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        return result
        

    def start_monitoring(self, camera_id, model_path):
        """开始监控"""
        # 加载模型
        if not self.load_model(model_path):
            return False

        # 如果没有指定摄像头或指定的摄像头不可用，使用检测到的可用摄像头
        if camera_id < 0 or not self.check_camera_availability(camera_id):
            if self.available_camera is not None:
                print(f"使用检测到的可用摄像头: {self.available_camera}")
                camera_id = self.available_camera
            else:
                # 再次尝试检测可用摄像头
                detected_camera = self.detect_available_camera()
                if detected_camera is None:
                    print("错误: 没有找到可用的摄像头!")
                    return False
                camera_id = detected_camera
                self.available_camera = detected_camera

        # 打开摄像头 
        cap = cv2.VideoCapture(camera_id) 
        if not cap.isOpened(): 
            print(f"无法打开摄像头 {camera_id}!")
            return False 

        return cap

    # def process_frame(self, frame):
    #     """处理单帧"""
    #     start_time = time.time()

    #     # 模型推理
    #     if self.model:
    #         results = self.model(frame, conf=self.conf_threshold)
    #         # 绘制检测结果
    #         annotated_frame = results[0].plot()

    #         # 计算检测时间
    #         detection_time = (time.time() - start_time) * 1000

    #         # 提取缺陷信息
    #         defects = []
    #         for result in results:
    #             for box in result.boxes:
    #                 defects.append({
    #                     "class": result.names[int(box.cls)],
    #                     "confidence": float(box.conf),
    #                     "bbox": box.xyxy[0].tolist()
    #                 })

    #         return annotated_frame, detection_time, defects
    #     else:
    #         return frame, 0, []
    def process_frame(self, frame):
        """处理单帧"""
        start_time = time.time()
        defects = []
        
        # 水平翻转图像，解决左右镜像问题
        frame = cv2.flip(frame, 1)
        
        if self.is_darknet_model and self.net:
            # 使用OpenCV DNN处理Darknet模型
            height, width = frame.shape[:2]
            
            # 创建blob
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # 获取输出层
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # 前向传播
            outputs = self.net.forward(output_layers)
            
            # 处理输出
            class_ids = []
            confidences = []
            boxes = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.conf_threshold:
                        # 计算边界框
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 左上角坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 非极大值抑制
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.45)
            
            # 绘制检测结果
            for i in indices:
                i = i[0] if isinstance(i, list) else i
                box = boxes[i]
                x, y, w, h = box
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # 绘制边界框
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # 获取类别名称并映射为中文
                class_name = self.classes[class_id]
                chinese_class = self.class_mapping.get(class_name, class_name)
                
                # 绘制标签
                label = f"{chinese_class}: {confidence:.2f}"
                frame = self.draw_chinese_text(frame, label, (x, y - 10), font_size=12, color=color)
                
                # 添加到缺陷列表
                defects.append({
                    "class": chinese_class,
                    "confidence": confidence,
                    "bbox": [x, y, w, h]
                })
        elif self.model:
            colors_custom = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            colors_coco = [(128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0), (255, 0, 128)]
            
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    chinese_class = self.class_mapping.get(class_name, class_name)
                    confidence = float(box.conf)
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    color = colors_custom[int(box.cls) % len(colors_custom)]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{chinese_class}: {confidence:.2f}"
                    frame = self.draw_chinese_text(frame, label, (x, y - 10), font_size=12, color=color)
                    
                    defects.append({
                        "class": chinese_class,
                        "confidence": confidence,
                        "bbox": [x, y, w, h],
                        "source": "custom"
                    })
                
                # 处理姿态关键点（如果是姿态模型）
                if self.is_pose_model and hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    for kp in keypoints:
                        frame = self.draw_pose(frame, kp, color=(255, 0, 0), thickness=2)
                
                # 处理分割掩码（如果是分割模型）
                if self.is_seg_model and hasattr(result, 'masks') and result.masks is not None:
                    frame = self.draw_segmentation(frame, result.masks, color=(0, 255, 255), alpha=0.4)
            
            if self.model_coco and self.use_multi_model:
                results_coco = self.model_coco(frame, conf=self.conf_threshold, verbose=False)
                for result in results_coco:
                    for box in result.boxes:
                        class_name = result.names[int(box.cls)]
                        chinese_class = self.class_mapping.get(class_name, class_name)
                        confidence = float(box.conf)
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        color = colors_coco[int(box.cls) % len(colors_coco)]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        
                        label = f"COCO:{chinese_class}: {confidence:.2f}"
                        frame = self.draw_chinese_text(frame, label, (x, y + h + 5), font_size=10, color=color)
                        
                        defects.append({
                            "class": chinese_class,
                            "confidence": confidence,
                            "bbox": [x, y, w, h],
                            "source": "coco"
                        })
                
                # 处理COCO模型的姿态关键点（如果是姿态模型）
                if self.is_coco_pose_model and hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()
                    for kp in keypoints:
                        frame = self.draw_pose(frame, kp, color=(128, 0, 255), thickness=2)
                
                # 处理COCO模型的分割掩码（如果是分割模型）
                if self.is_coco_seg_model and hasattr(result, 'masks') and result.masks is not None:
                    frame = self.draw_segmentation(frame, result.masks, color=(255, 0, 128), alpha=0.4)
        
        # 计算检测时间
        detection_time = (time.time() - start_time) * 1000
        
        return frame, detection_time, defects

    def detect_anomalies(self, defects):
        """检测异常"""
        return len(defects) > 0

    def trigger_alert(self, anomaly):
        """触发报警 - 静默处理，不打印"""
        pass
    
    def load_pose_model(self, model_path=None):
        """
        @brief 加载姿态估计模型
        @param model_path: 模型路径，默认使用 yolov8n-pose.pt
        @return: 是否加载成功
        """
        if model_path is None:
            model_path = os.path.join("yolo", "yolov8n-pose.pt")
        
        if not os.path.exists(model_path):
            print(f"姿态模型不存在: {model_path}")
            default_pose_models = [
                os.path.join("yolo", "yolov8n-pose.pt"),
                os.path.join("yolo", "yolo26m-pose.pt"),
                os.path.join("yolo", "yolov8m-pose.pt")
            ]
            for path in default_pose_models:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                print("未找到可用的姿态模型")
                return False
        
        try:
            self.pose_model = YOLO(model_path)
            print(f"成功加载姿态模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载姿态模型失败: {e}")
            return False
    
    def set_monitor_mode(self, mode):
        """
        @brief 设置监控模式
        @param mode: 模式名称 ("detection" 或 "pose_analysis")
        """
        self.monitor_mode = mode
        if mode == "pose_analysis" and self.pose_model is None:
            self.load_pose_model()
    
    def calculate_angle(self, p1, p2, p3):
        """
        @brief 计算三个点形成的角度
        @param p1: 第一个点坐标 (x, y)
        @param p2: 顶点坐标 (x, y)
        @param p3: 第三个点坐标 (x, y)
        @return: 角度（度数）
        """
        try:
            radians = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - \
                      math.atan2(p1[1] - p2[1], p1[0] - p2[0])
            angle = abs(radians * 180.0 / math.pi)
            if angle > 180.0:
                angle = 360.0 - angle
            return angle
        except Exception:
            return 0.0
    
    def calculate_joint_angles(self, keypoints):
        """
        @brief 计算主要关节角度
        @param keypoints: 关键点数组，形状为 (17, 3)，每行为 (x, y, confidence)
        @return: 关节角度字典
        """
        angles = {}
        
        joint_pairs = {
            "左肘": (5, 7, 9),
            "右肘": (6, 8, 10),
            "左肩": (11, 5, 7),
            "右肩": (12, 6, 8),
            "左髋": (5, 11, 13),
            "右髋": (6, 12, 14),
            "左膝": (11, 13, 15),
            "右膝": (12, 14, 16)
        }
        
        for joint_name, (i1, i2, i3) in joint_pairs.items():
            if i1 < len(keypoints) and i2 < len(keypoints) and i3 < len(keypoints):
                p1 = keypoints[i1]
                p2 = keypoints[i2]
                p3 = keypoints[i3]
                
                if p1[2] > 0.5 and p2[2] > 0.5 and p3[2] > 0.5:
                    angle = self.calculate_angle(
                        (p1[0], p1[1]),
                        (p2[0], p2[1]),
                        (p3[0], p3[1])
                    )
                    angles[joint_name] = round(angle, 1)
        
        return angles
    
    def draw_pose_skeleton(self, img, keypoints, color=(0, 255, 0), thickness=2, draw_labels=True):
        """
        @brief 绘制姿态骨架
        @param img: 输入图像
        @param keypoints: 关键点数组
        @param color: 绘制颜色
        @param thickness: 线条粗细
        @param draw_labels: 是否绘制关键点标签
        @return: 绘制后的图像
        """
        if keypoints is None or len(keypoints) == 0:
            return img
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
                
                if draw_labels and i in self.COCO_KEYPOINTS:
                    label = self.COCO_KEYPOINTS[i]
                    img = self.draw_chinese_text(img, label, (int(x) + 8, int(y) - 8), 
                                                  font_size=10, color=(255, 255, 255))
        
        for idx, (start, end) in enumerate(self.COCO_SKELETON):
            if start < len(keypoints) and end < len(keypoints):
                if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                    x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                    x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                    line_color = self.SKELETON_COLORS[idx % len(self.SKELETON_COLORS)]
                    cv2.line(img, (x1, y1), (x2, y2), line_color, thickness)
        
        return img
    
    def process_pose_frame(self, frame):
        """
        @brief 处理姿态分析帧
        @param frame: 输入帧
        @return: (处理后的帧, 检测时间, 姿态结果列表)
        """
        start_time = time.time()
        pose_results = []
        
        frame = cv2.flip(frame, 1)
        
        if self.pose_model is None:
            if not self.load_pose_model():
                return frame, 0, []
        
        try:
            results = self.pose_model(frame, conf=self.pose_conf_threshold, verbose=False)
            
            person_count = 0
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    
                    for person_idx, keypoints in enumerate(keypoints_data):
                        person_count += 1
                        
                        valid_kps = sum(1 for kp in keypoints if kp[2] > 0.5)
                        
                        angles = self.calculate_joint_angles(keypoints)
                        
                        frame = self.draw_pose_skeleton(frame, keypoints, draw_labels=False)
                        
                        if result.boxes is not None and len(result.boxes) > person_idx:
                            box = result.boxes[person_idx]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            label = f"Person {person_count}"
                            frame = self.draw_chinese_text(frame, label, (x, y - 25), 
                                                          font_size=14, color=(0, 255, 0))
                        
                        pose_results.append({
                            "person_id": person_count,
                            "keypoints": keypoints.tolist(),
                            "valid_keypoints": valid_kps,
                            "angles": angles,
                            "bbox": [x, y, w, h] if result.boxes is not None else None
                        })
            
            info_text = f"检测人数: {person_count}"
            frame = self.draw_chinese_text(frame, info_text, (10, 30), font_size=16, color=(0, 255, 0))
            
            if pose_results and len(pose_results) > 0:
                angles = pose_results[0]["angles"]
                y_offset = 60
                for joint_name, angle in angles.items():
                    angle_text = f"{joint_name}: {angle}°"
                    frame = self.draw_chinese_text(frame, angle_text, (10, y_offset), 
                                                  font_size=12, color=(255, 255, 0))
                    y_offset += 25
        
        except Exception as e:
            print(f"姿态分析错误: {e}")
        
        detection_time = (time.time() - start_time) * 1000
        
        return frame, detection_time, pose_results