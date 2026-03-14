import os
import time
from ultralytics import YOLO
import yaml


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.model_path = ""

    def prepare_model(self, model_type, model_size):
        """
        准备模型

        Args:
            model_type: 模型类型 (yolov3,yolov5, yolov8)
            model_size: 模型大小 (n, s, m, l, x)

        Returns:
            bool: 准备是否成功
        """
        try:
            if model_type == "yolov8":
                self.model = YOLO(f"yolov8{model_size}.pt")
            elif model_type == "yolov5":
                self.model = YOLO(f"yolov5{model_size}.pt")
            elif model_type == "yolov3":
                # YOLOv5 也可以通过 ultralytics 加载
                self.model = YOLO(f"yolov3{model_size}.pt")
            return True
        except Exception as e:
            print(f"准备模型失败: {e}")
            return False

    def create_data_config(self, data_dir, classes):
        """
        创建数据配置文件

        Args:
            data_dir: 数据目录
            classes: 类别列表

        Returns:
            str: 配置文件路径
        """
        config = {
            "path": data_dir,
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(classes),
            "names": classes
        }

        config_path = os.path.join(data_dir, "data.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def train(self, data_config, model_config, hyperparams):
        """
        训练模型

        Args:
            data_config: 数据配置
            model_config: 模型配置
            hyperparams: 超参数

        Returns:
            str: 训练结果路径
        """
        try:
            # 开始训练
            results = self.model.train(
                data=data_config,
                epochs=hyperparams.get("epochs", 100),
                batch=hyperparams.get("batch_size", 16),
                lr0=hyperparams.get("learning_rate", 0.001),
                imgsz=hyperparams.get("img_size", 640),
                project=os.path.join("runs", "train"),
                name=f"exp_{int(time.time())}"
            )

            # 返回最佳模型路径
            return results.save_dir
        except Exception as e:
            print(f"训练模型失败: {e}")
            return ""

    def export_model(self, model_path, export_format):
        """
        导出模型

        Args:
            model_path: 模型路径
            export_format: 导出格式 (onnx, torchscript, etc.)

        Returns:
            str: 导出模型路径
        """
        try:
            model = YOLO(model_path)
            export_results = model.export(format=export_format)
            return export_results
        except Exception as e:
            print(f"导出模型失败: {e}")
            return ""