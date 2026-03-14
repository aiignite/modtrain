import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO


class ModelEvaluator:
    def __init__(self):
        self.model = None

    def evaluate(self, model_path, test_data):
        """
        评估模型

        Args:
            model_path: 模型路径
            test_data: 测试数据yaml路径

        Returns:
            dict: 评估结果
        """
        try:
            # 加载模型
            self.model = YOLO(model_path)

            # 评估模型（使用验证集）
            results = self.model.val(data=test_data, split='val')

            # 提取评估指标
            metrics = {
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr)
            }

            return metrics
        except Exception as e:
            print(f"评估模型失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_confusion_matrix(self, predictions, ground_truth):
        """
        生成混淆矩阵

        Args:
            predictions: 预测结果
            ground_truth: 真实标签

        Returns:
            numpy.ndarray: 混淆矩阵
        """
        try:
            cm = confusion_matrix(ground_truth, predictions)
            return cm
        except Exception as e:
            print(f"生成混淆矩阵失败: {e}")
            return None

    def analyze_errors(self, model, test_data):
        """
        分析错误案例

        Args:
            model: 模型
            test_data: 测试数据

        Returns:
            dict: 错误分析结果
        """
        try:
            # 这里可以添加错误分析逻辑
            # 例如：分析误报、漏报的案例
            error_analysis = {
                "false_positives": 0,
                "false_negatives": 0,
                "most_confused_classes": []
            }

            return error_analysis
        except Exception as e:
            print(f"分析错误失败: {e}")
            return {}