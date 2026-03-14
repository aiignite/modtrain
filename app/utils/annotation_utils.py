import os
import json


def read_yolo_annotation(annotation_path, image_width, image_height):
    """
    读取YOLO格式标注

    Args:
        annotation_path: 标注文件路径
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        list: 标注列表
    """
    annotations = []

    if not os.path.exists(annotation_path):
        return annotations

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height

                # 转换为 (x1, y1, x2, y2) 格式
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                annotations.append({
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2]
                })

    return annotations


def write_yolo_annotation(annotation_path, annotations, image_width, image_height):
    """
    写入YOLO格式标注

    Args:
        annotation_path: 标注文件路径
        annotations: 标注列表
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        bool: 写入是否成功
    """
    try:
        with open(annotation_path, 'w') as f:
            for ann in annotations:
                class_id = ann["class_id"]
                x1, y1, x2, y2 = ann["bbox"]

                # 转换为 YOLO 格式 (x_center, y_center, width, height) 归一化
                x_center = ((x1 + x2) / 2) / image_width
                y_center = ((y1 + y2) / 2) / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True
    except Exception as e:
        print(f"写入标注失败: {e}")
        return False