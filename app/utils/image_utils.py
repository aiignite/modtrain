import cv2
import numpy as np


def resize_image(image, width=None, height=None):
    """
    调整图像大小

    Args:
        image: 输入图像
        width: 目标宽度
        height: 目标高度

    Returns:
        numpy.ndarray: 调整大小后的图像
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def draw_bbox(image, bbox, label, color=(0, 255, 0), thickness=2):
    """
    绘制边界框

    Args:
        image: 输入图像
        bbox: 边界框坐标 (x1, y1, x2, y2)
        label: 标签文本
        color: 边界框颜色
        thickness: 边界框厚度

    Returns:
        numpy.ndarray: 绘制边界框后的图像
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image


def preprocess_image(image, size=(640, 640)):
    """
    预处理图像

    Args:
        image: 输入图像
        size: 目标大小

    Returns:
        numpy.ndarray: 预处理后的图像
    """
    # 调整大小
    image = cv2.resize(image, size)
    # 归一化
    image = image / 255.0
    # 添加批次维度
    image = np.expand_dims(image, axis=0)
    return image