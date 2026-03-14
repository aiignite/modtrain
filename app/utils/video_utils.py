import cv2
import os


def extract_frames(video_path, output_dir, interval=1):
    """
    从视频提取帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        interval: 帧提取间隔

    Returns:
        int: 提取的帧数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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


def get_video_info(video_path):
    """
    获取视频信息

    Args:
        video_path: 视频路径

    Returns:
        dict: 视频信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    cap.release()
    return info