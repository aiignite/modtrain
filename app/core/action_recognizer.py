"""
@file action_recognizer.py
@brief 基于关节角度和物体识别的动作识别引擎
@author AI Assistant
@date 2026-03-10
@note 嵌入式优化版本 - 纯代码逻辑判断，无需复杂深度学习
"""

import math
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from collections import deque


class ActionType(Enum):
    """动作类型枚举"""
    IDLE = "空闲"
    PICKING = "取料"
    PLACING = "放置"
    INSPECTING = "检查"
    CONFIRMING = "确认"
    REACHING = "伸手"
    RETRACTING = "收回"
    OPERATING = "操作"
    UNKNOWN = "未知"


class ActionRecognizer:
    """
    @class ActionRecognizer
    @brief 基于规则的动作识别器

    通过关节角度、物体位置、动作状态等特征，
    使用预定义的规则逻辑判断当前动作类型。
    支持动作平滑（滑动窗口投票）和自定义规则配置。
    """

    def __init__(self, config: Dict = None):
        """
        @brief 初始化动作识别器
        @param config: 识别配置参数
        """
        self.config = config or self._get_default_config()
        self.action_history = deque(maxlen=100)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        # 滑动窗口用于动作平滑
        self.smooth_window = deque(maxlen=self.config.get("smooth_window_size", 5))
        # 自定义动作规则
        self.custom_rules = []

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "arm_extension_high": 140,       # 手臂伸展角度阈值（度）
            "arm_extension_low": 60,         # 手臂弯曲角度阈值
            "hand_front_threshold": 0.3,     # 手部在前方阈值（相对比例）
            "hand_down_threshold": -0.1,     # 手部向下阈值
            "body_forward_threshold": 150,   # 身体前倾阈值（髋关节角度）
            "stability_frames": 5,           # 稳定帧数
            "smooth_window_size": 5,         # 平滑窗口大小
            "confidence_threshold": 0.6,     # 置信度阈值
            "shoulder_raise_threshold": 60,  # 抬肩阈值
        }

    def recognize_action(self, keypoints: np.ndarray,
                         detected_objects: List[Dict] = None) -> Tuple[ActionType, float]:
        """
        @brief 识别当前动作
        @param keypoints: 17个关键点数据 (17, 3) - x, y, confidence
        @param detected_objects: 检测到的物体列表
        @return: (动作类型, 置信度)
        """
        if keypoints is None or len(keypoints) < 17:
            return ActionType.IDLE, 0.0

        features = self._extract_features(keypoints, detected_objects)
        raw_action, confidence = self._apply_rules(features)

        # 滑动窗口平滑
        self.smooth_window.append(raw_action)
        smoothed_action = self._smooth_action()

        self.action_history.append({
            "action": smoothed_action,
            "raw_action": raw_action,
            "confidence": confidence,
            "features": features
        })

        return smoothed_action, confidence

    def _smooth_action(self) -> ActionType:
        """通过投票平滑动作识别结果"""
        if not self.smooth_window:
            return ActionType.IDLE
        counts = {}
        for action in self.smooth_window:
            counts[action] = counts.get(action, 0) + 1
        return max(counts, key=counts.get)

    def _extract_features(self, keypoints: np.ndarray,
                          objects: List[Dict] = None) -> Dict:
        """
        @brief 提取姿态特征
        @param keypoints: 关键点数据 (17, 3)
        @param objects: 物体检测结果
        @return: 特征字典
        """
        features = {}
        kp = keypoints

        # --- 关节角度 ---
        features["left_arm_angle"] = self._calculate_angle(kp[5], kp[7], kp[9])
        features["right_arm_angle"] = self._calculate_angle(kp[6], kp[8], kp[10])
        features["left_shoulder_angle"] = self._calculate_angle(kp[11], kp[5], kp[7])
        features["right_shoulder_angle"] = self._calculate_angle(kp[12], kp[6], kp[8])
        features["left_hip_angle"] = self._calculate_angle(kp[5], kp[11], kp[13])
        features["right_hip_angle"] = self._calculate_angle(kp[6], kp[12], kp[14])
        features["left_knee_angle"] = self._calculate_angle(kp[11], kp[13], kp[15])
        features["right_knee_angle"] = self._calculate_angle(kp[12], kp[14], kp[16])

        # --- 手部相对位置 ---
        left_wrist_y = kp[9][1]
        right_wrist_y = kp[10][1]
        left_shoulder_y = kp[5][1]
        right_shoulder_y = kp[6][1]
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2

        # 注意：图像坐标系y轴向下，所以 wrist_y < shoulder_y 表示手在肩上方
        features["arm_height_ratio"] = (avg_shoulder_y - avg_wrist_y) / (avg_shoulder_y + 1)

        # 手部是否在前方（基于x坐标相对身体中心偏移）
        body_center_x = (kp[5][0] + kp[6][0] + kp[11][0] + kp[12][0]) / 4
        left_wrist_x = kp[9][0]
        right_wrist_x = kp[10][0]
        features["hands_spread"] = abs(left_wrist_x - right_wrist_x) / (abs(kp[5][0] - kp[6][0]) + 1)

        # 手是否高于肩
        features["left_hand_above_shoulder"] = left_wrist_y < left_shoulder_y
        features["right_hand_above_shoulder"] = right_wrist_y < right_shoulder_y

        # 手是否在腰部以下
        left_hip_y = kp[11][1]
        right_hip_y = kp[12][1]
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        features["hands_below_hip"] = avg_wrist_y > avg_hip_y

        # --- 身体姿态 ---
        features["body_upright"] = (
            features["left_hip_angle"] > 160 and features["right_hip_angle"] > 160
        )
        features["body_bending"] = (
            features["left_hip_angle"] < 140 or features["right_hip_angle"] < 140
        )

        # --- 物体交互 ---
        features["object_in_hand"] = False
        features["object_type"] = None
        if objects:
            for obj in objects:
                obj_class = obj.get("class", "")
                obj_box = obj.get("bbox", None)
                if obj_box and self._is_near_hand(kp, obj_box):
                    features["object_in_hand"] = True
                    features["object_type"] = obj_class
                    break

        # --- 关键点置信度 ---
        features["avg_confidence"] = np.mean(kp[:, 2]) if kp.shape[1] >= 3 else 0.5

        return features

    def _is_near_hand(self, keypoints: np.ndarray, bbox: list, threshold: float = 80) -> bool:
        """判断物体是否靠近手部"""
        if bbox is None or len(bbox) < 4:
            return False
        obj_cx = (bbox[0] + bbox[2]) / 2
        obj_cy = (bbox[1] + bbox[3]) / 2

        for wrist_idx in [9, 10]:  # 左腕、右腕
            wx, wy = keypoints[wrist_idx][0], keypoints[wrist_idx][1]
            dist = math.sqrt((wx - obj_cx) ** 2 + (wy - obj_cy) ** 2)
            if dist < threshold:
                return True
        return False

    def _calculate_angle(self, p1, p2, p3) -> float:
        """
        @brief 计算三点形成的角度（p2为顶点）
        @param p1, p2, p3: 三个关键点 (x, y, conf)
        @return: 角度（度），失败返回0.0
        """
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                return 0.0
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle) * 180.0 / math.pi
            return angle
        except Exception:
            return 0.0

    def _apply_rules(self, features: Dict) -> Tuple[ActionType, float]:
        """
        @brief 应用规则判断动作
        @param features: 提取的特征
        @return: (动作类型, 置信度)
        """
        cfg = self.config

        # 先尝试自定义规则
        for rule in self.custom_rules:
            result = rule(features)
            if result is not None:
                return result

        arm_extended = (
            features["left_arm_angle"] > cfg["arm_extension_high"] or
            features["right_arm_angle"] > cfg["arm_extension_high"]
        )
        arm_bent = (
            features["left_arm_angle"] < cfg["arm_extension_low"] or
            features["right_arm_angle"] < cfg["arm_extension_low"]
        )
        arms_high = features["arm_height_ratio"] > cfg["hand_front_threshold"]
        hands_low = features.get("hands_below_hip", False)
        has_object = features.get("object_in_hand", False)

        # 1. 取料：手臂伸展+手在上方+持有物体
        if arm_extended and arms_high and has_object:
            return ActionType.PICKING, 0.85

        # 2. 放置：手臂弯曲+手在下方+持有物体
        if arm_bent and hands_low and has_object:
            return ActionType.PLACING, 0.80

        # 3. 操作：手臂弯曲+手在中间区域+持有物体
        if arm_bent and has_object and not hands_low:
            return ActionType.OPERATING, 0.75

        # 4. 检查：身体直立+手在下方+无物体
        if features.get("body_upright", False) and hands_low and not has_object:
            return ActionType.INSPECTING, 0.75

        # 5. 伸手：手臂伸展+手在上方+无物体
        if arm_extended and arms_high:
            return ActionType.REACHING, 0.70

        # 6. 收回：手臂弯曲+手在下方+无物体
        if arm_bent and hands_low:
            return ActionType.RETRACTING, 0.70

        # 7. 确认：身体弯曲
        if features.get("body_bending", False):
            return ActionType.CONFIRMING, 0.60

        # 8. 空闲
        return ActionType.IDLE, 0.90

    def add_custom_rule(self, rule_func):
        """
        @brief 添加自定义动作识别规则
        @param rule_func: 规则函数，接收features字典，返回(ActionType, float)或None
        """
        self.custom_rules.append(rule_func)

    def get_recent_actions(self, n: int = 10) -> List[Dict]:
        """获取最近N个动作记录"""
        return list(self.action_history)[-n:]

    def get_action_statistics(self) -> Dict:
        """获取动作统计"""
        if not self.action_history:
            return {}
        counts = {}
        for record in self.action_history:
            action = record["action"]
            counts[action] = counts.get(action, 0) + 1
        total = len(self.action_history)
        return {k.value: {"count": v, "ratio": v / total} for k, v in counts.items()}

    def reset(self):
        """重置识别器状态"""
        self.action_history.clear()
        self.smooth_window.clear()
