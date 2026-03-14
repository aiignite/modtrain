# 生产过程人工操作监控程序设计方案（嵌入式优化版）

## 一、方案调整说明

### 1.1 嵌入式平台特点

| 平台特点       | 影响分析                   |
| -------------- | -------------------------- |
| **RK系列芯片** | ARM架构，支持GPU/NPU加速   |
| **资源有限**   | 需要轻量级模型，低功耗运行 |
| **离线运行**   | 不依赖网络，本地推理       |
| **实时性要求** | 需要快速响应，低延迟       |

### 1.2 方案核心思路

```
用户建议方案（已采纳）：
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  关节角度计算   │ +  │  关键物体识别   │ +  │  代码规则判断   │
│ (已有YOLO-Pose)│    │ (YOLO目标检测)  │    │ (可配置规则引擎)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   动作识别与顺序验证  │
                    └─────────────────┘
```

**优势**：

- 纯代码逻辑判断，计算量小，适合嵌入式
- 可配置规则，无需重新训练模型
- 易于调试和维护
- 可选离线深度学习提升效果

---

## 二、系统架构设计

### 2.1 整体架构（嵌入式优化版）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         嵌入式生产操作监控系统                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐     │
│  │   摄像头输入   │    │   视频录制     │    │   数据存储     │     │
│  └───────┬────────┘    └───────┬────────┘    └───────┬────────┘     │
│          │                     │                     │                  │
│          └─────────────────────┼─────────────────────┘                  │
│                                ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    双模型推理引擎（可选切换）                      │     │
│  │  ┌─────────────────────┐      ┌─────────────────────┐           │     │
│  │  │  YOLO-Pose 姿态估计 │      │  YOLO 物体检测     │           │     │
│  │  │  (17关键点/30fps)   │      │  (关键对象识别)   │           │     │
│  │  └─────────────────────┘      └─────────────────────┘           │     │
│  └───────────────────────────────┬──────────────────────────────────┘     │
│                                  ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                      特征提取计算模块                             │     │
│  │  • 关节角度计算（8个）  • 身体朝向    • 手部位置区域            │     │
│  │  • 物体位置检测        • 物体交互状态 • 运动轨迹分析            │     │
│  └───────────────────────────────┬──────────────────────────────────┘     │
│                                  ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                      规则引擎（核心判断逻辑）                      │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │     │
│  │  │ 动作状态识别  │  │ 步骤顺序验证  │  │ 时间约束检查  │         │     │
│  │  │ (代码逻辑)   │  │ (状态机)     │  │ (计时器)     │         │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘         │     │
│  └───────────────────────────────┬──────────────────────────────────┘     │
│                                  ▼                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │   告警输出   │  │   界面显示   │  │   数据记录   │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 嵌入式部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      RK3588 嵌入式部署                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Camera     │    │  USB/CSI    │    │  HDMI      │        │
│  │  (USB/Webcam)│    │  Camera    │    │  Display   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   RK3588 NPU/GPU                         │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │  ONNX Runtime (NPU加速)                          │  │   │
│  │  │  • YOLOv8n-Pose (姿态估计) ~3ms/帧              │  │   │
│  │  │  • YOLOv8n (物体检测) ~2ms/帧                   │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Python 应用层                          │   │
│  │  • 规则引擎 (Python)  • UI (PyQt5)  • 数据存储         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、核心实现方案

### 3.1 动作识别算法（代码逻辑版）

#### 3.1.1 基于关节角度的动作判断

```python
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


class ActionType(Enum):
    """动作类型枚举"""
    IDLE = "空闲"
    PICKING = "取料"
    PLACING = "放置"
    INSPECTING = "检查"
    CONFIRMING = "确认"
    REACHING = "伸手"
    RETRACTING = "收回"


class ActionRecognizer:
    """
    @class ActionRecognizer
    @brief 基于规则的动作识别器

    通过关节角度、物体位置、动作状态等特征，
    使用预定义的规则逻辑判断当前动作类型
    """

    def __init__(self, config: Dict = None):
        """
        @brief 初始化动作识别器
        @param config: 识别配置参数
        """
        self.config = config or self._get_default_config()
        self.action_history = []
        self.confidence_threshold = 0.7

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "arm_extension_high": 140,    # 手臂伸展角度阈值
            "arm_extension_low": 60,     # 手臂弯曲角度阈值
            "hand_front_threshold": 0.6, # 手部在前方阈值
            "hand_down_threshold": 0.4,   # 手部向下阈值
            "body_forward_threshold": 0.7, # 身体前倾阈值
            "stability_frames": 5,        # 稳定帧数
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
        action, confidence = self._apply_rules(features)

        self.action_history.append({
            "action": action,
            "confidence": confidence,
            "features": features
        })

        return action, confidence

    def _extract_features(self, keypoints: np.ndarray,
                         objects: List[Dict] = None) -> Dict:
        """
        @brief 提取姿态特征
        @param keypoints: 关键点数据
        @param objects: 物体检测结果
        @return: 特征字典
        """
        features = {}

        kp = keypoints

        features["left_arm_angle"] = self._calculate_angle(
            kp[5], kp[7], kp[9])   # 左肩-左肘-左腕
        features["right_arm_angle"] = self._calculate_angle(
            kp[6], kp[8], kp[10])  # 右肩-右肘-右腕

        features["left_shoulder_angle"] = self._calculate_angle(
            kp[11], kp[5], kp[7])  # 左髋-左肩-左肘
        features["right_shoulder_angle"] = self._calculate_angle(
            kp[12], kp[6], kp[8])  # 右髋-右肩-右肘

        features["left_hip_angle"] = self._calculate_angle(
            kp[5], kp[11], kp[13])  # 左肩-左髋-左膝
        features["right_hip_angle"] = self._calculate_angle(
            kp[6], kp[12], kp[14])  # 右肩-右髋-右膝

        features["left_knee_angle"] = self._calculate_angle(
            kp[11], kp[13], kp[15])  # 左髋-左膝-左踝
        features["right_knee_angle"] = self._calculate_angle(
            kp[12], kp[14], kp[16])  # 右髋-右膝-右踝

        wrist_y = (kp[9][1] + kp[10][1]) / 2
        shoulder_y = (kp[5][1] + kp[6][1]) / 2
        features["arm_height_ratio"] = (shoulder_y - wrist_y) / (shoulder_y + 1)

        left_hand_pos = (kp[9][0], kp[9][1])
        right_hand_pos = (kp[10][0], kp[10][1])
        nose_pos = (kp[0][0], kp[0][1])

        features["hands_front_of_nose"] = (
            left_hand_pos[1] < nose_pos[1] + 50 or
            right_hand_pos[1] < nose_pos[1] + 50
        )

        features["object_in_hand"] = False
        if objects:
            for obj in objects:
                if obj.get("class") in ["hand", "object", "component"]:
                    features["object_in_hand"] = True
                    break

        return features

    def _calculate_angle(self, p1, p2, p3) -> float:
        """
        @brief 计算三点形成的角度
        @param p1, p2, p3: 三个关键点 (x, y, conf)
        @return: 角度（度）
        """
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle) * 180.0 / math.pi
            return angle
        except:
            return 0.0

    def _apply_rules(self, features: Dict) -> Tuple[ActionType, float]:
        """
        @brief 应用规则判断动作
        @param features: 提取的特征
        @return: (动作类型, 置信度)
        """
        arm_extended = (
            features["left_arm_angle"] > self.config["arm_extension_high"] or
            features["right_arm_angle"] > self.config["arm_extension_high"]
        )

        arm_bent = (
            features["left_arm_angle"] < self.config["arm_extension_low"] or
            features["right_arm_angle"] < self.config["arm_extension_low"]
        )

        arms_high = features["arm_height_ratio"] > self.config["hand_front_threshold"]

        arms_down = features["arm_height_ratio"] < self.config["hand_down_threshold"]

        inspecting = (
            features["left_hip_angle"] > 160 and
            features["right_hip_angle"] > 160 and
            arms_down
        )

        if arm_extended and arms_high and features.get("object_in_hand", False):
            return ActionType.PICKING, 0.85

        if arm_bent and arms_down and features.get("object_in_hand", False):
            return ActionType.PLACING, 0.80

        if inspecting:
            return ActionType.INSPECTING, 0.75

        if arm_extended and arms_high:
            return ActionType.REACHING, 0.70

        if arm_bent and arms_down:
            return ActionType.RETRACTING, 0.70

        if features.get("object_in_hand", False):
            return ActionType.CONFIRMING, 0.60

        return ActionType.IDLE, 0.90
```

#### 3.1.2 关键物体识别配置

```python
# 物体检测配置 - 可根据实际生产场景定义
OBJECT_DETECTION_CONFIG = {
    "classes": {
        "component": {
            "id": 0,
            "name": "元件",
            "color": (255, 0, 0),
            "critical": True
        },
        "tool": {
            "id": 1,
            "name": "工具",
            "color": (0, 255, 0),
            "critical": False
        },
        "button": {
            "id": 2,
            "name": "按钮",
            "color": (0, 0, 255),
            "critical": True
        },
        "fixture": {
            "id": 3,
            "name": "夹具",
            "color": (255, 255, 0),
            "critical": False
        }
    },
    "iou_threshold": 0.45,
    "confidence_threshold": 0.5
}
```

### 3.2 步骤顺序验证引擎

```python
"""
@file sequence_validator.py
@brief 操作步骤顺序验证引擎 - 有限状态机实现
@author AI Assistant
@date 2026-03-10
"""

from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "待执行"
    IN_PROGRESS = "进行中"
    COMPLETED = "已完成"
    SKIPPED = "已跳过"
    FAILED = "失败"
    TIMEOUT = "超时"


class ValidationResult(Enum):
    """验证结果"""
    VALID = "有效"
    INVALID_ORDER = "顺序错误"
    INVALID_ACTION = "动作不匹配"
    TIMEOUT = "超时"
    MISSING_STEP = "遗漏步骤"
    UNKNOWN = "未知"


@dataclass
class StepDefinition:
    """步骤定义"""
    step_id: int
    step_name: str
    expected_actions: List[str]
    min_duration: float = 1.0
    max_duration: float = 10.0
    required: bool = True
    allow_skip: bool = False


class SequenceValidator:
    """
    @class SequenceValidator
    @brief 步骤顺序验证器 - 有限状态机实现

    核心功能：
    - 验证操作步骤顺序
    - 检查步骤持续时间
    - 检测遗漏步骤
    - 触发超时告警
    """

    def __init__(self, steps: List[StepDefinition]):
        """
        @brief 初始化验证器
        @param steps: 步骤定义列表
        """
        self.steps = {step.step_id: step for step in steps}
        self.step_order = [step.step_id for step in steps]

        self.current_step_id = None
        self.step_status = {}  # step_id -> StepStatus
        self.step_start_time = {}
        self.step_durations = {}

        self.error_history = []
        self.on_error_callback = None
        self.on_step_complete_callback = None

    def validate(self, detected_action: str,
                 timestamp: float = None) -> Tuple[ValidationResult, str]:
        """
        @brief 验证当前动作
        @param detected_action: 检测到的动作类型
        @param timestamp: 时间戳
        @return: (验证结果, 详情消息)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        if self.current_step_id is None:
            first_step = self.steps[self.step_order[0]]
            if self._action_matches(detected_action, first_step.expected_actions):
                self._start_step(self.step_order[0], timestamp)
                return ValidationResult.VALID, f"开始步骤: {first_step.step_name}"
            return ValidationResult.UNKNOWN, "等待开始第一个步骤"

        current_step = self.steps[self.current_step_id]

        if self._action_matches(detected_action, current_step.expected_actions):
            duration = timestamp - self.step_start_time.get(self.current_step_id, timestamp)

            if duration >= current_step.min_duration:
                self._complete_step(self.current_step_id)

                next_step_id = self._get_next_step(self.current_step_id)
                if next_step_id:
                    self._start_step(next_step_id, timestamp)
                    return ValidationResult.VALID, f"步骤完成，进入: {self.steps[next_step_id].step_name}"
                else:
                    return ValidationResult.VALID, "所有步骤已完成"

        if duration > current_step.max_duration:
            self.step_status[self.current_step_id] = StepStatus.TIMEOUT
            error_msg = f"步骤 {current_step.step_name} 超时"
            self._record_error(error_msg)
            return ValidationResult.TIMEOUT, error_msg

        return ValidationResult.INVALID_ACTION, f"动作不匹配，当前应为: {current_step.expected_actions}"

    def _action_matches(self, action: str, expected_actions: List[str]) -> bool:
        """检查动作是否匹配"""
        return action in expected_actions

    def _start_step(self, step_id: int, timestamp: float):
        """开始步骤"""
        self.current_step_id = step_id
        self.step_status[step_id] = StepStatus.IN_PROGRESS
        self.step_start_time[step_id] = timestamp

    def _complete_step(self, step_id: int):
        """完成步骤"""
        self.step_status[step_id] = StepStatus.COMPLETED
        step = self.steps[step_id]

        if self.on_step_complete_callback:
            self.on_step_complete_callback(step)

    def _get_next_step(self, current_step_id: int) -> Optional[int]:
        """获取下一步"""
        try:
            current_idx = self.step_order.index(current_step_id)
            if current_idx + 1 < len(self.step_order):
                return self.step_order[current_idx + 1]
        except ValueError:
            pass
        return None

    def _record_error(self, error_msg: str):
        """记录错误"""
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": error_msg,
            "current_step": self.current_step_id
        })

        if self.on_error_callback:
            self.on_error_callback(error_msg)

    def get_progress(self) -> Dict:
        """获取进度"""
        completed = sum(1 for s in self.step_status.values() if s == StepStatus.COMPLETED)
        total = len(self.steps)

        return {
            "total_steps": total,
            "completed_steps": completed,
            "current_step": self.current_step_id,
            "current_step_name": self.steps[self.current_step_id].step_name if self.current_step_id else None,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }

    def reset(self):
        """重置验证器"""
        self.current_step_id = None
        self.step_status = {}
        self.step_start_time = {}
        self.step_durations = {}
        self.error_history = []
```

### 3.3 可配置规则系统

```python
"""
@file rule_engine.py
@brief 可配置规则引擎 - 支持JSON配置文件定义规则
@author AI Assistant
@date 2026-03-10
"""

import json
from typing import Dict, List, Any
from pathlib import Path


class RuleEngine:
    """
    @class RuleEngine
    @brief 可配置规则引擎

    用户可以通过JSON配置文件定义：
    - 操作步骤序列
    - 每个步骤的预期动作
    - 时间约束
    - 告警规则
    """

    def __init__(self, config_path: str = None):
        """
        @brief 初始化规则引擎
        @param config_path: 配置文件路径
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def get_process_config(self) -> Dict:
        """获取流程配置"""
        return self.config.get("process", {})

    def get_steps_config(self) -> List[Dict]:
        """获取步骤配置"""
        return self.config.get("steps", [])

    def get_alarm_config(self) -> Dict:
        """获取告警配置"""
        return self.config.get("alarms", {})

    def validate_config(self) -> bool:
        """验证配置有效性"""
        required_keys = ["process", "steps"]
        return all(key in self.config for key in required_keys)


# 示例配置文件
EXAMPLE_CONFIG = {
    "process": {
        "process_id": "PROC_001",
        "process_name": "PCB元件安装流程",
        "version": "1.0",
        "description": "标准PCB元件安装操作流程"
    },
    "steps": [
        {
            "step_id": 1,
            "step_name": "取料",
            "expected_actions": ["PICKING", "REACHING"],
            "min_duration": 1.0,
            "max_duration": 5.0,
            "required": True,
            "description": "从料盒取出元件"
        },
        {
            "step_id": 2,
            "step_name": "移动",
            "expected_actions": ["REACHING"],
            "min_duration": 0.5,
            "max_duration": 3.0,
            "required": True,
            "description": "将元件移动到安装位置"
        },
        {
            "step_id": 3,
            "step_name": "放置",
            "expected_actions": ["PLACING"],
            "min_duration": 1.0,
            "max_duration": 5.0,
            "required": True,
            "description": "将元件放置到PCB指定位置"
        },
        {
            "step_id": 4,
            "step_name": "检查",
            "expected_actions": ["INSPECTING"],
            "min_duration": 2.0,
            "max_duration": 10.0,
            "required": True,
            "description": "检查安装位置是否正确"
        },
        {
            "step_id": 5,
            "step_name": "确认",
            "expected_actions": ["CONFIRMING", "RETRACTING"],
            "min_duration": 0.5,
            "max_duration": 3.0,
            "required": True,
            "description": "按键确认完成"
        }
    ],
    "alarms": {
        "order_error": {
            "enabled": True,
            "level": "ERROR",
            "message": "操作顺序错误"
        },
        "timeout": {
            "enabled": True,
            "level": "WARNING",
            "message": "步骤执行超时"
        },
        "missing_step": {
            "enabled": True,
            "level": "ERROR",
            "message": "遗漏关键步骤"
        },
        "action_mismatch": {
            "enabled": True,
            "level": "WARNING",
            "message": "动作不匹配"
        }
    }
}
```

---

## 四、嵌入式部署优化

### 4.1 模型优化策略

| 优化方法                 | 效果              | 适用场景     |
| ------------------------ | ----------------- | ------------ |
| **模型量化 (FP16/INT8)** | 推理速度提升2-4倍 | RK3588 NPU   |
| **模型裁剪**             | 参数量减少50%+    | 资源受限场景 |
| **TensorRT优化**         | 延迟降低30-50%    | NVIDIA GPU   |
| **ONNX Runtime**         | 跨平台，高效      | RK3588/通用  |

### 4.2 RK3588 性能预估

```
┌─────────────────────────────────────────────────────────────┐
│              RK3588 性能测试预估 (YOLOv8n)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  模型            分辨率    NPU延迟    GPU延迟    CPU延迟   │
│  ─────────────────────────────────────────────────────────  │
│  YOLOv8n-pose   640x640   ~3ms      ~8ms      ~25ms      │
│  YOLOv8n        640x640   ~2ms      ~5ms      ~15ms      │
│  双模型同时      640x640   ~5ms      ~13ms     ~40ms      │
│                                                              │
│  总帧率预估:                                                  │
│  - 单姿态模型: ~30 FPS (33ms/帧，含处理)                    │
│  - 双模型:     ~20 FPS (50ms/帧，含处理)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 资源需求

```
┌─────────────────────────────────────────────────────────────┐
│                    嵌入式平台资源需求                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  资源项          最小配置        推荐配置                    │
│  ─────────────────────────────────────────────────────────  │
│  内存            2GB           4GB+                        │
│  存储            8GB           16GB+ (含视频存储)          │
│  NPU/GPU         必须           必须 (加速推理)             │
│  摄像头          1个           2个 (多角度)                │
│  系统            Debian/Ubuntu 18.04+                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、实施计划

### 5.1 开发阶段

| 阶段         | 任务         | 预计时间 | 交付物               |
| ------------ | ------------ | -------- | -------------------- |
| **第一阶段** | 核心框架     | 2天      | 动作识别器、规则引擎 |
| **第二阶段** | 物体检测集成 | 2天      | 双模型推理配置       |
| **第三阶段** | 流程配置系统 | 2天      | JSON配置、规则引擎   |
| **第四阶段** | UI界面开发   | 2天      | 监控界面、配置界面   |
| **第五阶段** | 告警系统     | 1天      | 多级别告警           |
| **第六阶段** | 嵌入式优化   | 2天      | ONNX模型优化         |

### 5.2 文件结构

```
app/
├── core/
│   ├── action_recognizer.py      # 动作识别（代码逻辑）
│   ├── sequence_validator.py      # 步骤顺序验证
│   ├── rule_engine.py            # 可配置规则引擎
│   ├── alarm_manager.py          # 告警管理
│   └── dual_model_runner.py      # 双模型推理
├── ui/
│   ├── operation_monitoring.py   # 操作监控界面
│   └── process_config_ui.py     # 流程配置界面
└── main.py                       # 主入口

config/
├── processes/                     # 流程配置目录
│   ├── pcb_assembly.json         # PCB装配流程
│   ├── welding.json              # 焊接流程
│   └── inspection.json           # 检测流程
└── model_config.json             # 模型配置

models/
├── yolov8n-pose.onnx             # 姿态模型(ONNX)
└── yolov8n.onnx                  # 物体检测模型(ONNX)
```

---

## 六、方案优势总结

### 6.1 嵌入式友好

- ✅ 轻量级模型：YOLOv8n-Pose + YOLOv8n
- ✅ 代码逻辑判断：无需复杂深度学习推理
- ✅ 可配置规则：无需重新训练模型
- ✅ ONNX格式：支持RK3588 NPU加速

### 6.2 效果优化选项

如需更好效果，可选方案：

| 方案                  | 效果提升 | 资源消耗 | 实施难度 |
| --------------------- | -------- | -------- | -------- |
| **更大模型(yolov8m)** | +15%     | +3倍     | 低       |
| **ST-GCN动作识别**    | +20%     | +2倍     | 中       |
| **多摄像头融合**      | +25%     | +2倍     | 中       |
| **自定义动作模型**    | +30%     | +2倍     | 高       |

---

**请确认以上方案是否符合您的需求，确认后我将开始实施。**
