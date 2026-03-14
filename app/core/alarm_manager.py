"""
@file alarm_manager.py
@brief 多级别告警管理器 - 支持声音、界面、日志告警
@author AI Assistant
@date 2026-03-10
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from collections import deque


class AlarmLevel(Enum):
    """告警级别"""
    INFO = "提示"
    WARNING = "警告"
    ERROR = "错误"
    CRITICAL = "严重"


class AlarmType(Enum):
    """告警类型"""
    ORDER_ERROR = "顺序错误"
    TIMEOUT = "超时"
    MISSING_STEP = "遗漏步骤"
    ACTION_MISMATCH = "动作不匹配"
    SAFETY_VIOLATION = "安全违规"
    CUSTOM = "自定义"


class Alarm:
    """告警对象"""

    def __init__(self, level: AlarmLevel, alarm_type: AlarmType,
                 message: str, step_id: int = None, **kwargs):
        self.alarm_id = f"alarm_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.timestamp = datetime.now().isoformat()
        self.level = level
        self.alarm_type = alarm_type
        self.message = message
        self.step_id = step_id
        self.acknowledged = False
        self.screenshot_path = kwargs.get("screenshot_path", "")
        self.detected_action = kwargs.get("detected_action", "")
        self.expected_action = kwargs.get("expected_action", "")
        self.extra = kwargs

    def to_dict(self) -> Dict:
        return {
            "alarm_id": self.alarm_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "type": self.alarm_type.value,
            "message": self.message,
            "step_id": self.step_id,
            "acknowledged": self.acknowledged,
            "screenshot_path": self.screenshot_path,
            "detected_action": self.detected_action,
            "expected_action": self.expected_action,
        }


class AlarmManager:
    """
    @class AlarmManager
    @brief 多级别告警管理器

    功能：
    - 触发不同级别的告警
    - 播放告警声音
    - 保存告警截图
    - 记录告警日志
    - 告警历史查询
    """

    ALARM_LOG_DIR = os.path.join("data", "logs", "alarm")
    SCREENSHOT_DIR = os.path.join("data", "recordings", "screenshots")

    def __init__(self):
        self.active_alarms: List[Alarm] = []
        self.alarm_history: deque = deque(maxlen=1000)
        self.alarm_config = {
            "order_error": {"enabled": True, "level": AlarmLevel.ERROR},
            "timeout": {"enabled": True, "level": AlarmLevel.WARNING},
            "missing_step": {"enabled": True, "level": AlarmLevel.ERROR},
            "action_mismatch": {"enabled": True, "level": AlarmLevel.WARNING},
            "safety_violation": {"enabled": True, "level": AlarmLevel.CRITICAL},
        }
        self.sound_enabled = True
        self.screenshot_enabled = True

        # 回调
        self.on_alarm: Optional[Callable] = None  # 告警触发回调

        os.makedirs(self.ALARM_LOG_DIR, exist_ok=True)
        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)

    def load_config(self, alarm_config: Dict):
        """从流程配置加载告警设置"""
        for key, cfg in alarm_config.items():
            if key in self.alarm_config:
                self.alarm_config[key]["enabled"] = cfg.get("enabled", True)
                level_str = cfg.get("level", "WARNING")
                try:
                    self.alarm_config[key]["level"] = AlarmLevel[level_str]
                except KeyError:
                    self.alarm_config[key]["level"] = AlarmLevel.WARNING

    def trigger_alarm(self, alarm_type: AlarmType, message: str,
                      step_id: int = None, frame: np.ndarray = None,
                      **kwargs) -> Optional[Alarm]:
        """
        @brief 触发告警
        @param alarm_type: 告警类型
        @param message: 告警消息
        @param step_id: 关联的步骤ID
        @param frame: 当前帧（用于截图）
        @return: 告警对象，若被配置禁用则返回None
        """
        type_key = alarm_type.name.lower()
        config = self.alarm_config.get(type_key, {"enabled": True, "level": AlarmLevel.WARNING})
        if not config.get("enabled", True):
            return None

        level = config.get("level", AlarmLevel.WARNING)

        # 保存截图
        screenshot_path = ""
        if self.screenshot_enabled and frame is not None:
            screenshot_path = self._save_screenshot(frame, alarm_type)

        alarm = Alarm(
            level=level,
            alarm_type=alarm_type,
            message=message,
            step_id=step_id,
            screenshot_path=screenshot_path,
            **kwargs
        )

        self.active_alarms.append(alarm)
        self.alarm_history.append(alarm)

        # 写日志
        self._log_alarm(alarm)

        # 播放声音
        if self.sound_enabled and level in (AlarmLevel.WARNING, AlarmLevel.ERROR, AlarmLevel.CRITICAL):
            self._play_sound(level)

        # 回调通知UI
        if self.on_alarm:
            self.on_alarm(alarm)

        return alarm

    def trigger_from_validation(self, result_str: str, message: str,
                                step_id: int = None, frame: np.ndarray = None,
                                **kwargs) -> Optional[Alarm]:
        """从验证结果字符串触发告警"""
        type_map = {
            "顺序错误": AlarmType.ORDER_ERROR,
            "超时": AlarmType.TIMEOUT,
            "遗漏步骤": AlarmType.MISSING_STEP,
            "动作不匹配": AlarmType.ACTION_MISMATCH,
        }
        alarm_type = type_map.get(result_str, AlarmType.CUSTOM)
        return self.trigger_alarm(alarm_type, message, step_id, frame, **kwargs)

    def acknowledge_alarm(self, alarm_id: str) -> bool:
        """确认告警"""
        for alarm in self.active_alarms:
            if alarm.alarm_id == alarm_id:
                alarm.acknowledged = True
                self.active_alarms.remove(alarm)
                return True
        return False

    def acknowledge_all(self):
        """确认所有告警"""
        for alarm in self.active_alarms:
            alarm.acknowledged = True
        self.active_alarms.clear()

    def get_active_alarms(self) -> List[Dict]:
        """获取活跃告警"""
        return [a.to_dict() for a in self.active_alarms]

    def get_alarm_history(self, limit: int = 100) -> List[Dict]:
        """获取告警历史"""
        return [a.to_dict() for a in list(self.alarm_history)[-limit:]]

    def get_alarm_statistics(self) -> Dict:
        """获取告警统计"""
        stats = {"total": len(self.alarm_history)}
        level_counts = {}
        type_counts = {}
        for alarm in self.alarm_history:
            lv = alarm.level.value
            level_counts[lv] = level_counts.get(lv, 0) + 1
            at = alarm.alarm_type.value
            type_counts[at] = type_counts.get(at, 0) + 1
        stats["by_level"] = level_counts
        stats["by_type"] = type_counts
        return stats

    def _save_screenshot(self, frame: np.ndarray, alarm_type: AlarmType) -> str:
        """保存告警截图"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{alarm_type.name}_{timestamp}.jpg"
            filepath = os.path.join(self.SCREENSHOT_DIR, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"保存告警截图失败: {e}")
            return ""

    def _log_alarm(self, alarm: Alarm):
        """写告警日志"""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = os.path.join(self.ALARM_LOG_DIR, f"{date_str}.log")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alarm.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"写告警日志失败: {e}")

    def _play_sound(self, level: AlarmLevel):
        """播放告警声音（简化版本，使用系统提示音）"""
        try:
            import platform
            if platform.system() == "Darwin":
                if level == AlarmLevel.CRITICAL:
                    os.system('afplay /System/Library/Sounds/Sosumi.aiff &')
                else:
                    os.system('afplay /System/Library/Sounds/Tink.aiff &')
            elif platform.system() == "Linux":
                # RK3588 等嵌入式Linux
                if level == AlarmLevel.CRITICAL:
                    os.system('aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null &')
                else:
                    os.system('aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null &')
        except Exception:
            pass  # 静默失败

    def clear_history(self):
        """清空历史"""
        self.alarm_history.clear()
        self.active_alarms.clear()
