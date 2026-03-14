"""
@file sequence_validator.py
@brief 操作步骤顺序验证引擎 - 有限状态机实现
@author AI Assistant
@date 2026-03-10
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


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
    STEP_STARTED = "步骤开始"
    STEP_COMPLETED = "步骤完成"
    ALL_COMPLETED = "全部完成"
    INVALID_ORDER = "顺序错误"
    INVALID_ACTION = "动作不匹配"
    TIMEOUT = "超时"
    MISSING_STEP = "遗漏步骤"
    WAITING = "等待中"
    UNKNOWN = "未知"


class MonitoringState(Enum):
    """监控状态"""
    IDLE = "空闲"
    WAITING = "等待动作"
    IN_PROGRESS = "进行中"
    COMPLETED = "已完成"
    ERROR = "错误"
    TIMEOUT = "超时"
    PAUSED = "暂停"


@dataclass
class StepDefinition:
    """步骤定义"""
    step_id: int
    step_name: str
    expected_actions: List[str]
    description: str = ""
    min_duration: float = 0.5
    max_duration: float = 30.0
    required: bool = True
    allow_skip: bool = False
    alarm_on_timeout: bool = True
    alarm_on_skip: bool = True
    alarm_on_error: bool = True
    reference_video_path: str = ""

    @staticmethod
    def from_dict(data: Dict) -> 'StepDefinition':
        """从字典创建"""
        return StepDefinition(
            step_id=data.get("step_id", 0),
            step_name=data.get("step_name", data.get("name", "")),
            expected_actions=data.get("expected_actions", []),
            description=data.get("description", ""),
            min_duration=data.get("min_duration", 0.5),
            max_duration=data.get("max_duration", 30.0),
            required=data.get("required", True),
            allow_skip=data.get("allow_skip", False),
            alarm_on_timeout=data.get("alarm_on_timeout", True),
            alarm_on_skip=data.get("alarm_on_skip", True),
            alarm_on_error=data.get("alarm_on_error", True),
            reference_video_path=data.get("reference_video_path", ""),
        )

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "expected_actions": self.expected_actions,
            "description": self.description,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "required": self.required,
            "allow_skip": self.allow_skip,
            "alarm_on_timeout": self.alarm_on_timeout,
            "alarm_on_skip": self.alarm_on_skip,
            "alarm_on_error": self.alarm_on_error,
            "reference_video_path": self.reference_video_path,
        }


@dataclass
class ValidationEvent:
    """验证事件记录"""
    timestamp: str
    result: str
    message: str
    step_id: Optional[int] = None
    step_name: str = ""
    detected_action: str = ""
    duration: float = 0.0


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

    def __init__(self, steps: List[StepDefinition] = None):
        """
        @brief 初始化验证器
        @param steps: 步骤定义列表
        """
        self.steps = {}
        self.step_order = []
        if steps:
            self.load_steps(steps)

        # 运行时状态
        self.state = MonitoringState.IDLE
        self.current_step_idx = -1
        self.step_status = {}
        self.step_start_time = {}
        self.step_durations = {}
        self.step_match_counts = {}     # 每个步骤的动作匹配次数
        self.step_total_counts = {}     # 每个步骤的总检测次数
        self.step_mismatch_streak = 0   # 连续不匹配帧数

        # 事件记录
        self.event_history: List[ValidationEvent] = []
        self.error_count = 0
        self.warning_count = 0

        # 回调函数
        self.on_step_complete: Optional[Callable] = None
        self.on_step_start: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_all_complete: Optional[Callable] = None

        # 配置
        self.strict_order = True
        self.allow_reverse = False
        self.match_ratio_threshold = 0.3    # 匹配率阈值，超过此值认为步骤完成
        self.mismatch_tolerance = 30        # 连续不匹配帧数容忍上限

    def load_steps(self, steps: List[StepDefinition]):
        """加载步骤定义"""
        self.steps = {step.step_id: step for step in steps}
        self.step_order = [step.step_id for step in steps]
        self.reset()

    def load_from_config(self, config: Dict):
        """从配置字典加载"""
        steps_data = config.get("steps", [])
        steps = [StepDefinition.from_dict(s) for s in steps_data]
        self.load_steps(steps)

        constraints = config.get("constraints", {})
        self.strict_order = constraints.get("strict_order", True)
        self.allow_reverse = constraints.get("allow_reverse", False)
        self.match_ratio_threshold = constraints.get("match_ratio_threshold", 0.3)
        self.mismatch_tolerance = constraints.get("mismatch_tolerance", 30)

    def start(self):
        """开始验证流程"""
        self.state = MonitoringState.WAITING
        self.current_step_idx = 0
        self._record_event(ValidationResult.WAITING, "等待开始第一个步骤")

    def validate(self, detected_action: str,
                 timestamp: float = None) -> Tuple[ValidationResult, str]:
        """
        @brief 验证当前检测到的动作
        @param detected_action: 检测到的动作类型字符串（中文名称，如"取料"、"放置"）
        @param timestamp: 时间戳（秒），为None则使用当前时间
        @return: (验证结果, 描述消息)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        if self.state == MonitoringState.IDLE:
            return ValidationResult.UNKNOWN, "验证器未启动"

        if self.state == MonitoringState.COMPLETED:
            return ValidationResult.ALL_COMPLETED, "所有步骤已完成"

        if self.state == MonitoringState.PAUSED:
            return ValidationResult.WAITING, "验证已暂停"

        if self.current_step_idx < 0 or self.current_step_idx >= len(self.step_order):
            return ValidationResult.ALL_COMPLETED, "所有步骤已完成"

        current_step_id = self.step_order[self.current_step_idx]
        current_step = self.steps[current_step_id]
        is_match = self._action_matches(detected_action, current_step.expected_actions)

        # ===== 状态：等待动作 =====
        if self.state == MonitoringState.WAITING:
            if is_match:
                self._start_step(current_step_id, timestamp)
                self.step_match_counts[current_step_id] = 1
                self.step_total_counts[current_step_id] = 1
                self.step_mismatch_streak = 0
                msg = f"开始步骤 {current_step.step_id}: {current_step.step_name}"
                self._record_event(ValidationResult.STEP_STARTED, msg,
                                   step_id=current_step_id,
                                   detected_action=detected_action)
                return ValidationResult.STEP_STARTED, msg

            # 检查是否是后续步骤的动作（顺序错误）
            if self.strict_order:
                for future_idx in range(self.current_step_idx + 1, len(self.step_order)):
                    future_step = self.steps[self.step_order[future_idx]]
                    if self._action_matches(detected_action, future_step.expected_actions):
                        msg = f"顺序错误: 检测到步骤 {future_step.step_name} 的动作，当前应为 {current_step.step_name}"
                        self._record_event(ValidationResult.INVALID_ORDER, msg,
                                           step_id=current_step_id,
                                           detected_action=detected_action)
                        self.error_count += 1
                        if self.on_error:
                            self.on_error(msg, "ERROR")
                        return ValidationResult.INVALID_ORDER, msg

            return ValidationResult.WAITING, f"等待: {current_step.step_name} (期望: {', '.join(current_step.expected_actions)}, 当前: {detected_action})"

        # ===== 状态：进行中 =====
        if self.state == MonitoringState.IN_PROGRESS:
            duration = timestamp - self.step_start_time.get(current_step_id, timestamp)

            # 统计匹配率
            self.step_total_counts[current_step_id] = self.step_total_counts.get(current_step_id, 0) + 1
            if is_match:
                self.step_match_counts[current_step_id] = self.step_match_counts.get(current_step_id, 0) + 1
                self.step_mismatch_streak = 0
            else:
                self.step_mismatch_streak += 1

            total_cnt = self.step_total_counts[current_step_id]
            match_cnt = self.step_match_counts.get(current_step_id, 0)
            match_ratio = match_cnt / total_cnt if total_cnt > 0 else 0

            # 检查超时
            if duration > current_step.max_duration:
                self.step_status[current_step_id] = StepStatus.TIMEOUT
                msg = f"步骤 {current_step.step_name} 超时 ({duration:.1f}s > {current_step.max_duration}s)"
                self._record_event(ValidationResult.TIMEOUT, msg,
                                   step_id=current_step_id, duration=duration)
                self.warning_count += 1
                if self.on_error:
                    self.on_error(msg, "WARNING")
                # 超时后自动进入下一步
                has_next = self._advance_step(timestamp)
                if not has_next:
                    self.state = MonitoringState.COMPLETED
                return ValidationResult.TIMEOUT, msg

            # 检查连续不匹配是否超过容忍上限
            if self.step_mismatch_streak >= self.mismatch_tolerance:
                # 动作中断太久，检查匹配率是否足够
                if duration >= current_step.min_duration and match_ratio >= self.match_ratio_threshold:
                    # 匹配率达标，可以完成
                    self._complete_step(current_step_id, duration)
                    step_name = current_step.step_name
                    has_next = self._advance_step(timestamp)
                    if has_next:
                        next_step = self.steps[self.step_order[self.current_step_idx]]
                        msg = f"步骤 {step_name} 完成 ({duration:.1f}s, 匹配率{match_ratio:.0%})，下一步: {next_step.step_name}"
                    else:
                        msg = f"步骤 {step_name} 完成 ({duration:.1f}s, 匹配率{match_ratio:.0%})，所有步骤已完成！"
                        self.state = MonitoringState.COMPLETED
                        if self.on_all_complete:
                            self.on_all_complete()
                    self._record_event(ValidationResult.STEP_COMPLETED, msg,
                                       step_id=current_step_id, duration=duration,
                                       detected_action=detected_action)
                    return ValidationResult.STEP_COMPLETED, msg
                else:
                    # 匹配率不足，步骤中断，回到等待状态重新检测
                    self.step_mismatch_streak = 0
                    msg = f"步骤 {current_step.step_name} 动作中断 (匹配率{match_ratio:.0%} < {self.match_ratio_threshold:.0%})，重新等待"
                    self.state = MonitoringState.WAITING
                    self.step_status[current_step_id] = StepStatus.PENDING
                    self.warning_count += 1
                    return ValidationResult.INVALID_ACTION, msg

            # 动作匹配且持续时间达标
            if is_match and duration >= current_step.min_duration and match_ratio >= self.match_ratio_threshold:
                self._complete_step(current_step_id, duration)
                step_name = current_step.step_name
                has_next = self._advance_step(timestamp)
                if has_next:
                    next_step = self.steps[self.step_order[self.current_step_idx]]
                    msg = f"步骤 {step_name} 完成 ({duration:.1f}s, 匹配率{match_ratio:.0%})，下一步: {next_step.step_name}"
                else:
                    msg = f"步骤 {step_name} 完成 ({duration:.1f}s, 匹配率{match_ratio:.0%})，所有步骤已完成！"
                    self.state = MonitoringState.COMPLETED
                    if self.on_all_complete:
                        self.on_all_complete()
                self._record_event(ValidationResult.STEP_COMPLETED, msg,
                                   step_id=current_step_id, duration=duration,
                                   detected_action=detected_action)
                return ValidationResult.STEP_COMPLETED, msg

            # 步骤进行中
            status_str = "✓" if is_match else "✗"
            return ValidationResult.VALID, f"步骤 {current_step.step_name} 进行中 ({duration:.1f}s, 匹配: {status_str}, 率: {match_ratio:.0%})"

        return ValidationResult.UNKNOWN, "未知状态"

    def _action_matches(self, action: str, expected_actions: List[str]) -> bool:
        """检查动作是否匹配（不区分大小写，支持中文动作名）"""
        if not action:
            return False
        action_lower = action.lower().strip()
        for exp in expected_actions:
            if exp.lower().strip() == action_lower:
                return True
        return False

    def _start_step(self, step_id: int, timestamp: float):
        """开始步骤"""
        self.step_status[step_id] = StepStatus.IN_PROGRESS
        self.step_start_time[step_id] = timestamp
        self.state = MonitoringState.IN_PROGRESS
        if self.on_step_start:
            self.on_step_start(self.steps[step_id])

    def _complete_step(self, step_id: int, duration: float):
        """完成步骤"""
        self.step_status[step_id] = StepStatus.COMPLETED
        self.step_durations[step_id] = duration
        if self.on_step_complete:
            self.on_step_complete(self.steps[step_id], duration)

    def _advance_step(self, timestamp: float) -> bool:
        """前进到下一步，返回是否有下一步"""
        self.current_step_idx += 1
        if self.current_step_idx < len(self.step_order):
            self.state = MonitoringState.WAITING
            return True
        return False

    def _record_event(self, result: ValidationResult, message: str, **kwargs):
        """记录事件"""
        event = ValidationEvent(
            timestamp=datetime.now().isoformat(),
            result=result.value,
            message=message,
            step_id=kwargs.get("step_id"),
            step_name=self.steps[kwargs["step_id"]].step_name if kwargs.get("step_id") and kwargs["step_id"] in self.steps else "",
            detected_action=kwargs.get("detected_action", ""),
            duration=kwargs.get("duration", 0.0)
        )
        self.event_history.append(event)

    def pause(self):
        """暂停验证"""
        if self.state in (MonitoringState.WAITING, MonitoringState.IN_PROGRESS):
            self._prev_state = self.state
            self.state = MonitoringState.PAUSED

    def resume(self):
        """恢复验证"""
        if self.state == MonitoringState.PAUSED:
            self.state = getattr(self, '_prev_state', MonitoringState.WAITING)

    def get_progress(self) -> Dict:
        """获取当前进度"""
        completed = sum(1 for s in self.step_status.values() if s == StepStatus.COMPLETED)
        total = len(self.steps)
        current_step = None
        current_step_name = None
        remaining_time = None

        if 0 <= self.current_step_idx < len(self.step_order):
            step_id = self.step_order[self.current_step_idx]
            current_step = step_id
            current_step_name = self.steps[step_id].step_name
            if step_id in self.step_start_time:
                elapsed = datetime.now().timestamp() - self.step_start_time[step_id]
                remaining_time = max(0, self.steps[step_id].max_duration - elapsed)

        return {
            "total_steps": total,
            "completed_steps": completed,
            "current_step": current_step,
            "current_step_name": current_step_name,
            "current_step_idx": self.current_step_idx,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "state": self.state.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "remaining_time": remaining_time,
        }

    def get_step_details(self) -> List[Dict]:
        """获取所有步骤的详细状态"""
        details = []
        for step_id in self.step_order:
            step = self.steps[step_id]
            status = self.step_status.get(step_id, StepStatus.PENDING)
            duration = self.step_durations.get(step_id, 0)
            details.append({
                "step_id": step.step_id,
                "step_name": step.step_name,
                "description": step.description,
                "status": status.value,
                "duration": duration,
                "expected_actions": step.expected_actions,
                "min_duration": step.min_duration,
                "max_duration": step.max_duration,
                "required": step.required,
            })
        return details

    def get_event_history(self) -> List[Dict]:
        """获取事件历史"""
        return [
            {
                "timestamp": e.timestamp,
                "result": e.result,
                "message": e.message,
                "step_id": e.step_id,
                "step_name": e.step_name,
                "detected_action": e.detected_action,
                "duration": e.duration
            }
            for e in self.event_history
        ]

    def reset(self):
        """重置验证器"""
        self.state = MonitoringState.IDLE
        self.current_step_idx = -1
        self.step_status = {}
        self.step_start_time = {}
        self.step_durations = {}
        self.step_match_counts = {}
        self.step_total_counts = {}
        self.step_mismatch_streak = 0
        self.event_history = []
        self.error_count = 0
        self.warning_count = 0
