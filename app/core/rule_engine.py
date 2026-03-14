"""
@file rule_engine.py
@brief 可配置规则引擎 - 支持JSON配置文件定义操作流程和规则
@author AI Assistant
@date 2026-03-10
"""

import json
import os
import shutil
from typing import Dict, List, Optional, Tuple
from datetime import datetime
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

    PROCESSES_DIR = os.path.join("data", "processes")
    TEMPLATES_DIR = os.path.join("data", "templates")

    def __init__(self):
        """初始化规则引擎"""
        self.config = {}
        self.current_process_id = None
        os.makedirs(self.PROCESSES_DIR, exist_ok=True)
        os.makedirs(self.TEMPLATES_DIR, exist_ok=True)

    def create_process(self, name: str, description: str = "",
                       version: str = "1.0", category: str = "general") -> Dict:
        """
        @brief 创建新流程
        @return: 新创建的流程配置字典
        """
        process_id = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now().isoformat()
        config = {
            "process_id": process_id,
            "name": name,
            "version": version,
            "description": description,
            "category": category,
            "created_at": now,
            "updated_at": now,
            "steps": [],
            "constraints": {
                "max_total_time": 300.0,
                "allow_skip_steps": [],
                "allow_reverse": False,
                "strict_order": True,
            },
            "alarms": {
                "order_error": {"enabled": True, "level": "ERROR", "message": "操作顺序错误"},
                "timeout": {"enabled": True, "level": "WARNING", "message": "步骤执行超时"},
                "missing_step": {"enabled": True, "level": "ERROR", "message": "遗漏关键步骤"},
                "action_mismatch": {"enabled": True, "level": "WARNING", "message": "动作不匹配"},
            }
        }
        self.config = config
        self.current_process_id = process_id
        self.save_process()
        return config

    def load_process(self, process_id: str) -> Optional[Dict]:
        """加载流程配置"""
        filepath = os.path.join(self.PROCESSES_DIR, f"{process_id}.json")
        if not os.path.exists(filepath):
            print(f"流程配置不存在: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.current_process_id = process_id
                return self.config
        except Exception as e:
            print(f"加载流程配置失败: {e}")
            return None

    def save_process(self, config: Dict = None) -> bool:
        """保存流程配置"""
        if config:
            self.config = config
        if not self.config:
            print("没有可保存的配置")
            return False

        process_id = self.config.get("process_id", self.current_process_id)
        if not process_id:
            print("没有流程ID")
            return False

        self.config["updated_at"] = datetime.now().isoformat()
        filepath = os.path.join(self.PROCESSES_DIR, f"{process_id}.json")
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存流程配置失败: {e}")
            return False

    def delete_process(self, process_id: str) -> bool:
        """删除流程"""
        filepath = os.path.join(self.PROCESSES_DIR, f"{process_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            # 也删除模板目录
            template_dir = os.path.join(self.TEMPLATES_DIR, process_id)
            if os.path.exists(template_dir):
                shutil.rmtree(template_dir)
            if self.current_process_id == process_id:
                self.config = {}
                self.current_process_id = None
            return True
        except Exception as e:
            print(f"删除流程失败: {e}")
            return False

    def copy_process(self, process_id: str, new_name: str = None) -> Optional[Dict]:
        """复制流程"""
        original = self.load_process(process_id)
        if not original:
            return None
        new_config = json.loads(json.dumps(original))
        new_config["process_id"] = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_config["name"] = new_name or f"{original['name']} (副本)"
        new_config["created_at"] = datetime.now().isoformat()
        new_config["updated_at"] = datetime.now().isoformat()
        self.config = new_config
        self.current_process_id = new_config["process_id"]
        self.save_process()
        return new_config

    def list_processes(self) -> List[Dict]:
        """列出所有流程（摘要信息）"""
        processes = []
        if not os.path.exists(self.PROCESSES_DIR):
            return processes
        for filename in sorted(os.listdir(self.PROCESSES_DIR)):
            if filename.endswith('.json'):
                filepath = os.path.join(self.PROCESSES_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        processes.append({
                            "process_id": config.get("process_id", ""),
                            "name": config.get("name", ""),
                            "version": config.get("version", ""),
                            "description": config.get("description", ""),
                            "category": config.get("category", ""),
                            "step_count": len(config.get("steps", [])),
                            "created_at": config.get("created_at", ""),
                            "updated_at": config.get("updated_at", ""),
                        })
                except Exception:
                    continue
        return processes

    def add_step(self, step_data: Dict) -> bool:
        """添加步骤到当前流程"""
        if not self.config:
            return False
        steps = self.config.setdefault("steps", [])
        # 自动分配step_id
        if "step_id" not in step_data:
            step_data["step_id"] = len(steps) + 1
        steps.append(step_data)
        return self.save_process()

    def update_step(self, step_id: int, step_data: Dict) -> bool:
        """更新步骤"""
        if not self.config:
            return False
        steps = self.config.get("steps", [])
        for i, step in enumerate(steps):
            if step.get("step_id") == step_id:
                step_data["step_id"] = step_id
                steps[i] = step_data
                return self.save_process()
        return False

    def remove_step(self, step_id: int) -> bool:
        """删除步骤"""
        if not self.config:
            return False
        steps = self.config.get("steps", [])
        self.config["steps"] = [s for s in steps if s.get("step_id") != step_id]
        # 重新编号
        for i, step in enumerate(self.config["steps"]):
            step["step_id"] = i + 1
        return self.save_process()

    def reorder_steps(self, step_ids: List[int]) -> bool:
        """重排步骤顺序"""
        if not self.config:
            return False
        steps = self.config.get("steps", [])
        step_map = {s["step_id"]: s for s in steps}
        new_steps = []
        for i, sid in enumerate(step_ids):
            if sid in step_map:
                step = step_map[sid]
                step["step_id"] = i + 1
                new_steps.append(step)
        self.config["steps"] = new_steps
        return self.save_process()

    def export_process(self, process_id: str, filepath: str) -> bool:
        """导出流程配置"""
        config = self.load_process(process_id)
        if not config:
            return False
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def import_process(self, filepath: str) -> Optional[Dict]:
        """导入流程配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 给新ID避免冲突
            config["process_id"] = f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config["created_at"] = datetime.now().isoformat()
            config["updated_at"] = datetime.now().isoformat()
            self.config = config
            self.current_process_id = config["process_id"]
            self.save_process()
            return config
        except Exception as e:
            print(f"导入失败: {e}")
            return None

    def get_process_config(self) -> Dict:
        """获取当前流程配置"""
        return self.config

    def get_steps_config(self) -> List[Dict]:
        """获取步骤配置列表"""
        return self.config.get("steps", [])

    def get_alarm_config(self) -> Dict:
        """获取告警配置"""
        return self.config.get("alarms", {})

    def get_constraints(self) -> Dict:
        """获取约束配置"""
        return self.config.get("constraints", {})

    def update_constraints(self, constraints: Dict) -> bool:
        """更新约束配置"""
        if not self.config:
            return False
        self.config["constraints"] = constraints
        return self.save_process()

    def update_alarms(self, alarms: Dict) -> bool:
        """更新告警配置"""
        if not self.config:
            return False
        self.config["alarms"] = alarms
        return self.save_process()

    def validate_config(self) -> Tuple:
        """验证配置有效性，返回 (是否有效, 错误列表)"""
        errors = []
        if not self.config:
            errors.append("配置为空")
            return False, errors
        if not self.config.get("name"):
            errors.append("缺少流程名称")
        steps = self.config.get("steps", [])
        if not steps:
            errors.append("没有定义任何步骤")
        for step in steps:
            if not step.get("step_name") and not step.get("name"):
                errors.append(f"步骤 {step.get('step_id', '?')} 缺少名称")
            if not step.get("expected_actions"):
                errors.append(f"步骤 {step.get('step_name', '?')} 没有设置预期动作")
        return len(errors) == 0, errors
