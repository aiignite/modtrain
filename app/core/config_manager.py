import json
import os


class ConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.default_config = os.path.join(config_dir, "default_config.json")
        self.user_config = os.path.join(config_dir, "user_config.json")

        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

        # 如果默认配置不存在，创建默认配置
        if not os.path.exists(self.default_config):
            self.create_default_config()

    def create_default_config(self):
        """创建默认配置"""
        default_config = {
            "camera": {
                "default_id": 0,
                "resolution": {
                    "width": 640,
                    "height": 480
                }
            },
            "model": {
                "default_path": "",
                "conf_threshold": 0.5,
                "iou_threshold": 0.45
            },
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "learning_rate": 0.001,
                "img_size": 640
            },
            "data": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1
            },
            "defects": [
                "solder_bridge",
                "missing_solder",
                "insufficient_solder",
                "excess_solder",
                "tombstoning",
                "misalignment"
            ]
        }

        with open(self.default_config, 'w') as f:
            json.dump(default_config, f, indent=2)

    def load_config(self, config_file=None):
        """加载配置"""
        if config_file is None:
            # 优先加载用户配置
            if os.path.exists(self.user_config):
                config_file = self.user_config
            else:
                config_file = self.default_config

        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"加载配置失败: {e}")
            return {}

    def save_config(self, config, config_file=None):
        """保存配置"""
        if config_file is None:
            config_file = self.user_config

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def get_default_config(self, config_type):
        """获取默认配置"""
        config = self.load_config()
        return config.get(config_type, {})