import os
import logging
from logging.handlers import RotatingFileHandler


class LogManager:
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level

        # 确保日志目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def setup_logger(self, logger_name, log_file):
        """设置日志记录器"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)

        # 文件处理器
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)

        return logger

    def log_training(self, message, level=logging.INFO):
        """记录训练日志"""
        logger = self.setup_logger("training", "training.log")
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.CRITICAL:
            logger.critical(message)

    def log_monitoring(self, message, level=logging.INFO):
        """记录监控日志"""
        logger = self.setup_logger("monitoring", "monitoring.log")
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.CRITICAL:
            logger.critical(message)

    def query_logs(self, start_time, end_time, log_level=None):
        """查询日志"""
        # 这里可以添加日志查询逻辑
        pass