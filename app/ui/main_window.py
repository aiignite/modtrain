from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStatusBar)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt

from app.ui.data_management import DataManagementWidget
from app.ui.model_training import ModelTrainingWidget
from app.ui.model_evaluation import ModelEvaluationWidget
from app.ui.model_inference import ModelInferenceWidget
from app.ui.realtime_monitoring import RealtimeMonitoringWidget
from app.ui.operation_monitoring import OperationMonitoringWidget
from app.ui.process_config_ui import ProcessConfigWidget
from app.ui.offline_learning_ui import OfflineLearningWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("生产操作监控系统 - PCB SMT焊接质量检测")
        self.setGeometry(100, 100, 1500, 950)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
                border-bottom: 3px solid #2980b9;
            }
            QTabBar::tab:!selected {
                background-color: #ecf0f1;
                color: #2c3e50;
            }
            QTabBar::tab:hover {
                background-color: #d5dbdb;
            }
        """)

        # 创建各个功能模块
        self.data_management = DataManagementWidget()
        self.model_training = ModelTrainingWidget()
        self.model_evaluation = ModelEvaluationWidget()
        self.model_inference = ModelInferenceWidget()
        self.realtime_monitoring = RealtimeMonitoringWidget()
        self.operation_monitoring = OperationMonitoringWidget()
        self.process_config = ProcessConfigWidget()
        self.offline_learning = OfflineLearningWidget()

        # 添加标签页（新增操作监控和流程配置）
        self.tab_widget.addTab(self.data_management, "📁 数据管理")
        self.tab_widget.addTab(self.model_training, "🏋 模型训练")
        self.tab_widget.addTab(self.model_evaluation, "📊 模型评估")
        self.tab_widget.addTab(self.model_inference, "🔍 模型推理")
        self.tab_widget.addTab(self.realtime_monitoring, "📹 实时监控")
        self.tab_widget.addTab(self.operation_monitoring, "🎯 操作监控")
        self.tab_widget.addTab(self.process_config, "⚙️ 流程配置")
        self.tab_widget.addTab(self.offline_learning, "🧠 离线学习")

        # 流程配置变更时通知操作监控刷新
        self.process_config.process_changed.connect(
            self.operation_monitoring.refresh_process_list
        )

        # 添加到主布局
        main_layout.addWidget(self.tab_widget)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 | 生产操作监控系统 v2.0")