from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QGroupBox, QGridLayout, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTextEdit, QFileDialog, QCheckBox,
                             QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os

from app.core.model_trainer import ModelTrainer
from app.core.data_manager import DataManager


class ModelTrainingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.trainer = ModelTrainer()
        self.data_manager = DataManager()
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 模型选择区
        model_group = QGroupBox("模型选择")
        model_layout = QGridLayout()

        # 模型文件选择（优先）
        model_layout.addWidget(QLabel("模型文件:"), 0, 0)
        self.model_path_label = QLabel("未选择，使用默认下载")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label, 0, 1)
        self.select_model_btn = QPushButton("选择模型文件")
        self.select_model_btn.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.select_model_btn, 1, 0, 1, 2)

        # 或使用默认模型
        model_layout.addWidget(QLabel("或使用默认模型:"), 2, 0)
        self.use_default_model_check = QCheckBox("使用默认模型")
        self.use_default_model_check.setChecked(True)
        model_layout.addWidget(self.use_default_model_check, 2, 1)

        # YOLO版本
        model_layout.addWidget(QLabel("YOLO版本:"), 3, 0)
        self.yolo_version = QComboBox()
        self.yolo_version.addItems(["YOLOv3", "YOLOv5", "YOLOv8"])
        self.yolo_version.setCurrentIndex(2)
        model_layout.addWidget(self.yolo_version, 3, 1)

        # 模型大小
        model_layout.addWidget(QLabel("模型大小:"), 4, 0)
        self.model_size = QComboBox()
        self.model_size.addItems(["n", "s", "m", "l", "x"])
        model_layout.addWidget(self.model_size, 4, 1)

        # 数据集选择
        model_layout.addWidget(QLabel("数据集:"), 5, 0)
        self.dataset_path_label = QLabel("未选择数据集")
        model_layout.addWidget(self.dataset_path_label, 5, 1)
        self.select_dataset_btn = QPushButton("选择数据集")
        self.select_dataset_btn.clicked.connect(self.select_dataset)
        model_layout.addWidget(self.select_dataset_btn, 6, 0, 1, 2)

        model_group.setLayout(model_layout)

        # 参数配置区
        param_group = QGroupBox("参数配置")
        param_layout = QGridLayout()

        # 训练轮次
        param_layout.addWidget(QLabel("训练轮次:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        param_layout.addWidget(self.epochs, 0, 1)

        # 批次大小
        param_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(4)
        param_layout.addWidget(self.batch_size, 1, 1)

        # 学习率
        param_layout.addWidget(QLabel("学习率:"), 2, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.00001, 0.1)
        self.learning_rate.setValue(0.01)
        self.learning_rate.setDecimals(5)
        param_layout.addWidget(self.learning_rate, 2, 1)

        # 图像大小
        param_layout.addWidget(QLabel("图像大小:"), 3, 0)
        self.img_size = QSpinBox()
        self.img_size.setRange(320, 1280)
        self.img_size.setSingleStep(32)
        self.img_size.setValue(640)
        param_layout.addWidget(self.img_size, 3, 1)

        param_group.setLayout(param_layout)

        # 训练监控区
        monitor_group = QGroupBox("训练监控与结果")
        monitor_layout = QHBoxLayout()
        
        # 左侧：训练日志
        log_layout = QVBoxLayout()
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setPlainText("训练日志将显示在这里...")
        log_layout.addWidget(self.training_log)
        
        # 右侧：训练结果预览
        result_layout = QVBoxLayout()
        
        # 训练进度
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("训练进度: %p% (%v/%m 轮)")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        self.metrics_label = QLabel("损失值: N/A | mAP: N/A")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.metrics_label)
        progress_group.setLayout(progress_layout)
        result_layout.addWidget(progress_group)
        
        # 训练结果
        result_info_group = QGroupBox("训练结果")
        result_info_layout = QVBoxLayout()
        self.result_path_label = QLabel("模型保存路径: 未开始训练")
        self.result_path_label.setWordWrap(True)
        result_info_layout.addWidget(self.result_path_label)
        self.best_model_label = QLabel("最佳模型: N/A")
        result_info_layout.addWidget(self.best_model_label)
        self.train_time_label = QLabel("训练时间: N/A")
        result_info_layout.addWidget(self.train_time_label)
        result_info_group.setLayout(result_info_layout)
        result_layout.addWidget(result_info_group)
        
        # 评估指标
        metrics_group = QGroupBox("评估指标")
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("mAP@0.5:"), 0, 0)
        self.map50_label = QLabel("0.0")
        metrics_layout.addWidget(self.map50_label, 0, 1)
        metrics_layout.addWidget(QLabel("mAP@0.5:0.95:"), 1, 0)
        self.map50_95_label = QLabel("0.0")
        metrics_layout.addWidget(self.map50_95_label, 1, 1)
        metrics_layout.addWidget(QLabel("精确率:"), 2, 0)
        self.precision_label = QLabel("0.0")
        metrics_layout.addWidget(self.precision_label, 2, 1)
        metrics_layout.addWidget(QLabel("召回率:"), 3, 0)
        self.recall_label = QLabel("0.0")
        metrics_layout.addWidget(self.recall_label, 3, 1)
        metrics_group.setLayout(metrics_layout)
        result_layout.addWidget(metrics_group)
        
        # 查看结果按钮
        self.view_result_btn = QPushButton("查看详细结果")
        self.view_result_btn.setEnabled(False)
        self.view_result_btn.clicked.connect(self.view_training_result)
        result_layout.addWidget(self.view_result_btn)
        
        # 添加到主监控布局
        monitor_layout.addLayout(log_layout, 1)
        monitor_layout.addLayout(result_layout, 1)
        monitor_group.setLayout(monitor_layout)

        # 训练控制区
        control_group = QGroupBox("训练控制")
        control_layout = QHBoxLayout()
        self.start_train_btn = QPushButton("开始训练")
        self.start_train_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.start_train_btn.clicked.connect(self.start_training)
        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.clicked.connect(self.stop_training)
        control_layout.addWidget(self.start_train_btn)
        control_layout.addWidget(self.stop_train_btn)
        control_group.setLayout(control_layout)

        # 布局组合
        top_layout = QHBoxLayout()
        top_layout.addWidget(model_group, 1)
        top_layout.addWidget(param_group, 1)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(monitor_group)
        main_layout.addWidget(control_group)

    def select_model_file(self):
        """选择本地模型文件"""
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pt *.pth *.onnx)", options=options)
        if file:
            self.custom_model_path = file
            self.model_path_label.setText(file)
            self.use_default_model_check.setChecked(False)
            self.training_log.append(f"已选择模型文件: {file}")

    def select_dataset(self):
        """选择数据集目录"""
        options = QFileDialog.Options()
        dataset_dir = QFileDialog.getExistingDirectory(self, "选择数据集目录", "", options=options)
        if dataset_dir:
            self.dataset_path = dataset_dir
            self.dataset_path_label.setText(dataset_dir)
            self.training_log.append(f"已选择数据集: {dataset_dir}")
            
            # 检查是否有data.yaml
            data_yaml = os.path.join(dataset_dir, "data.yaml")
            if os.path.exists(data_yaml):
                self.training_log.append(f"找到数据配置文件: {data_yaml}")

    def start_training(self):
        """开始训练逻辑"""
        if not hasattr(self, 'dataset_path') or not self.dataset_path:
            self.training_log.append("错误: 请先选择数据集!")
            return
        
        # 检查data.yaml是否存在
        data_yaml = os.path.join(self.dataset_path, "data.yaml")
        if not os.path.exists(data_yaml):
            self.training_log.append(f"错误: 数据集目录下未找到data.yaml!")
            self.training_log.append(f"请先在数据管理模块点击'准备训练数据'")
            return

        # 获取自定义模型路径
        custom_model_path = None
        if not self.use_default_model_check.isChecked() and hasattr(self, 'custom_model_path'):
            custom_model_path = self.custom_model_path

        # 获取模型参数
        yolo_version = self.yolo_version.currentText()
        model_size = self.model_size.currentText()
        
        # 转换YOLO版本为模型类型
        model_type_map = {
            "YOLOv3": "yolov3",
            "YOLOv5": "yolov5", 
            "YOLOv8": "yolov8"
        }
        model_type = model_type_map.get(yolo_version, "yolov8")
        
        # 获取训练参数
        hyperparams = {
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "learning_rate": self.learning_rate.value(),
            "img_size": self.img_size.value()
        }
        
        # 创建训练线程
        self.training_thread = TrainingThread(
            self.trainer, 
            model_type, 
            model_size, 
            self.dataset_path,
            hyperparams,
            custom_model_path
        )
        self.training_thread.log_signal.connect(self.training_log.append)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.start()
        
        # 更新按钮状态
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        
        # 设置进度条最大值
        total_epochs = self.epochs.value()
        self.progress_bar.setMaximum(total_epochs)
        self.progress_bar.setValue(0)
        
        if custom_model_path:
            self.training_log.append(f"使用自定义模型: {custom_model_path}")
        self.training_log.append(f"开始训练 {yolo_version}-{model_size} 模型...")
        self.training_log.append(f"数据配置: {data_yaml}")

    def update_progress(self, epoch, loss, map50):
        """更新训练进度"""
        self.progress_bar.setValue(epoch)
        self.metrics_label.setText(f"损失值: {loss:.4f} | mAP@0.5: {map50:.4f}")

    def stop_training(self):
        """停止训练逻辑"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
        self.training_log.append("停止训练...")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def on_training_finished(self, success, message):
        """训练完成回调"""
        if success:
            self.training_log.append(f"训练完成! {message}")
            self.result_path_label.setText(f"模型保存路径: {message.split(': ')[1]}")
            self.best_model_label.setText(f"最佳模型: best.pt")
            self.view_result_btn.setEnabled(True)
        else:
            self.training_log.append(f"训练失败: {message}")
            self.result_path_label.setText(f"模型保存路径: 训练失败")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
    
    def view_training_result(self):
        """查看并分析训练结果"""
        result_path = self.result_path_label.text().replace("模型保存路径: ", "")
        
        if not result_path or result_path == "未开始训练" or result_path == "训练失败":
            # 尝试查找最新的训练结果
            runs_dir = os.path.join(os.getcwd(), "runs", "train")
            if os.path.exists(runs_dir):
                exp_dirs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))],
                                 key=lambda x: os.path.getmtime(os.path.join(runs_dir, x)), reverse=True)
                if exp_dirs:
                    result_path = os.path.join(runs_dir, exp_dirs[0])
        
        if result_path and os.path.exists(result_path):
            # 分析训练结果
            analysis = self.analyze_training_result(result_path)
            self.training_log.append("\n" + "="*50)
            self.training_log.append("训练结果分析:")
            self.training_log.append("="*50)
            for line in analysis:
                self.training_log.append(line)
            
            # 显示结果图片
            results_img = os.path.join(result_path, "results.png")
            if os.path.exists(results_img):
                self.show_result_image(results_img)
            
            # 打开结果目录
            os.startfile(result_path)
        else:
            self.training_log.append("错误: 没有训练结果可以查看")
    
    def analyze_training_result(self, result_dir):
        """分析训练结果"""
        analysis = []
        
        # 读取results.csv
        csv_path = os.path.join(result_dir, "results.csv")
        if os.path.exists(csv_path):
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if rows:
                # 获取最后一行的指标
                last = rows[-1]
                analysis.append(f"训练轮次: {len(rows)}")
                analysis.append(f"总训练时间: {sum(float(r['time']) for r in rows):.2f}s")
                
                # 最终指标
                map50 = float(last.get('metrics/mAP50(B)', 0))
                map50_95 = float(last.get('metrics/mAP50-95(B)', 0))
                precision = float(last.get('metrics/precision(B)', 0))
                recall = float(last.get('metrics/recall(B)', 0))
                
                analysis.append(f"mAP@0.5: {map50:.4f}")
                analysis.append(f"mAP@0.5:0.95: {map50_95:.4f}")
                analysis.append(f"精确率(Precision): {precision:.4f}")
                analysis.append(f"召回率(Recall): {recall:.4f}")
                
                # 更新UI显示
                self.map50_label.setText(f"{map50:.4f}")
                self.map50_95_label.setText(f"{map50_95:.4f}")
                self.precision_label.setText(f"{precision:.4f}")
                self.recall_label.setText(f"{recall:.4f}")
                
                # 评估训练效果
                analysis.append("\n效果评估:")
                if map50 > 0.9:
                    analysis.append("  ✓ 优秀 - 模型表现很好!")
                elif map50 > 0.7:
                    analysis.append("  ✓ 良好 - 可以继续训练提升")
                elif map50 > 0.5:
                    analysis.append("  ⚠ 一般 - 建议增加数据或调整参数")
                elif map50 > 0:
                    analysis.append("  ✗ 较差 - 需要检查数据和参数")
                else:
                    analysis.append("  ✗ 未收敛 - 检查数据标注是否正确!")
        
        # 检查权重文件
        weights_dir = os.path.join(result_dir, "weights")
        if os.path.exists(weights_dir):
            best_model = os.path.join(weights_dir, "best.pt")
            if os.path.exists(best_model):
                size_mb = os.path.getsize(best_model) / (1024 * 1024)
                analysis.append(f"\n最佳模型: {best_model}")
                analysis.append(f"模型大小: {size_mb:.2f} MB")
                self.best_model_label.setText(f"best.pt ({size_mb:.1f}MB)")
        
        return analysis
    
    def show_result_image(self, img_path):
        """在预览区域显示结果图片"""
        # 这里可以在UI上添加一个QLabel来显示结果图片
        pass


class TrainingThread(QThread):
    """训练线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int, float, float)
    
    def __init__(self, trainer, model_type, model_size, dataset_path, hyperparams, custom_model_path=None):
        super().__init__()
        self.trainer = trainer
        self.model_type = model_type
        self.model_size = model_size
        self.dataset_path = dataset_path
        self.hyperparams = hyperparams
        self.custom_model_path = custom_model_path
        self._is_running = True
        self.model = None
        self.total_epochs = hyperparams['epochs']
        self.current_epoch = 0
    
    def run(self):
        """运行训练"""
        try:
            from ultralytics import YOLO
            
            # 使用自定义模型或默认模型
            if self.custom_model_path and os.path.exists(self.custom_model_path):
                self.log_signal.emit(f"加载本地模型: {self.custom_model_path}")
                self.model = YOLO(self.custom_model_path)
            else:
                model_name = f"{self.model_type}{self.model_size}.pt"
                self.log_signal.emit(f"使用默认模型: {model_name}")
                self.model = YOLO(model_name)
            
            # 使用已存在的data.yaml
            data_yaml = os.path.join(self.dataset_path, "data.yaml")
            if not os.path.exists(data_yaml):
                self.finished_signal.emit(False, f"未找到数据配置文件: {data_yaml}")
                return
            
            self.log_signal.emit(f"数据配置: {data_yaml}")
            
            self.log_signal.emit(f"开始训练: epochs={self.hyperparams['epochs']}, "
                               f"batch_size={self.hyperparams['batch_size']}")

            # 创建回调来监控训练进度
            def on_train_epoch_end(trainer):
                epoch = trainer.epoch + 1
                loss = trainer.metrics.get('train/box_loss', 0)
                map50 = trainer.metrics.get('metrics/mAP50(B)', 0)
                self.progress_signal.emit(epoch, float(loss), float(map50))
            
            # 注册回调
            self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
            
            # 直接使用YOLO的train方法
            results = self.model.train(
                data=data_yaml,
                epochs=self.hyperparams['epochs'],
                batch=self.hyperparams['batch_size'],
                imgsz=self.hyperparams['img_size'],
                lr0=self.hyperparams['learning_rate'],
                project=os.path.join(os.getcwd(), "runs", "train"),
                name=f"exp",
                exist_ok=True,
                verbose=True
            )
            
            result_path = str(results.save_dir)
            
            # 训练完成，设置进度条为100%
            self.progress_signal.emit(self.total_epochs, 0, 0)
            
            if result_path:
                self.finished_signal.emit(True, f"训练结果保存在: {result_path}")
            else:
                self.finished_signal.emit(False, "训练过程出错")
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.finished_signal.emit(False, f"训练异常: {str(e)}\n{error_msg}")
    
    def stop(self):
        """停止训练"""
        self._is_running = False
        # 注意：YOLO训练过程很难中断，这里只能终止线程
        self.terminate()