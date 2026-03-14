# main.py
import sys
import os
import traceback

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

print("=== 启动应用 ===")
print(f"Python 版本: {sys.version}")
print(f"当前目录: {os.getcwd()}")
print(f"Python 路径: {sys.path[:5]}")

try:
    print("\n=== 导入 PyQt5 ===")
    from PyQt5.QtWidgets import QApplication
    print("✓ PyQt5 导入成功")
    
    print("\n=== 导入主窗口 ===")
    from ui.main_window import MainWindow  # 使用相对导入
    print("✓ MainWindow 导入成功")
    
    def main():
        print("\n=== 创建应用 ===")
        app = QApplication(sys.argv)
        print("✓ 应用创建成功")
        
        print("\n=== 创建主窗口 ===")
        window = MainWindow()
        print("✓ 主窗口创建成功")
        
        print("\n=== 显示窗口 ===")
        window.show()
        print("✓ 窗口显示成功")
        
        print("\n=== 进入事件循环 ===")
        sys.exit(app.exec_())
    
    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"\n✗ 启动失败: {e}")
    traceback.print_exc()
    sys.exit(1)