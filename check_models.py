import os

models = {
    'yolov8n.pt': 'yolo/yolov8n.pt',
    'yolo26n.pt': 'yolo/yolo26n.pt',
    'yolo26m.pt': 'yolo/yolo26m.pt'
}

print("模型文件大小对比:")
print("-" * 50)
for name, path in models.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"{name:15s} : {size_mb:8.2f} MB")
    else:
        print(f"{name:15s} : 文件不存在")
print("-" * 50)

print("\n检测时间没有明显变化的原因分析:")
print("1. yolov8n 和 yolo26n 都是 'nano' 级别模型，参数量相近")
print("2. 在 CPU 上运行时，瓶颈主要在计算能力而非模型大小")
print("3. 如果使用 GPU，两个模型都能充分利用 GPU 并行计算")
print("4. 检测时间还受以下因素影响:")
print("   - 输入图像分辨率")
print("   - 检测到的目标数量")
print("   - 后处理（NMS）时间")
print("   - 系统资源占用情况")
print("\n建议:")
print("- 使用更大差异的模型对比（如 yolov8n vs yolov8m）")
print("- 在相同条件下多次测试取平均值")
print("- 使用 GPU 加速会看到更明显的差异")