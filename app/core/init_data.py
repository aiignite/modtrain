"""
数据目录初始化脚本
用于创建PCB SMT焊接质量检测的标准数据目录结构
"""
import os
import yaml


def create_default_data_structure(base_dir="data"):
    """
    创建默认数据目录结构
    
    Args:
        base_dir: 基础数据目录
        
    Returns:
        bool: 创建是否成功
    """
    try:
        # 创建主目录
        os.makedirs(base_dir, exist_ok=True)
        
        # 创建训练、验证、测试目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)
        
        # 创建示例数据配置文件
        data_config = {
            "path": os.path.abspath(base_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 6,
            "names": [
                "solder_bridge",
                "missing_solder", 
                "insufficient_solder",
                "excess_solder",
                "tombstoning",
                "misalignment"
            ]
        }
        
        config_path = os.path.join(base_dir, "data.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)
        
        # 创建README文件
        readme_content = f"""# PCB SMT焊接质量检测数据集

## 目录结构
```
{base_dir}/
├── train/
│   ├── images/  # 训练图片
│   └── labels/   # 训练标签 (YOLO格式)
├── val/
│   ├── images/  # 验证图片
│   └── labels/   # 验证标签
├── test/
│   ├── images/  # 测试图片
│   └── labels/   # 测试标签
└── data.yaml    # 数据配置文件
```

## 使用说明
1. 将训练图片放入 `train/images/` 目录
2. 将对应的YOLO格式标签文件放入 `train/labels/` 目录
3. 同样方式准备验证集和测试集
4. 标签文件名应与图片文件名相同（扩展名不同）

## YOLO标签格式
每个标签文件包含多行，每行格式：
```
<class_id> <x_center> <y_center> <width> <height>
```
其中所有坐标值都是归一化的（0-1之间）

## 类别定义
0: solder_bridge (桥接缺陷)
1: missing_solder (漏焊缺陷)
2: insufficient_solder (焊锡不足)
3: excess_solder (焊锡过多)
4: tombstoning (立碑缺陷)
5: misalignment (偏移缺陷)
"""
        
        readme_path = os.path.join(base_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"数据目录结构创建成功: {base_dir}")
        print(f"配置文件: {config_path}")
        print(f"说明文件: {readme_path}")
        
        return True
    except Exception as e:
        print(f"创建数据目录结构失败: {e}")
        return False


if __name__ == "__main__":
    print("初始化PCB SMT焊接质量检测数据目录结构...")
    create_default_data_structure("data")