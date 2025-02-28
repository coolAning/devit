import argparse
import copy
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# VOC 类别
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']  # fmt: skip

# 基本类别（前15个）和新类别（后5个）
BASE_CLASSES = VOC_CLASSES[:15]
NOVEL_CLASSES = VOC_CLASSES[15:]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 20], help="Range of seeds"
    )
    parser.add_argument(
        "--voc-root", type=str, default="/mnt/d/datasets/VOCdevkit",
        help="Root directory of VOC dataset"
    )
    parser.add_argument(
        "--save-dir", type=str, default="/mnt/d/datasets/VOCdevkit/VOC/vocsplit",
        help="Directory to save split files"
    )
    args = parser.parse_args()
    return args

def generate_seeds(args):
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 收集每个类别的图像
    data_per_cat = {c: [] for c in VOC_CLASSES}
    
    # 遍历2007和2012数据集
    for year in ["2007", "2012"]:
        # 检查数据集是否存在
        year_path = os.path.join(args.voc_root, f"VOC{year}")
        if not os.path.exists(year_path):
            print(f"警告: 找不到{year_path}，跳过")
            continue
            
        # 加载trainval文件ID
        data_file = os.path.join(year_path, "ImageSets", "Main", "trainval.txt")
        if not os.path.exists(data_file):
            print(f"警告: 找不到{data_file}，跳过")
            continue
            
        print(f"加载数据集 VOC{year}...")
        with open(data_file) as f:
            fileids = f.read().strip().split('\n')
        
        # 解析XML标注，收集每个类别的图像
        print(f"解析VOC{year}标注...")
        for fileid in fileids:
            # 构建XML文件路径
            anno_file = os.path.join(year_path, "Annotations", f"{fileid}.xml")
            if not os.path.exists(anno_file):
                continue
                
            # 解析XML获取类别信息
            try:
                tree = ET.parse(anno_file)
                img_file = tree.find("filename").text
                img_path = os.path.join(year_path, "JPEGImages", img_file)
                
                # 收集图像中的所有类别
                classes_in_img = set()
                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    if cls in VOC_CLASSES:
                        classes_in_img.add(cls)
                
                # 添加图像到对应类别的列表中
                for cls in classes_in_img:
                    data_per_cat[cls].append(img_path)
            except Exception as e:
                print(f"解析文件{anno_file}时出错: {e}")
    
    # 打印每个类别的图像数量
    print("\n每个类别的图像数量:")
    for cls, images in data_per_cat.items():
        print(f"{cls}: {len(images)}张图片")
    
    # 创建不同shot设置的样本
    shots = [1, 2, 3, 5, 10]
    
    # 为每个种子生成数据
    for seed in range(args.seeds[0], args.seeds[1]):
        print(f"\n使用种子 {seed} 生成few-shot样本...")
        random.seed(seed)
        
        # 创建种子目录
        seed_dir = os.path.join(args.save_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # 分别处理基本类和新类
        # 1. 先生成基本类的完整数据集
        base_data = {c: data_per_cat[c] for c in BASE_CLASSES}
        for cls, images in base_data.items():
            filename = os.path.join(seed_dir, f"base_{cls}_train.txt")
            with open(filename, 'w') as f:
                f.write('\n'.join(images))
        
        # 2. 为新类生成few-shot样本
        novel_data = {c: {} for c in NOVEL_CLASSES}
        for cls in NOVEL_CLASSES:
            available_images = data_per_cat[cls]
            # 随机打乱确保独立采样
            random.shuffle(available_images)
            
            # 为每个shot设置创建样本
            for shot in shots:
                # 确保有足够的样本
                if len(available_images) < shot:
                    print(f"警告: 类别 {cls} 只有 {len(available_images)} 张图片，少于请求的 {shot} shot")
                    # 使用所有可用图像，重复采样来满足要求
                    selected = available_images
                    while len(selected) < shot:
                        selected.extend(random.sample(available_images, min(shot - len(selected), len(available_images))))
                else:
                    # 随机选择shot数量的样本
                    selected = random.sample(available_images, shot)
                
                # 保存到文件
                filename = os.path.join(seed_dir, f"novel_{shot}shot_{cls}_train.txt")
                with open(filename, 'w') as f:
                    f.write('\n'.join(selected))
                
                novel_data[cls][shot] = selected
        
        # 3. 创建合并数据集(基本类+新类)
        for shot in shots:
            combined_data = {}
            # 添加所有基本类
            for cls in BASE_CLASSES:
                combined_data[cls] = base_data[cls]
            
            # 添加新类（使用shot数量的样本）
            for cls in NOVEL_CLASSES:
                combined_data[cls] = novel_data[cls][shot]
            
            # 保存合并数据集
            combined_file = os.path.join(seed_dir, f"combined_{shot}shot_train.txt")
            with open(combined_file, 'w') as f:
                for cls, images in combined_data.items():
                    cls_line = f"# {cls}"
                    f.write(f"{cls_line}\n")
                    f.write('\n'.join(images))
                    f.write('\n\n')
    
    print(f"\n所有few-shot数据集已保存到 {args.save_dir}")

def main():
    args = parse_args()
    print(f"VOC Few-Shot数据准备工具")
    print(f"VOC数据集路径: {args.voc_root}")
    print(f"保存目录: {args.save_dir}")
    print(f"种子范围: {args.seeds[0]}-{args.seeds[1]-1}")
    print("\n基本类别:", BASE_CLASSES)
    print("新类别:", NOVEL_CLASSES)
    
    # 验证数据路径
    if not os.path.exists(args.voc_root):
        print(f"错误: VOC数据集路径 {args.voc_root} 不存在!")
        return False
    
    # 生成数据
    generate_seeds(args)
    return True

if __name__ == "__main__":
    main()