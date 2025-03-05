import torch
import numpy as np
import os
from collections import OrderedDict
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze background prototypes file")
    parser.add_argument("--file", default="weights/initial/background/background_prototypes.vits14.pth",
                        help="Path to background prototypes file")
    return parser.parse_args()

def analyze_bg_prototypes(file_path):
    print(f"分析背景原型文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return
    
    # 加载PyTorch文件
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        print(f"成功加载文件.")
    except Exception as e:
        print(f"加载文件出错: {e}")
        return
    
    # 分析数据结构
    if isinstance(data, dict):
        print("文件包含一个字典，具有以下键:")
        for key in data.keys():
            print(f"  - {key}")
        
        # 查看是否有类别信息
        if 'label_names' in data:
            print("\n背景类别:")
            for i, label in enumerate(data['label_names']):
                print(f"  {i}: {label}")
        else:
            print("\n文件中没有显式的标签名称.")
        
        # 查看原型数据
        if 'prototypes' in data:
            prototypes = data['prototypes']
            print(f"\n原型形状: {prototypes.shape}")
            print(f"原型数据类型: {prototypes.dtype}")
            
            # 分析原型数据
            if len(prototypes.shape) == 2:
                num_classes, feat_dim = prototypes.shape
                print(f"背景类别/原型数量: {num_classes}")
                print(f"特征维度: {feat_dim}")
            elif len(prototypes.shape) == 3:
                num_classes, num_scales, feat_dim = prototypes.shape
                print(f"背景类别数量: {num_classes}")
                print(f"尺度数量: {num_scales}")
                print(f"特征维度: {feat_dim}")
            
            # 显示一些统计信息
            print(f"原型平均范数: {torch.norm(prototypes, dim=-1).mean().item():.4f}")
            print(f"最小值: {prototypes.min().item():.4f}")
            print(f"最大值: {prototypes.max().item():.4f}")
            print(f"平均值: {prototypes.mean().item():.4f}")
            print(f"标准差: {prototypes.std().item():.4f}")
            
            # 尝试检测原型是否已经过归一化
            norms = torch.norm(prototypes.reshape(-1, feat_dim), dim=1)
            is_normalized = torch.allclose(norms, torch.ones_like(norms), rtol=1e-2)
            print(f"原型是否归一化: {'是' if is_normalized else '否'}")
            
            # 检查原型的聚类情况
            if num_classes > 1:
                # 计算原型间的余弦相似度
                if len(prototypes.shape) == 3:
                    flat_protos = prototypes.mean(dim=1)  # 平均所有尺度
                else:
                    flat_protos = prototypes
                
                flat_protos_norm = flat_protos / torch.norm(flat_protos, dim=1, keepdim=True)
                cosine_sim = torch.mm(flat_protos_norm, flat_protos_norm.t())
                
                # 不考虑自身相似度
                mask = torch.ones_like(cosine_sim) - torch.eye(num_classes)
                masked_sim = cosine_sim * mask
                
                mean_sim = masked_sim.sum() / (num_classes * (num_classes - 1))
                max_sim = masked_sim.max()
                
                print(f"原型间平均余弦相似度: {mean_sim.item():.4f}")
                print(f"原型间最大余弦相似度: {max_sim.item():.4f}")
                
                # 提取最相似的两个原型
                max_idx = masked_sim.argmax().item()
                i, j = max_idx // num_classes, max_idx % num_classes
                print(f"最相似的原型对: {i} 和 {j}, 相似度: {masked_sim[i, j].item():.4f}")
    elif isinstance(data, torch.Tensor):
        print("文件包含直接张量 (非字典).")
        print(f"张量形状: {data.shape}")
        print(f"张量数据类型: {data.dtype}")
        
        # 分析张量结构
        if len(data.shape) == 2:
            num_prototypes, feat_dim = data.shape
            print(f"背景原型数量: {num_prototypes}")
            print(f"特征维度: {feat_dim}")
        elif len(data.shape) == 3:
            num_prototypes, num_scales, feat_dim = data.shape
            print(f"背景原型数量: {num_prototypes}")
            print(f"尺度数量: {num_scales}")
            print(f"特征维度: {feat_dim}")
        
        # 显示一些统计信息
        print(f"原型平均范数: {torch.norm(data, dim=-1).mean().item():.4f}")
        print(f"最小值: {data.min().item():.4f}")
        print(f"最大值: {data.max().item():.4f}")
        print(f"平均值: {data.mean().item():.4f}")
        print(f"标准差: {data.std().item():.4f}")
        
        # 尝试检测原型是否已经过归一化
        if len(data.shape) >= 2:
            last_dim = len(data.shape) - 1
            norms = torch.norm(data.reshape(-1, data.shape[last_dim]), dim=1)
            is_normalized = torch.allclose(norms, torch.ones_like(norms), rtol=1e-2)
            print(f"原型是否归一化: {'是' if is_normalized else '否'}")
    else:
        print(f"意外的数据类型: {type(data)}")
        print("无法进一步分析.")

if __name__ == "__main__":
    args = parse_args()
    analyze_bg_prototypes(args.file)