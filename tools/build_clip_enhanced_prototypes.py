import torch
import clip
from pathlib import Path
import argparse
import os
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Fuse CLIP text features with existing visual prototypes")
    parser.add_argument("--input-prototype", required=True, help="Input prototype file")
    parser.add_argument("--output-prototype", required=True, help="Output prototype file")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model variant")
    parser.add_argument("--alpha", type=float, default=0.7, help="Visual feature weight")
    parser.add_argument("--semantic-desc", default=None, help="Path to semantic descriptions")
    return parser.parse_args()

def load_clip_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    return model, device

def extract_clip_features(model, texts, device):
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features

def generate_class_templates(class_name):
    """为每个类别生成丰富的文本模板描述"""
    templates = [
        f"a photo of a {class_name}",
        f"a close-up photo of a {class_name}",
        f"a {class_name} in the wild",
        f"this is a {class_name}",
        f"a {class_name} in the scene",
        f"a bright photo of a {class_name}",
        f"a dark photo of a {class_name}",
        f"a photo of the {class_name}",
        f"a photo of many {class_name}s",
        f"a small {class_name}",
        f"a large {class_name}",
    ]
    return templates

def main():
    args = parse_args()
    
    # 加载CLIP模型
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, device = load_clip_model(args.clip_model)
    
    # 加载现有原型
    print(f"Loading prototypes from: {args.input_prototype}")
    proto_data = torch.load(args.input_prototype)
    
    visual_prototypes = proto_data['prototypes']
    class_names = proto_data['label_names']
    
    print(f"Found {len(class_names)} classes")
    
    # 提取CLIP文本特征
    text_features_list = []
    
    print("Extracting CLIP text features...")
    for class_name in tqdm(class_names):
        # 生成文本描述
        templates = generate_class_templates(class_name)
        
        # 提取特征
        text_features = extract_clip_features(clip_model, templates, device)
        
        # 平均多个模板的特征
        text_feature = text_features.mean(dim=0)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_features_list.append(text_feature.cpu())
    
    text_prototypes = torch.stack(text_features_list)
    
    # 处理多尺度原型
    is_multiscale = len(visual_prototypes.shape) == 3
    
    if is_multiscale:
        # 多尺度原型 [num_classes, num_scales, dim]
        visual_prototypes_mean = visual_prototypes.mean(dim=1)
    else:
        # 单尺度原型 [num_classes, dim]
        visual_prototypes_mean = visual_prototypes
    
    # 投影文本特征以匹配视觉特征维度
    visual_dim = visual_prototypes_mean.shape[-1]
    text_dim = text_prototypes.shape[-1]
    
    print(f"Visual feature dimension: {visual_dim}")
    print(f"Text feature dimension: {text_dim}")
    
    projection = torch.nn.Linear(text_dim, visual_dim).to(device)
    # 初始化为尽可能保持内容的映射
    torch.nn.init.eye_(projection.weight[:min(text_dim, visual_dim), :min(text_dim, visual_dim)])
    print(f"Text prototypes dtype: {text_prototypes.dtype}")
    print(f"Projection weight dtype: {projection.weight.dtype}")
    # 将所有内容转换为相同的数据类型(float32)
    text_prototypes = text_prototypes.to(torch.float32)
    projection = projection.to(torch.float32)

    with torch.no_grad():
        projected_text_features = projection(text_prototypes.to(device)).cpu()
    
    # 归一化
    visual_prototypes_mean = visual_prototypes_mean / visual_prototypes_mean.norm(dim=-1, keepdim=True)
    projected_text_features = projected_text_features / projected_text_features.norm(dim=-1, keepdim=True)
    
    # 融合视觉和文本特征
    alpha = args.alpha
    beta = 1.0 - alpha
    
    fused_prototypes = alpha * visual_prototypes_mean + beta * projected_text_features
    fused_prototypes = fused_prototypes / fused_prototypes.norm(dim=-1, keepdim=True)
    
    # 恢复多尺度结构
    if is_multiscale:
        num_scales = visual_prototypes.shape[1]
        fused_prototypes = fused_prototypes.unsqueeze(1).expand(-1, num_scales, -1)
    
    # 保存融合原型
    output_dir = os.path.dirname(args.output_prototype)
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = proto_data.copy()
    output_data['prototypes'] = fused_prototypes
    output_data['info'] = f"CLIP-enhanced prototypes (alpha={alpha})"
    
    torch.save(output_data, args.output_prototype)
    print(f"Fused prototypes saved to: {args.output_prototype}")

if __name__ == "__main__":
    main()