import torch
import clip
from pathlib import Path
import argparse
import os
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Enhance background prototypes with CLIP text features")
    parser.add_argument("--input-prototype", required=True, help="Input background prototype file")
    parser.add_argument("--output-prototype", required=True, help="Output prototype file")
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model variant")
    parser.add_argument("--alpha", type=float, default=0.7, help="Visual feature weight")
    parser.add_argument("--bg-classes", default="stuff_classes.txt", help="Background classes file")
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

def generate_bg_templates(bg_class):
    """为每个背景类别生成文本模板描述"""
    templates = [
        f"a photo of {bg_class}",
        f"{bg_class} in the background",
        f"a scene with {bg_class}",
        f"{bg_class} texture",
        f"a background of {bg_class}",
        f"a scene showing {bg_class}",
        f"{bg_class} environment",
    ]
    return templates

# COCOStuff主要背景类别列表
DEFAULT_BG_CLASSES = [
    "sky", "ground", "road", "grass", "water", "floor", "wall", "ceiling", 
    "mountain", "building", "tree", "sea", "river", "field", "sand", "snow", 
    "sidewalk", "earth", "hill", "rock", "dirt", "mud", "gravel", "pavement", 
    "concrete", "asphalt", "wood", "metal", "plastic", "paper", "fabric", "fence",
    "cloud", "fog", "hill", "forest", "railing", "net", "cage", "window", "door",
    "cardboard", "light", "tile", "marble", "brick", "stone", "carpet", "rug"
]

def main():
    args = parse_args()
    
    # 加载CLIP模型
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, device = load_clip_model(args.clip_model)
    
    # 加载现有背景原型
    print(f"Loading background prototypes from: {args.input_prototype}")
    bg_data = torch.load(args.input_prototype)
    
    # 处理不同格式的背景原型
    if isinstance(bg_data, dict):
        bg_protos = bg_data['prototypes']
        bg_classes = bg_data.get('label_names', [])
    else:
        bg_protos = bg_data
        bg_classes = []
    
    # 如果没有类别名称，尝试从文件加载或使用默认列表
    if not bg_classes:
        if os.path.exists(args.bg_classes):
            with open(args.bg_classes, 'r') as f:
                bg_classes = [line.strip() for line in f.readlines()]
        else:
            print(f"No background classes found, using default COCOStuff classes")
            bg_classes = DEFAULT_BG_CLASSES
    
    # 确保背景原型数量与类别数匹配，否则截断
    if len(bg_protos.shape) == 3:
        # 多尺度背景原型 [num_classes, num_scales, dim]
        num_bg_classes = bg_protos.shape[0]
    else:
        # 单尺度背景原型
        num_bg_classes = len(bg_protos)
    
    # 如果类别数量不匹配，截断到较小值
    num_classes_to_use = min(num_bg_classes, len(bg_classes))
    bg_classes = bg_classes[:num_classes_to_use]
    
    print(f"Processing {len(bg_classes)} background classes")
    
    # 提取CLIP文本特征
    text_features_list = []
    
    print("Extracting CLIP text features for background classes...")
    for bg_class in tqdm(bg_classes):
        # 生成文本描述
        templates = generate_bg_templates(bg_class)
        
        # 提取特征
        text_features = extract_clip_features(clip_model, templates, device)
        
        # 平均多个模板的特征
        text_feature = text_features.mean(dim=0)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_features_list.append(text_feature.cpu())
    
    # 如果背景原型数量多于类别数，为剩余背景生成通用特征
    if num_bg_classes > len(bg_classes):
        generic_bg_templates = [
            "a background area", "non-object region", "background texture",
            "empty space in image", "general background"
        ]
        generic_text_features = extract_clip_features(clip_model, generic_bg_templates, device)
        generic_text_feature = generic_text_features.mean(dim=0)
        generic_text_feature = generic_text_feature / generic_text_feature.norm(dim=-1, keepdim=True)
        
        # 复制通用特征以匹配背景原型数量
        for _ in range(num_bg_classes - len(bg_classes)):
            text_features_list.append(generic_text_feature.cpu())
    
    text_prototypes = torch.stack(text_features_list)
    
    # 处理多尺度原型
    is_multiscale = len(bg_protos.shape) == 3
    
    if is_multiscale:
        # 多尺度原型 [num_classes, num_scales, dim]
        visual_prototypes_mean = bg_protos.mean(dim=1)
    else:
        # 单尺度原型 [num_classes, dim]
        visual_prototypes_mean = bg_protos
    
    # 投影文本特征以匹配视觉特征维度
    visual_dim = visual_prototypes_mean.shape[-1]
    text_dim = text_prototypes.shape[-1]
    
    print(f"Visual feature dimension: {visual_dim}")
    print(f"Text feature dimension: {text_dim}")
    
    projection = torch.nn.Linear(text_dim, visual_dim).to(device)
    torch.nn.init.eye_(projection.weight[:min(text_dim, visual_dim), :min(text_dim, visual_dim)])
    
    # 确保数据类型一致
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
        num_scales = bg_protos.shape[1]
        fused_prototypes = fused_prototypes.unsqueeze(1).expand(-1, num_scales, -1)
    
    # 保存融合原型
    output_dir = os.path.dirname(args.output_prototype)
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备输出数据
    if isinstance(bg_data, dict):
        output_data = bg_data.copy()
        output_data['prototypes'] = fused_prototypes
        output_data['info'] = f"CLIP-enhanced background prototypes (alpha={alpha})"
    else:
        output_data = fused_prototypes
    
    torch.save(output_data, args.output_prototype)
    print(f"Enhanced background prototypes saved to: {args.output_prototype}")

if __name__ == "__main__":
    main()