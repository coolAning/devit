import argparse
import os
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm

# VOC 类别
VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']  # fmt: skip

# 基本类别（前15个）和新类别（后5个）
BASE_CLASSES = VOC_CLASSES[:15]
NOVEL_CLASSES = VOC_CLASSES[15:]

# 增强词汇表（为每个类别提供多样化描述）
CLASS_VOCABULARY = {
    'aeroplane': ['airplane', 'aircraft', 'plane', 'jet', 'flying vehicle'],
    'bicycle': ['bike', 'cycle', 'two-wheeler', 'pedal bike', 'push bike'],
    'bird': ['avian', 'flying animal', 'feathered animal', 'winged creature'],
    'boat': ['ship', 'vessel', 'watercraft', 'sailing boat', 'marine vehicle'],
    'bottle': ['flask', 'container', 'glass bottle', 'beverage container'],
    'bus': ['coach', 'autobus', 'passenger vehicle', 'transit vehicle', 'public transport'],
    'car': ['automobile', 'vehicle', 'motor car', 'passenger car', 'auto'],
    'cat': ['feline', 'kitty', 'house cat', 'domestic cat', 'pet'],
    'chair': ['seat', 'stool', 'seating furniture', 'sitting furniture'],
    'cow': ['bovine', 'cattle', 'farm animal', 'beef animal', 'livestock'],
    'diningtable': ['table', 'dining table', 'kitchen table', 'eating surface'],
    'dog': ['canine', 'puppy', 'domestic dog', 'pet', 'hound'],
    'horse': ['equine', 'stallion', 'mare', 'pony', 'steed'],
    'motorbike': ['motorcycle', 'motor cycle', 'bike', 'moped', 'two-wheeler'],
    'person': ['human', 'man', 'woman', 'individual', 'people'],
    'pottedplant': ['houseplant', 'indoor plant', 'plant in pot', 'decorative plant', 'potted flower'],
    'sheep': ['ram', 'ewe', 'lamb', 'woolly animal', 'farm animal'],
    'sofa': ['couch', 'settee', 'loveseat', 'lounge', 'divan'],
    'train': ['locomotive', 'rail vehicle', 'railway transport', 'rail transport'],
    'tvmonitor': ['television', 'TV', 'display', 'screen', 'monitor']
}

# 位置描述词汇
POSITION_TERMS = {
    'top-left': ['top left corner', 'upper left', 'northwest area'],
    'top-center': ['top center', 'upper middle', 'top of the image'],
    'top-right': ['top right corner', 'upper right', 'northeast area'],
    'middle-left': ['middle left', 'left side', 'western part of the image'],
    'center': ['center', 'middle', 'central area', 'middle of the image'],
    'middle-right': ['middle right', 'right side', 'eastern part of the image'],
    'bottom-left': ['bottom left corner', 'lower left', 'southwest area'],
    'bottom-center': ['bottom center', 'lower middle', 'bottom of the image'],
    'bottom-right': ['bottom right corner', 'lower right', 'southeast area']
}

# 大小描述词汇
SIZE_TERMS = {
    'tiny': ['very small', 'tiny', 'minute', 'extremely small'],
    'small': ['small', 'little', 'compact', 'undersized'],
    'medium': ['medium-sized', 'moderate', 'average-sized', 'regular'],
    'large': ['large', 'big', 'sizable', 'substantial'],
    'huge': ['very large', 'huge', 'enormous', 'gigantic']
}

# 动作描述词汇（针对不同类别）
ACTION_TERMS = {
    'aeroplane': ['flying', 'soaring', 'cruising', 'taking off', 'landing'],
    'bicycle': ['parked', 'leaning', 'standing', 'positioned'],
    'bird': ['perching', 'flying', 'sitting', 'resting', 'standing'],
    'boat': ['floating', 'sailing', 'docked', 'moving across water'],
    'bottle': ['standing', 'positioned', 'placed'],
    'bus': ['parked', 'moving', 'stopped', 'driving'],
    'car': ['parked', 'driving', 'moving', 'stopped'],
    'cat': ['sitting', 'lying', 'walking', 'sleeping', 'resting'],
    'chair': ['standing', 'positioned', 'placed'],
    'cow': ['standing', 'grazing', 'resting', 'lying'],
    'diningtable': ['standing', 'positioned', 'placed'],
    'dog': ['sitting', 'standing', 'lying', 'playing', 'resting'],
    'horse': ['standing', 'grazing', 'running', 'walking'],
    'motorbike': ['parked', 'standing', 'moving', 'positioned'],
    'person': ['standing', 'sitting', 'walking', 'running', 'posing'],
    'pottedplant': ['growing', 'placed', 'positioned', 'standing'],
    'sheep': ['grazing', 'standing', 'resting', 'grouped together'],
    'sofa': ['positioned', 'placed', 'standing'],
    'train': ['moving', 'stopped', 'stationed', 'traveling'],
    'tvmonitor': ['displaying', 'showing', 'mounted', 'positioned']
}

def parse_args():
    parser = argparse.ArgumentParser(description='生成VOC数据集的多样化语义描述（针对对比学习）')
    parser.add_argument('--voc-root', type=str, default='/mnt/d/datasets/VOCdevkit',
                        help='VOC数据集根目录')
    parser.add_argument('--split-dir', type=str, default='/mnt/d/datasets/VOCdevkit/VOC2012/vocsplit',
                        help='Few-shot数据分割目录')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/d/datasets/VOCdevkit/VOC2012/semantic_descriptions',
                        help='语义描述输出目录')
    parser.add_argument('--num-descriptions', type=int, default=3,
                        help='为每张图像生成的描述数量')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    return parser.parse_args()

def get_object_position(bbox, img_width, img_height):
    """确定对象在图像中的位置"""
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    # 水平位置
    if cx < img_width / 3:
        h_pos = 'left'
    elif cx < 2 * img_width / 3:
        h_pos = 'center'
    else:
        h_pos = 'right'
    
    # 垂直位置
    if cy < img_height / 3:
        v_pos = 'top'
    elif cy < 2 * img_height / 3:
        v_pos = 'middle'
    else:
        v_pos = 'bottom'
    
    # 组合位置，中心特殊处理
    if v_pos == 'middle' and h_pos == 'center':
        return 'center'
    else:
        return f"{v_pos}-{h_pos}"

def get_object_size(bbox, img_width, img_height):
    """确定对象在图像中的相对大小"""
    _, _, width, height = bbox
    area = width * height
    img_area = img_width * img_height
    ratio = area / img_area
    
    if ratio < 0.02:
        return 'tiny'
    elif ratio < 0.1:
        return 'small'
    elif ratio < 0.25:
        return 'medium'
    elif ratio < 0.5:
        return 'large'
    else:
        return 'huge'

def extract_objects_from_annotation(anno_file):
    """从VOC XML标注文件中提取对象信息"""
    try:
        tree = ET.parse(anno_file)
        root = tree.getroot()
        
        # 获取图像尺寸
        size_elem = root.find('size')
        img_width = int(size_elem.find('width').text)
        img_height = int(size_elem.find('height').text)
        filename = root.find('filename').text
        
        # 提取对象
        objects = []
        for obj_elem in root.findall('object'):
            name = obj_elem.find('name').text
            if name not in VOC_CLASSES:
                continue
                
            bbox_elem = obj_elem.find('bndbox')
            xmin = float(bbox_elem.find('xmin').text)
            ymin = float(bbox_elem.find('ymin').text)
            xmax = float(bbox_elem.find('xmax').text)
            ymax = float(bbox_elem.find('ymax').text)
            
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # [x, y, width, height]
            
            # 确定位置和大小
            position = get_object_position(bbox, img_width, img_height)
            size = get_object_size(bbox, img_width, img_height)
            
            objects.append({
                "name": name,
                "bbox": bbox,
                "position": position,
                "size": size
            })
        
        return objects, img_width, img_height, filename
    except Exception as e:
        print(f"解析{anno_file}时出错: {e}")
        return [], 0, 0, ""

def generate_basic_description(objects):
    """生成基本描述：只包含对象类别"""
    # 统计每个类别的数量
    class_counts = defaultdict(int)
    for obj in objects:
        class_counts[obj["name"]] += 1
    
    # 构建描述
    parts = []
    for cls, count in class_counts.items():
        if count == 1:
            parts.append(f"a {cls}")
        else:
            parts.append(f"{count} {cls}s")
    
    # 根据对象数量构建句子
    if len(parts) == 0:
        return "This image does not contain any recognized objects."
    elif len(parts) == 1:
        return f"This image contains {parts[0]}."
    elif len(parts) == 2:
        return f"This image contains {parts[0]} and {parts[1]}."
    else:
        return f"This image contains {', '.join(parts[:-1])}, and {parts[-1]}."

def generate_positional_description(objects):
    """生成带位置和大小信息的描述"""
    descriptions = []
    
    # 按类别对对象进行分组
    class_objects = defaultdict(list)
    for obj in objects:
        class_objects[obj["name"]].append(obj)
    
    # 为每个类别生成描述
    for cls, objs in class_objects.items():
        if len(objs) == 1:
            obj = objs[0]
            pos_term = random.choice(POSITION_TERMS.get(obj["position"], [obj["position"]]))
            size_term = random.choice(SIZE_TERMS.get(obj["size"], [obj["size"]]))
            action = random.choice(ACTION_TERMS.get(cls, ["present"]))
            
            templates = [
                f"A {size_term} {cls} is {action} in the {pos_term} of the image.",
                f"There is a {size_term} {cls} {action} in the {pos_term}.",
                f"The {pos_term} shows a {size_term} {cls} {action}."
            ]
            descriptions.append(random.choice(templates))
        else:
            # 多个同类对象
            positions = [obj["position"] for obj in objs]
            sizes = [obj["size"] for obj in objs]
            
            # 位置描述
            if len(set(positions)) == 1:
                pos_desc = f"in the {random.choice(POSITION_TERMS.get(positions[0], [positions[0]]))}"
            else:
                pos_desc = "in different areas of the image"
            
            # 大小描述
            if len(set(sizes)) == 1:
                size_desc = random.choice(SIZE_TERMS.get(sizes[0], [sizes[0]]))
            else:
                size_desc = "various sized"
            
            templates = [
                f"There are {len(objs)} {size_desc} {cls}s {pos_desc}.",
                f"The image contains {len(objs)} {cls}s of {size_desc} size {pos_desc}.",
                f"{len(objs)} {cls}s can be seen {pos_desc}."
            ]
            descriptions.append(random.choice(templates))
    
    return " ".join(descriptions)

def generate_rich_description(objects):
    """生成丰富的语义描述，包含同义词和场景解释"""
    if not objects:
        return "This image does not appear to contain any recognizable objects."
    
    # 按类别对对象进行分组
    class_objects = defaultdict(list)
    for obj in objects:
        class_objects[obj["name"]].append(obj)
    
    descriptions = []
    
    # 1. 创建场景开头
    scene_intros = [
        "The image displays a scene with ",
        "This photograph captures ",
        "The picture shows ",
        "In this image, we can observe "
    ]
    
    # 2. 为每个类别生成丰富描述
    class_descriptions = []
    for cls, objs in class_objects.items():
        # 获取这个类的替代词
        alt_terms = CLASS_VOCABULARY.get(cls, [cls])
        class_term = random.choice(alt_terms)
        
        if len(objs) == 1:
            obj = objs[0]
            pos_term = random.choice(POSITION_TERMS.get(obj["position"], [obj["position"]]))
            size_term = random.choice(SIZE_TERMS.get(obj["size"], [obj["size"]]))
            action = random.choice(ACTION_TERMS.get(cls, ["present"]))
            
            templates = [
                f"a {size_term} {class_term} ({cls}) {action} in the {pos_term}",
                f"one {class_term} of {size_term} size {action} in the {pos_term}",
                f"a {size_term} {cls}, specifically a {class_term}, {action} in the {pos_term}"
            ]
            class_descriptions.append(random.choice(templates))
        else:
            # 计算主要位置和大小
            positions = [obj["position"] for obj in objs]
            common_pos = max(set(positions), key=positions.count)
            pos_term = random.choice(POSITION_TERMS.get(common_pos, [common_pos]))
            
            templates = [
                f"{len(objs)} {class_term}s ({cls}s) primarily in the {pos_term}",
                f"multiple {cls}s, which appear to be {class_term}s, scattered around the {pos_term}",
                f"a group of {len(objs)} {class_term}s in the {pos_term} and surrounding areas"
            ]
            class_descriptions.append(random.choice(templates))
    
    # 3. 组合描述
    if len(class_descriptions) == 1:
        main_content = class_descriptions[0]
    elif len(class_descriptions) == 2:
        main_content = f"{class_descriptions[0]} and {class_descriptions[1]}"
    else:
        main_content = f"{', '.join(class_descriptions[:-1])}, and {class_descriptions[-1]}"
    
    # 4. 添加场景解释（基于存在的对象）
    scene_contexts = {
        'person': ["appears to be a portrait", "shows people in their environment", "captures human activities"],
        'car': ["might be taken on a street or parking area", "shows transportation", "depicts vehicles"],
        'dog': ["appears to be a pet photograph", "shows a domestic scene", "captures animal companions"],
        'horse': ["seems to be in a rural setting", "shows farm animals", "depicts equestrian scenes"],
        'bird': ["appears to be wildlife photography", "shows natural elements", "captures avian life"]
    }
    
    context_classes = [cls for cls in scene_contexts.keys() if cls in class_objects]
    
    if context_classes:
        context_options = []
        for cls in context_classes:
            context_options.extend(scene_contexts[cls])
        context = random.choice(context_options)
        
        description = f"{random.choice(scene_intros)}{main_content}. The scene {context}."
    else:
        description = f"{random.choice(scene_intros)}{main_content}."
    
    return description

def generate_contrastive_descriptions(objects, img_width, img_height, filename, num_descriptions=3):
    """为一张图像生成多个语义描述，适合对比学习"""
    if not objects:
        return ["This image contains no recognizable objects."] * num_descriptions
    
    descriptions = []
    
    # 1. 基本描述（对象列表）
    descriptions.append(generate_basic_description(objects))
    
    # 2. 位置描述（包含空间信息）
    descriptions.append(generate_positional_description(objects))
    
    # 3. 丰富描述（更复杂的场景描述）
    descriptions.append(generate_rich_description(objects))
    
    # 如果需要更多描述，从丰富描述生成额外的变体
    while len(descriptions) < num_descriptions:
        descriptions.append(generate_rich_description(objects))
    
    return descriptions[:num_descriptions]

def process_split_file(split_file, voc_root, args):
    """处理一个分割文件，生成所有图像的描述"""
    with open(split_file, 'r') as f:
        lines = f.readlines()
    
    image_data = {}
    processed_count = 0
    
    for line in tqdm(lines, desc=f"处理 {os.path.basename(split_file)}"):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # 提取图像路径
        img_path = line
        if not os.path.exists(img_path):
            continue
        
        # 找到对应的XML标注
        # 处理不同格式的路径
        if 'VOC2007' in img_path:
            year = 'VOC2007'
        elif 'VOC2012' in img_path:
            year = 'VOC2012'
        else:
            continue
        
        img_name = os.path.basename(img_path).split('.')[0]
        anno_file = os.path.join(voc_root, year, 'Annotations', f"{img_name}.xml")
        
        if not os.path.exists(anno_file):
            continue
        
        # 提取对象信息
        objects, img_width, img_height, filename = extract_objects_from_annotation(anno_file)
        if not objects:
            continue
        
        # 生成描述
        descriptions = generate_contrastive_descriptions(
            objects, img_width, img_height, filename, args.num_descriptions
        )
        
        # 收集对象类别
        classes = list(set(obj["name"] for obj in objects))
        
        # 存储结果
        image_data[img_path] = {
            "descriptions": descriptions,
            "classes": classes,
            "objects": [{
                "class": obj["name"],
                "bbox": obj["bbox"],
                "position": obj["position"],
                "size": obj["size"]
            } for obj in objects]
        }
        
        processed_count += 1
    
    return image_data, processed_count

def generate_class_descriptions():
    """为每个类别生成多层次描述，用于原型构建"""
    class_descriptions = {}
    
    for cls in VOC_CLASSES:
        # 获取不同的类别词汇
        alt_terms = CLASS_VOCABULARY.get(cls, [cls])
        
        # 基础描述
        basic_desc = f"This is a {cls}."
        
        # 详细描述
        detailed_desc = f"This is an image of a {cls}, which is a type of {random.choice(alt_terms)}."
        
        # 丰富描述
        rich_templates = [
            f"This image features a {random.choice(alt_terms)}, commonly known as a {cls}.",
            f"The photograph shows a {cls}, which is a kind of {random.choice(alt_terms)}.",
            f"A {random.choice(alt_terms)} ({cls}) is the main subject of this image."
        ]
        rich_desc = random.choice(rich_templates)
        
        class_descriptions[cls] = {
            "basic": basic_desc,
            "detailed": detailed_desc,
            "rich": rich_desc,
            "alternatives": alt_terms
        }
    
    return class_descriptions

def main():
    args = parse_args()
    random.seed(args.seed)
    
    print(f"VOC语义描述生成工具 - 对比学习版")
    print(f"VOC数据集路径: {args.voc_root}")
    print(f"Few-shot分割目录: {args.split_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"每张图像描述数量: {args.num_descriptions}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成每个类别的标准描述
    print("生成类别描述...")
    class_descriptions = generate_class_descriptions()
    with open(os.path.join(args.output_dir, "class_descriptions.json"), 'w') as f:
        json.dump(class_descriptions, f, indent=2)
    
    # 读取分割目录
    if not os.path.exists(args.split_dir):
        print(f"错误: 分割目录 {args.split_dir} 不存在!")
        return False
    
    seed_dirs = [d for d in os.listdir(args.split_dir) if os.path.isdir(os.path.join(args.split_dir, d)) and d.startswith("seed")]
    if not seed_dirs:
        print(f"错误: 在 {args.split_dir} 中未找到种子目录!")
        return False
    
    # 处理每个种子目录
    total_images = 0
    
    for seed_dir in seed_dirs:
        print(f"\n处理种子目录: {seed_dir}")
        seed_path = os.path.join(args.split_dir, seed_dir)
        
        # 创建输出种子目录
        output_seed_dir = os.path.join(args.output_dir, seed_dir)
        os.makedirs(output_seed_dir, exist_ok=True)
        
        # 处理该种子目录下的所有文件
        split_files = [f for f in os.listdir(seed_path) if f.endswith('.txt')]
        
        for split_file in split_files:
            # 跳过空文件
            if os.path.getsize(os.path.join(seed_path, split_file)) == 0:
                continue
                
            # 处理分割文件
            split_path = os.path.join(seed_path, split_file)
            image_data, count = process_split_file(split_path, args.voc_root, args)
            
            if count > 0:
                # 保存描述
                output_file = os.path.join(output_seed_dir, f"{os.path.splitext(split_file)[0]}_descriptions.json")
                with open(output_file, 'w') as f:
                    json.dump(image_data, f, indent=2)
                
                print(f"  处理了 {count} 张图像，保存到 {os.path.basename(output_file)}")
                total_images += count
    
    print(f"\n完成! 共为 {total_images} 张图像生成了语义描述，并保存到 {args.output_dir}")
    return True

if __name__ == "__main__":
    main()