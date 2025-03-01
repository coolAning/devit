import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

def get_synonyms(class_name):
    """返回类别的同义词列表"""
    synonyms = {
        "person": ["human", "individual", "pedestrian", "man", "woman"],
        "car": ["automobile", "vehicle", "sedan", "auto"],
        "dog": ["canine", "puppy", "hound"],
        "cat": ["feline", "kitty", "kitten"],
        # 添加更多类别的同义词...
    }
    return synonyms.get(class_name, [class_name])

def get_attributes(class_name):
    """返回类别的可能属性"""
    attributes = {
        "person": ["tall", "short", "young", "old", "wearing clothes"],
        "car": ["red", "blue", "parked", "moving", "shiny"],
        "dog": ["furry", "small", "large", "playful", "brown"],
        "cat": ["furry", "small", "grey", "black", "white"],
        # 添加更多类别的属性...
    }
    return attributes.get(class_name, [])

def get_actions(class_name):
    """返回类别的可能动作"""
    actions = {
        "person": ["walking", "standing", "sitting", "running"],
        "car": ["parked", "moving", "stopped", "turning"],
        "dog": ["running", "sitting", "playing", "sleeping"],
        "cat": ["walking", "sitting", "playing", "sleeping"],
        # 添加更多类别的动作...
    }
    return actions.get(class_name, [])

def generate_descriptions(objects, img_width, img_height, num_descriptions=5):
    """生成多样化的语义描述"""
    descriptions = []
    
    # 基本描述 - 简单列出对象
    basic = "An image containing " + ", ".join([f"{obj['name']}" for obj in objects])
    descriptions.append(basic)
    
    # 位置描述 - 包含空间信息
    position_desc = []
    for obj in objects:
        x_center = (obj['bbox'][0] + obj['bbox'][2]) / 2 / img_width
        y_center = (obj['bbox'][1] + obj['bbox'][3]) / 2 / img_height
        w = (obj['bbox'][2] - obj['bbox'][0]) / img_width
        h = (obj['bbox'][3] - obj['bbox'][1]) / img_height
        area = w * h
        
        # 位置描述
        position = ""
        if y_center < 0.33:
            position = "top"
        elif y_center > 0.66:
            position = "bottom"
            
        if x_center < 0.33:
            position = f"{position} left" if position else "left"
        elif x_center > 0.66:
            position = f"{position} right" if position else "right"
        else:
            position = f"{position} center" if position else "center"
            
        # 大小描述
        size = ""
        if area < 0.1:
            size = "small"
        elif area > 0.3:
            size = "large"
        
        position_desc.append(f"a {size+' ' if size else ''}{obj['name']} in the {position} of the image")
    
    descriptions.append("The image shows " + ", and ".join(position_desc))
    
    # 属性描述 - 添加随机属性
    attribute_desc = []
    for obj in objects:
        attrs = get_attributes(obj['name'])
        if attrs:
            attr = random.choice(attrs)
            attribute_desc.append(f"a {attr} {obj['name']}")
        else:
            attribute_desc.append(f"a {obj['name']}")
    
    descriptions.append("The image contains " + ", and ".join(attribute_desc))
    
    # 动作描述 - 添加可能的动作
    action_desc = []
    for obj in objects:
        actions = get_actions(obj['name'])
        if actions:
            action = random.choice(actions)
            action_desc.append(f"a {obj['name']} {action}")
        else:
            action_desc.append(f"a {obj['name']}")
    
    descriptions.append("The image shows " + ", and ".join(action_desc))
    
    # 同义词描述 - 使用同义词替换
    synonym_desc = []
    for obj in objects:
        synonyms = get_synonyms(obj['name'])
        if len(synonyms) > 1:
            synonym = random.choice(synonyms)
            synonym_desc.append(f"a {synonym}")
        else:
            synonym_desc.append(f"a {obj['name']}")
    
    descriptions.append("There is " + ", and ".join(synonym_desc) + " in the image")
    
    return descriptions[:num_descriptions]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to annotations file")
    parser.add_argument("--output", required=True, help="Output path for semantic descriptions")
    parser.add_argument("--num-desc", type=int, default=5, help="Number of descriptions per image")
    args = parser.parse_args()
    
    # 加载注释
    with open(args.input, 'r') as f:
        annotations = json.load(f)
    
    output_data = {}
    
    for image_id, ann in annotations.items():
        img_width = ann['width']
        img_height = ann['height']
        
        objects = []
        for obj in ann['annotations']:
            objects.append({
                'name': obj['category_name'],
                'bbox': obj['bbox']  # [x1, y1, x2, y2]
            })
        
        descriptions = generate_descriptions(objects, img_width, img_height, args.num_desc)
        output_data[image_id] = {
            'image_id': image_id,
            'descriptions': descriptions
        }
    
    # 保存输出
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated semantic descriptions for {len(output_data)} images")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()