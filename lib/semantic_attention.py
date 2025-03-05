import math
import torch
import torch.nn.functional as F
from torch import nn

class SemanticAttention(nn.Module):
    def __init__(self, feat_dim, sem_dim):
        super().__init__()
        self.hidden_dim = 256  # 使用较小的隐藏维度
        
        # 使用更稳定的映射层
        self.proj_q = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),  # 添加层归一化
            nn.ReLU()  # 使用ReLU激活函数
        )
        
        self.proj_k = nn.Sequential(
            nn.Linear(sem_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        self.proj_v = nn.Sequential(
            nn.Linear(sem_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 添加输出映射和归一化
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim)
        )
        
        # 缩放因子
        self.temperature = 0.07
        
    def forward(self, roi_feats, class_embeddings):
        # roi_feats: [N, C, HW]
        # class_embeddings: [K, D]
        q = self.proj_q(roi_feats.permute(0, 2, 1))  # [N, HW, hidden]
        k = self.proj_k(class_embeddings)[None, :, :]  # [1, K, hidden]
        v = self.proj_v(class_embeddings)[None, :, :]  # [1, K, hidden]
        
        # 扩展k和v到批次
        batch_size = q.size(0)
        k = k.expand(batch_size, -1, -1)  # [N, K, hidden]
        v = v.expand(batch_size, -1, -1)  # [N, K, hidden]
        
        # 计算注意力分数，使用更稳定的温度参数
        attn_logits = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        
        # 裁剪极端值防止数值不稳定
        attn_logits = torch.clamp(attn_logits, min=-10.0, max=10.0)
        
        # 应用softmax获取注意力权重
        attn = F.softmax(attn_logits, dim=-1)
        
        # 防止过于尖锐的注意力分布
        attn = attn + 1e-6  # 添加小偏置
        attn = attn / attn.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 应用注意力融合语义信息
        context = torch.bmm(attn, v)  # [N, HW, hidden]
        
        # 应用输出投影和归一化
        output = self.output_projection(context)
        
        # 使用残差连接增加稳定性
        original_feats = roi_feats.permute(0, 2, 1)  # [N, HW, C]
        output = output + original_feats
        
        return output.permute(0, 2, 1)  # [N, C, HW]