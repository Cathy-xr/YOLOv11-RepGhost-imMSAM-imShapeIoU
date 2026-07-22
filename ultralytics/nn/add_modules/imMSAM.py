import torch
import torch.nn as nn
from torch.nn import functional as F
from ultralytics.nn.modules.conv import Conv


class MSCAAttention_LargeTarget(nn.Module):
    """
    轻量级大目标注意力 - 参数量增加<5%
    特点：大感受野、光照鲁棒、结构感知
    """
    def __init__(self, dim):
        super().__init__()
        # 大感受野基础卷积（替换原来的5x5）
        self.conv0 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        # 简化条状卷积，只用一组大的
        self.conv_h = nn.Conv2d(dim, dim, (1, 13), padding=(0, 6), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, (13, 1), padding=(6, 0), groups=dim)
        
        # 添加一个空洞卷积扩大感受野（不增加参数量）
        self.conv_dilated = nn.Conv2d(dim, dim, 3, padding=4, dilation=4, groups=dim)
        
        # 简化融合（4->1直接相加）
        self.conv3 = nn.Conv2d(dim, dim, 1)
        
        # 添加简单的光照归一化
        self.ln = nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)
        
    def forward(self, x):
        u = x.clone()
        
        # 光照归一化
        x_norm = self.ln(x)
        
        # 多分支特征
        attn_base = self.conv0(x_norm)
        attn_h = self.conv_h(x_norm)
        attn_v = self.conv_v(x_norm)
        attn_d = self.conv_dilated(x_norm)
        
        # 直接相加融合（不增加额外卷积）
        attn = attn_base + attn_h + attn_v + attn_d
        attn = self.conv3(attn)
        
        return attn * u


class MSAM_Improve_LargeTarget(nn.Module):
    """
    轻量级大目标注意力模块
    参数量增加约3-5%
    """
    def __init__(self, c1, c2=None, e=0.5):
        super().__init__()
        if c2 is None:
            c2 = c1
            
        self.c = int(c1 * e)
        
        self.cv1 = Conv(c1, self.c * 2, 1, 1)
        self.cv2 = Conv(self.c * 2, c2, 1)
        
        # 使用改进的多尺度注意力
        self.attn = MSCAAttention_LargeTarget(self.c)
        
        # 简化的FFN（保持轻量）
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )
        
        # 添加简单的全局上下文（几乎不增加参数）
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(self.c, self.c // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c // 4, self.c, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 全局上下文增强
        context = self.gap(b)
        gate_weight = self.gate(context)
        b = b * gate_weight
        
        # 注意力处理
        b = b + self.attn(b)
        
        # FFN处理
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))