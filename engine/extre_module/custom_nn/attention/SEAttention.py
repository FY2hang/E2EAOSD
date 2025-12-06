import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """SE注意力模块 (Squeeze-and-Excitation Attention)
    
    SE注意力机制通过以下三个步骤实现通道级别的特征重标定：
    1. Squeeze: 全局平均池化获取通道级别的全局信息
    2. Excitation: 通过FC层学习通道间的依赖关系
    3. Scale: 将学到的权重应用到原特征上
    
    Args:
        channels (int): 输入特征图的通道数
        reduction (int): 压缩比，用于控制中间层的维度，默认为16
    """
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        
        # Squeeze操作: 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation操作: 两层全连接网络
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: 全局平均池化 [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.global_pool(x).view(b, c)
        
        # Excitation: 学习通道注意力权重 [B, C] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale: 应用注意力权重
        return x * y.expand_as(x)


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)
    
    module = SEAttention(channel, reduction=16).to(device)
    
    outputs = module(inputs)
    
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)