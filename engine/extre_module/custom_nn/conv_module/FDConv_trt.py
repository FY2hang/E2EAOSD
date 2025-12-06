'''  
本文件由BiliBili：魔傀面具整理
engine/extre_module/module_images/CVPR2025-FDConv.png
论文链接：https://arxiv.org/abs/2503.18783   
''' 
  
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd     
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
from engine.extre_module.ultralytics_nn.conv import Conv, DWConv, DSConv
    
class FDConv_trt(nn.Module):
    """基于残差连接的高效实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        
        mid_channels = max(in_channels // 2, 16)
        
        # 频域模拟分支
        self.freq_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 空域分支
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 注意力门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.final_act = nn.SiLU()
    
    def forward(self, x):
        freq_feat = self.freq_branch(x)
        spatial_feat = self.spatial_branch(x)
        
        # 注意力加权
        weights = self.gate(x)
        freq_weight, spatial_weight = weights[:, 0:1], weights[:, 1:2]
        
        output = freq_feat * freq_weight + spatial_feat * spatial_weight
        return self.final_act(output)
     
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 2, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = FDConv_trt(in_channel, out_channel, 3).to(device)

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
     
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,    
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)    
