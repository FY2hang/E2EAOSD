'''
本文件由BiliBili：魔傀面具整理
论文链接：https://arxiv.org/pdf/2411.15941
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv
from engine.extre_module.custom_nn.module.MSBlock import MSBlock
from engine.extre_module.custom_nn.mamba.SAVSS import SAVSS

def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple

class MobileMamba(nn.Module):
    def __init__(self, inc, ouc, global_ratio=0.25, local_ratio=0.25) -> None:
        super().__init__()

        self.global_channels = int(nearest_multiple_of_16(int(inc * global_ratio)))
        self.local_channels = int(inc * local_ratio)
        self.identity_channels = inc - self.global_channels - self.local_channels
        
        self.global_branch = SAVSS(self.global_channels)
        self.local_branch = MSBlock(self.local_channels, self.local_channels, kernel_sizes=[1, 3, 5])

        self.proj = Conv(inc, ouc, 1)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, (self.global_channels, self.local_channels, self.identity_channels), dim=1)

        x1 = self.global_branch(x1)
        x2 = self.local_branch(x2)

        y = torch.cat([x1, x2, x3], dim=1)
        y = self.proj(y)
        return y

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 128, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = MobileMamba(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)