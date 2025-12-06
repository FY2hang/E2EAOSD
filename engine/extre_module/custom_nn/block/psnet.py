'''
本文件结合PartialConv和RGCSPELAN模块
运用PartialConv思想改进RGCSPELAN模块
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F
from engine.extre_module.ultralytics_nn.conv import Conv, DWConv, DSConv
from engine.extre_module.custom_nn.conv_module.FDConv import FDConv
from engine.extre_module.custom_nn.conv_module.FDConv_trt import FDConv_trt
from engine.extre_module.custom_nn.conv_module.psconv import PSConv
from engine.extre_module.custom_nn.conv_module.FourierConv import FourierConv
from engine.extre_module.torch_utils import model_fuse_test

# class PartialConv(nn.Module):
#     """
#     PartialConv: 部分卷积模块，只对部分通道进行卷积操作
    
#     参数:
#         inc (int): 输入通道数
#         ouc (int): 输出通道数  
#         n_div (int): 通道分割比例，inc//n_div为部分卷积通道数
#         kernel_size (int): 卷积核大小
#     """
#     def __init__(self, inc, ouc, n_div=4, kernel_size=3):
#         super().__init__()
        
#         self.partial_channels = inc // n_div
#         self.identity_channels = inc - self.partial_channels
        
#         # 对部分通道使用FDConv进行卷积操作
#         self.partial_conv = FDConv(self.partial_channels, self.partial_channels, kernel_size=kernel_size)
#         # self.partial_conv = PSConv(self.partial_channels, self.partial_channels, k=kernel_size, s=1)
#         # self.partial_conv = FourierConv(self.partial_channels, self.partial_channels, size=kernel_size, s=1, act=True)

        
#         # 通道数调整卷积
#         self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()
        
#     def forward(self, x):
#         # 将输入分割为部分卷积通道和恒等通道
#         x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        
#         # 只对部分通道进行卷积
#         x1 = self.partial_conv(x1)
        
#         # 拼接处理后的部分通道和恒等通道
#         y = torch.cat([x1, x2], 1)
        
#         # 通道数调整
#         y = self.conv_adjust(y)
#         return y

class PartialConv(nn.Module):
    """
    按照提供架构设计的PartialConv模块
    
    架构流程：
    1. 输入分割：1/4通道走复杂路径，3/4通道直接跳跃
    2. 复杂路径：FDConv3×3 → AvgPool+MaxPool → DWConv3×3  
    3. 特征融合：Concat → Conv1×1
    
    参数:
        inc (int): 输入通道数
        ouc (int): 输出通道数
        n_div (int): 通道分割比例，默认4（即1/4走复杂路径）
    """
    def __init__(self, inc, ouc, n_div=4, kernel_size=3):
        super().__init__()
        
        # 通道分割
        self.complex_channels = inc // n_div  # 1/4通道走复杂路径
        self.skip_channels = inc - self.complex_channels  # 3/4通道直接跳跃
        
        # 1. FDConv 3×3
        self.fdconv = FDConv_trt(self.complex_channels, self.complex_channels, kernel_size=3)
        self.fdconv = FDConv(self.complex_channels, self.complex_channels, kernel_size=3)

        # 2. AvgPool 和 MaxPool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 3. DWConv 3×3 (深度可分离卷积)
        self.dwconv = DWConv(self.complex_channels * 2, self.complex_channels, k=3, s=1, d=1, act=True)

        # 4. Conv 1×1 (最终通道调整)
        self.final_conv = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()
        
    def forward(self, x):
        # 通道分割
        x_complex = x[:, :self.complex_channels]  # 前1/4通道
        x_skip = x[:, self.complex_channels:]     # 后3/4通道
        
        # 复杂路径处理
        # 1. FDConv 3×3
        x_fd = self.fdconv(x_complex)
        
        # 2. AvgPool + MaxPool
        original_size = x_fd.shape[2:]
        x_avg = self.avg_pool(x_fd)  # [B, C, 1, 1]
        x_max = self.max_pool(x_fd)  # [B, C, 1, 1]
        
        # 上采样回原始尺寸
        x_avg = F.interpolate(x_avg, size=original_size, mode='nearest')
        x_max = F.interpolate(x_max, size=original_size, mode='nearest')
        
        # 拼接AvgPool和MaxPool结果
        x_pooled = torch.cat([x_avg, x_max], dim=1)  # [B, 2C, H, W]
        
        # 3. DWConv 3×3
        x_dw = self.dwconv(x_pooled)  # [B, C, H, W]
        
        # 4. Concat (复杂路径结果 + 跳跃连接)
        x_concat = torch.cat([x_dw, x_skip], dim=1)  # [B, inc, H, W]
        
        # 5. Conv 1×1
        output = self.final_conv(x_concat)
        
        return output

class PFE_Net(nn.Module):
    """
    PFE_Net: 部分卷积多阶段网络模块
    结合PartialConv思想改进的RGCSPELAN模块

    创新点：
    1. 使用PartialConv替代传统的全通道卷积，降低计算复杂度
    2. 在多阶段特征提取中应用部分卷积思想
    3. 保持原有的多路径特征聚合机制
    
    参数:
        c1 (int): 输入通道数
        c2 (int): 输出通道数
        n (int): 额外的中间卷积层数
        scale (float): 中间通道的缩放系数
        e (float): 隐藏通道的扩展因子
        n_div (int): PartialConv的通道分割比例
    """
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5, n_div=4):
        super(PFE_Net, self).__init__()

        # 计算中间通道数量
        self.c = int(c2 * e)  # 隐藏通道数
        self.mid = int(self.c * scale)  # 经过缩放后的中间通道数
        
        # 1x1卷积用于将输入特征拆分为两个部分
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # 最终的1x1卷积层，用于整合所有处理后的特征
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        # 使用PartialConv替代RepConv，处理输入特征的第二部分
        self.cv3 = PartialConv(self.c, self.mid, n_div=n_div, kernel_size=3)
        
        # 一系列PartialConv层，用于进一步特征提取
        self.m = nn.ModuleList([
            PartialConv(self.mid, self.mid, n_div=n_div, kernel_size=3) 
            for _ in range(n - 1)
        ])
        
        # 最后的PartialConv，用于进一步处理最后阶段的特征
        self.cv4 = PartialConv(self.mid, self.mid, n_div=n_div, kernel_size=1)
        
        # 残差连接：处理输入输出通道数匹配问题
        self.residual_conv = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
    
    def forward(self, x):
        """前向传播，使用chunk()方法分割特征图，添加残差连接"""
        
        # 保存输入用于残差连接
        identity = x
        
        # 步骤1: 使用1x1卷积将输入特征拆分成两部分
        y = list(self.cv1(x).chunk(2, 1))
        
        # 步骤2: 对拆分的第二部分应用PartialConv
        y[-1] = self.cv3(y[-1])
        
        # 步骤3: 依次通过多个PartialConv进行特征提取
        y.extend(m(y[-1]) for m in self.m)
        
        # 步骤4: 使用PartialConv进一步提取特征
        y.append(self.cv4(y[-1]))
        
        # 步骤5: 将所有处理后的特征图拼接，并通过最终1x1卷积得到输出
        output = self.cv2(torch.cat(y, 1))
        
        # 残差连接：将原始输入添加到输出
        residual = self.residual_conv(identity)
        output = output + residual
        
        return output

class EnhancedPFE_Net(nn.Module):
    """
    EnhancedPSNet: 进一步增强的PSNet模块
    
    额外改进：
    1. 引入自适应的通道分割比例
    2. 添加残差连接增强信息流
    3. 使用注意力机制增强特征表达
    """
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5, n_div=4, use_residual=True):
        super(EnhancedPFE_Net, self).__init__()
        
        self.c = int(c2 * e)
        self.mid = int(self.c * scale)
        self.use_residual = use_residual and c1 == c2
        
        # 通道分割卷积
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # 最终整合卷积
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        # 主要的PartialConv模块
        self.cv3 = PartialConv(self.c, self.mid, n_div=n_div, kernel_size=3)
        
        # 中间的PartialConv层
        self.m = nn.ModuleList([
            PartialConv(self.mid, self.mid, n_div=n_div, kernel_size=3)
            for _ in range(n - 1)
        ])
        
        # 最终的特征精炼
        self.cv4 = PartialConv(self.mid, self.mid, n_div=n_div, kernel_size=1)
        
        # 残差连接的调整层
        if self.use_residual:
            self.residual_conv = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
            
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 16, 1), 1),  # 防止通道数为0
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 16, 1), c2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # 主要的特征提取路径
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        
        # 特征聚合
        out = self.cv2(torch.cat(y, 1))
        
        # 添加残差连接
        if self.use_residual:
            out = out + self.residual_conv(identity)
            
        # 应用通道注意力
        attention = self.channel_attention(out)
        out = out * attention
        
        return out

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 32, 64, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    print(BLUE + "=== 测试 PSNet 模块 ===" + RESET)
    
    try:
        # 测试基础PFE_Net模块
        module1 = PFE_Net(in_channel, out_channel, n=2, scale=0.5, e=0.5, n_div=4).to(device)
        outputs1 = module1(inputs)
        print(GREEN + f'PFE_Net - inputs.size: {inputs.size()} outputs.size: {outputs1.size()}' + RESET)
        print(f'PFE_Net参数量: {count_parameters(module1):,}')

        # 测试增强版PFE_Net模块 - 修正了类名
        module2 = EnhancedPFE_Net(in_channel, out_channel, n=2, scale=0.5, e=0.5, n_div=4).to(device)
        outputs2 = module2(inputs)
        print(GREEN + f'EnhancedPFE_Net - inputs.size: {inputs.size()} outputs.size: {outputs2.size()}' + RESET)
        print(f'EnhancedPFE_Net参数量: {count_parameters(module2):,}')
        
        # 简单的前向传播速度测试
        import time
        
        print(YELLOW + "\n=== 速度测试 ===" + RESET)
        
        # 预热
        for _ in range(10):
            with torch.no_grad():
                _ = module1(inputs)
                _ = module2(inputs)
        
        # PSNet速度测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = module1(inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        PFE_Net_time = (time.time() - start_time) / 100
        
        # EnhancedPSNet速度测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = module2(inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        enhanced_time = (time.time() - start_time) / 100

        print(f'PFE_Net平均推理时间: {PFE_Net_time*1000:.3f}ms')
        print(f'EnhancedPFE_Net平均推理时间: {enhanced_time*1000:.3f}ms')

        print(GREEN + "\n测试完成! ✅" + RESET)
        
    except Exception as e:
        print(RED + f"错误: {e}" + RESET)
        import traceback
        traceback.print_exc()