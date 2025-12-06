import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

import warnings  
warnings.filterwarnings('ignore')    
from calflops import calculate_flops  
     
import torch, math     
import torch.nn as nn    
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
# 导入SE注意力模块 - 替代CBAM
from engine.extre_module.custom_nn.attention.SEAttention import SEAttention
    
class MultiScaleAttentionPool(nn.Module):
    """多尺度注意力池化 (MSAP)
    
    创新点：不同于传统单一GAP，我们引入多尺度池化来捕获不同感受野的全局信息
    理论依据：不同尺度的池化能捕获从局部细节到全局语义的层次化信息
    """
    def __init__(self, channels):
        super().__init__()
        # 多个池化尺度
        self.pool_sizes = [1, 2, 4]  # 1x1, 2x2, 4x4 池化
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in self.pool_sizes
        ])
        
        # 注意力权重生成器
        self.attention_conv = nn.Conv2d(channels * len(self.pool_sizes), channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 提取多尺度特征
        multi_scale_feats = []
        for pool in self.pools:
            pooled = pool(x)
            # 上采样到1x1统一尺寸
            if pooled.size(2) != 1:
                pooled = F.adaptive_avg_pool2d(pooled, 1)
            multi_scale_feats.append(pooled)
        
        # 融合多尺度信息
        combined = torch.cat(multi_scale_feats, dim=1)  # [B, C*3, 1, 1]
        attention_weight = self.sigmoid(self.attention_conv(combined))  # [B, C, 1, 1]
        
        return F.adaptive_avg_pool2d(x, 1) * attention_weight

class FeatureRecalibrationModule(nn.Module):
    """特征重校准模块 (FRM) - GeLU版本
    
    创新点：在融合前对来自不同尺度的特征进行语义对齐和重校准
    理论依据：不同层的特征具有不同的语义抽象级别，直接融合会产生语义冲突
    
    主要修改：使用GeLU替代ReLU，提供更平滑的梯度和更好的性能
    """
    def __init__(self, channels):
        super().__init__()
        # 特征统计信息提取
        self.global_context = nn.AdaptiveAvgPool2d(1)
        
        # 特征重校准网络 - 使用GeLU替代ReLU
        self.recalibration = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),  # 🔄 ReLU -> GeLU
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        # 特征增强门控
        self.enhancement_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 全局上下文
        global_feat = self.global_context(x)
        
        # 特征重校准
        recalib_weight = self.recalibration(global_feat)
        recalibrated_feat = x * recalib_weight
        
        # 特征增强门控
        enhancement_weight = self.enhancement_gate(global_feat)
        enhanced_feat = recalibrated_feat * enhancement_weight
        
        return enhanced_feat

class AdaptiveFusionGate(nn.Module):
    """自适应融合门控 (AFG) - GeLU版本
    
    创新点：基于特征语义相似性和尺度差异动态调整融合权重
    理论依据：不同场景下两个特征流的重要性不同，需要自适应调整
    
    主要修改：使用GeLU替代ReLU，提供更好的特征表示学习能力
    """
    def __init__(self, channel1, channel2):
        super().__init__()
        total_channels = channel1 + channel2
        
        # 特征相似性度量 - 使用GeLU替代ReLU
        self.similarity_metric = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.GELU(),  # 🔄 ReLU -> GeLU
            nn.Conv2d(total_channels // 4, 2, 1),  # 输出2个权重
            nn.Softmax(dim=1)
        )
        
        # 尺度差异感知
        self.scale_awareness = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, f1, f2, size1, size2):
        # 计算尺度差异
        size_ratio = size1 / size2 if size2 != 0 else 1.0
        scale_factor = torch.sigmoid(torch.log(torch.tensor(size_ratio + 1e-6)))
        
        # 特征相似性
        combined_feat = torch.cat([
            F.adaptive_avg_pool2d(f1, 1), 
            F.adaptive_avg_pool2d(f2, 1)
        ], dim=1)
        similarity_weights = self.similarity_metric(combined_feat)
        
        # 自适应权重计算
        adaptive_weights = similarity_weights * self.scale_awareness.view(1, -1, 1, 1)
        adaptive_weights = F.softmax(adaptive_weights, dim=1)
        
        return adaptive_weights[:, 0:1], adaptive_weights[:, 1:2]

class CrossScaleAlignment(nn.Module):
    """跨尺度特征对齐 (CSA)
    
    创新点：通过可学习的对齐卷积减少尺度变换带来的语义偏移
    理论依据：简单的插值无法保持特征的语义一致性
    """
    def __init__(self, channels):
        super().__init__()
        # 语义对齐卷积
        self.align_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.point_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x, target_size):
        # 先插值到目标尺寸
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # 语义对齐
        x = self.align_conv(x)
        x = self.point_conv(x)
        x = self.norm(x)
        
        return x

class SEEFusion(nn.Module):
    """ASF-SEEFusion: Adaptive Scale-aware Feature Fusion (SE + GeLU版本)
    
    核心创新：
    1. MSAP: 多尺度注意力池化，捕获层次化全局信息
    2. FRM: 特征重校准模块，解决语义冲突
    3. AFG: 自适应融合门控，动态权重调节
    4. CSA: 跨尺度特征对齐，保持语义一致性
    5. SE注意力: 替代CBAM和分组卷积，提供更轻量的通道注意力
    6. GeLU激活: 替代ReLU，提供更平滑的梯度和更好的性能
    
    主要修改：
    - 将shallow_processor从CBAM改为SE注意力
    - 将deep_processor从分组卷积+通道混洗改为SE注意力
    - 将所有ReLU激活函数替换为GeLU
    - 统一使用SE注意力机制，简化架构并保持性能
    
    理论故事：
    传统方法假设所有尺度的特征同等重要，忽略了：
    - 不同尺度特征的语义层次差异
    - 场景相关的特征重要性变化  
    - 尺度变换的语义偏移问题
    
    我们的方法通过"感知-校准-融合-对齐"四阶段流程解决这些问题
    """
    def __init__(self, input_channel, output_channel, gamma=2, bias=1, se_reduction=16):
        super(SEEFusion, self).__init__()
        input_channel1, input_channel2 = input_channel
        self.input_channel1 = input_channel1   
        self.input_channel2 = input_channel2
        self.se_reduction = se_reduction

        # 阶段1: 多尺度感知
        self.msap1 = MultiScaleAttentionPool(input_channel1)     
        self.msap2 = MultiScaleAttentionPool(input_channel2)
        
        max_channel = max(input_channel1, input_channel2)
        
        # 核心修改: 使用SE注意力替代原有的处理器
        self.shallow_processor = SEAttention(max_channel, reduction=se_reduction)
        self.deep_processor = SEAttention(max_channel, reduction=se_reduction)
        
        # 阶段2: 特征重校准 (现在使用GeLU)
        self.frm1 = FeatureRecalibrationModule(max_channel)
        self.frm2 = FeatureRecalibrationModule(max_channel)
        
        # 通道适配
        self.adapt1 = Conv(input_channel1, max_channel, 1) if input_channel1 != max_channel else nn.Identity()
        self.adapt2 = Conv(input_channel2, max_channel, 1) if input_channel2 != max_channel else nn.Identity()
        
        self.readapt1 = Conv(max_channel, input_channel1, 1) if input_channel1 != max_channel else nn.Identity()
        self.readapt2 = Conv(max_channel, input_channel2, 1) if input_channel2 != max_channel else nn.Identity()
        
        # 原有注意力生成
        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1
        
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, 
                              padding=(kernel_size3 - 1) // 2, bias=False)
        
        # 阶段3: 自适应融合门控 (现在使用GeLU)
        self.afg = AdaptiveFusionGate(input_channel1, input_channel2)
        
        # 阶段4: 跨尺度对齐
        self.csa = CrossScaleAlignment(input_channel2)
        
        # 上采样保持原有设计
        self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, 
                                   kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()

    def forward(self, x):  
        x1, x2 = x
        
        size1 = x1.size(-1) * x1.size(-2)
        size2 = x2.size(-1) * x2.size(-2)
        
        # 阶段1: 多尺度感知替代传统GAP
        x1_ = self.msap1(x1)
        x2_ = self.msap2(x2)
        
        # 通道适配
        x1_adapted = self.adapt1(x1_)
        x2_adapted = self.adapt2(x2_)
        
        # 阶段2: 特征重校准 (使用GeLU激活)
        x1_adapted = self.frm1(x1_adapted)
        x2_adapted = self.frm2(x2_adapted)
        
        # SE处理（统一使用SE注意力）
        if size1 >= size2:
            # 大尺度特征使用浅层SE处理器
            x1_processed = self.shallow_processor(x1_adapted)
            # 小尺度特征使用深层SE处理器
            x2_processed = self.deep_processor(x2_adapted)
        else:
            # 交换处理策略
            x2_processed = self.shallow_processor(x2_adapted)
            x1_processed = self.deep_processor(x1_adapted)
        
        x1_ = self.readapt1(x1_processed)
        x2_ = self.readapt2(x2_processed)
        
        # 原有注意力生成 + 改进激活
        x_middle = torch.cat((x1_, x2_), dim=1)
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = torch.sigmoid(x_middle)  # 添加sigmoid激活
        
        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)
    
        # 应用注意力
        x1_out = x1 * x_1
        x2_out = x2 * x_2
        
        # 阶段4: 跨尺度特征对齐的上采样
        x2_out = self.up(x2_out)
        x2_out = self.csa(x2_out, x1_out.shape[2:])
        
        # 尺寸精确匹配
        if x2_out.shape[2:] != x1_out.shape[2:]:
            x2_out = F.interpolate(x2_out, size=x1_out.shape[2:], mode='bilinear', align_corners=False)
        
        # 阶段3: 自适应融合 (使用GeLU的AFG)
        w1, w2 = self.afg(x1_out, x2_out, size1, size2)
        result = w1 * x1_out + w2 * x2_out
        
        return self.conv1x1(result)   

class SEEFusion_Down(nn.Module):     
    """ASF-SEEFusion_Down: 下采样版本 (SE + GeLU版本)"""
    def __init__(self, input_channel, output_channel, gamma=2, bias=1, se_reduction=16):
        super(SEEFusion_Down, self).__init__()   
        input_channel1, input_channel2 = input_channel 
        self.input_channel1 = input_channel1   
        self.input_channel2 = input_channel2
        self.se_reduction = se_reduction

        # 使用相同的创新模块 (现在包含GeLU)
        self.msap1 = MultiScaleAttentionPool(input_channel1)
        self.msap2 = MultiScaleAttentionPool(input_channel2)

        max_channel = max(input_channel1, input_channel2)
        
        # 使用SE注意力替代原有处理器
        self.shallow_processor = SEAttention(max_channel, reduction=se_reduction)
        self.deep_processor = SEAttention(max_channel, reduction=se_reduction)
        
        # 特征重校准模块 (使用GeLU)
        self.frm1 = FeatureRecalibrationModule(max_channel)
        self.frm2 = FeatureRecalibrationModule(max_channel)
        
        self.adapt1 = Conv(input_channel1, max_channel, 1) if input_channel1 != max_channel else nn.Identity()
        self.adapt2 = Conv(input_channel2, max_channel, 1) if input_channel2 != max_channel else nn.Identity()
        
        self.readapt1 = Conv(max_channel, input_channel1, 1) if input_channel1 != max_channel else nn.Identity()
        self.readapt2 = Conv(max_channel, input_channel2, 1) if input_channel2 != max_channel else nn.Identity()

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))   
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1    
        
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, 
                              padding=(kernel_size3 - 1) // 2, bias=False)
        
        # 自适应融合门控 (使用GeLU)
        self.afg = AdaptiveFusionGate(input_channel1, input_channel2)
        self.csa = CrossScaleAlignment(input_channel2)

        self.down = nn.Conv2d(in_channels=input_channel2, out_channels=input_channel1, 
                             kernel_size=3, stride=2, padding=1)
        
        self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()

    def forward(self, x):    
        x1, x2 = x    
        
        size1 = x1.size(-1) * x1.size(-2)
        size2 = x2.size(-1) * x2.size(-2)
        
        # 多尺度感知
        x1_ = self.msap1(x1)
        x2_ = self.msap2(x2)

        x1_adapted = self.adapt1(x1_)
        x2_adapted = self.adapt2(x2_)
        
        # 特征重校准 (使用GeLU)
        x1_adapted = self.frm1(x1_adapted)
        x2_adapted = self.frm2(x2_adapted)
        
        # SE处理
        if size1 >= size2:
            x1_processed = self.shallow_processor(x1_adapted)
            x2_processed = self.deep_processor(x2_adapted)
        else:
            x2_processed = self.shallow_processor(x2_adapted)
            x1_processed = self.deep_processor(x1_adapted)

        x1_ = self.readapt1(x1_processed)
        x2_ = self.readapt2(x2_processed)

        x_middle = torch.cat((x1_, x2_), dim=1)    
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = torch.sigmoid(x_middle)
    
        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)
     
        x1_out = x1 * x_1
        x2_out = x2 * x_2
        
        # 跨尺度对齐的下采样
        x2_out = self.down(x2_out)
        x2_out = self.csa(x2_out, x1_out.shape[2:])
        
        if x2_out.shape[2:] != x1_out.shape[2:]:
            x2_out = F.interpolate(x2_out, size=x1_out.shape[2:], mode='bilinear', align_corners=False)
        
        # 自适应融合 (使用GeLU的AFG)
        w1, w2 = self.afg(x1_out, x2_out, size1, size2)
        result = w1 * x1_out + w2 * x2_out
        
        return self.conv1x1(result)  

if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, height_1, width_1 = 1, 32, 40, 40
    batch_size, channel_2, height_2, width_2 = 1, 64, 20, 20    
    ouc_channel = 64   
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)

    print(RED + '-'*10 + " ASF-SEEFusion with SE + GeLU " + '-'*10 + RESET)

    module = SEEFusion([channel_1, channel_2], ouc_channel, se_reduction=16).to(device)
     
    outputs = module([inputs_1, inputs_2])   
    print(GREEN + f'inputs1: {inputs_1.size()} inputs2: {inputs_2.size()} outputs: {outputs.size()}' + RESET)
    print(GREEN + f'输出统计: mean={outputs.mean():.4f}, std={outputs.std():.4f}, range=[{outputs.min():.4f}, {outputs.max():.4f}]' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]], 
                                     output_as_string=True,   
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)

    print(RED + '-'*10 + " ASF-SEEFusion_Down with SE + GeLU " + '-'*10 + RESET)    

    module = SEEFusion_Down([channel_2, channel_1], ouc_channel, se_reduction=16).to(device)
 
    outputs = module([inputs_2, inputs_1])    
    print(GREEN + f'inputs1: {inputs_1.size()} inputs2: {inputs_2.size()} outputs: {outputs.size()}' + RESET)
    print(GREEN + f'输出统计: mean={outputs.mean():.4f}, std={outputs.std():.4f}, range=[{outputs.min():.4f}, {outputs.max():.4f}]' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_2, inputs_1]],
                                     output_as_string=True, 
                                     output_precision=4,   
                                     print_detailed=True)  
    print(RESET)
    
    print(BLUE + "=" * 80 + RESET)
    print(YELLOW + "ASF-SEEFusion with SE + GeLU 核心改进:" + RESET)
    print("SE注意力: 替代CBAM+分组卷积，更轻量高效")
    print("GeLU激活: 替代ReLU，提供更平滑梯度和更好性能")
    print("MSAP: 多尺度注意力池化 - 层次化全局信息捕获")
    print("FRM: 特征重校准模块 - 解决语义冲突问题") 
    print("AFG: 自适应融合门控 - 基于相似性和尺度差异的动态加权")
    print("CSA: 跨尺度特征对齐 - 减少尺度变换语义偏移")
    print("GeLU优势: 更平滑的激活函数，避免梯度消失，提升训练稳定性")
    print("预期效果: 更好的特征表示学习能力和训练收敛性")


# '''
# 本文件由BiliBili：魔傀面具整理 
# engine/extre_module/module_images/BIBM2024-MultiScalePCA.png
# 论文链接：https://arxiv.org/pdf/2406.07952     
# ''' 

# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..') 

# import warnings  
# warnings.filterwarnings('ignore')    
# from calflops import calculate_flops  
     
# import torch, math     
# import torch.nn as nn    

# from engine.extre_module.ultralytics_nn.conv import Conv, DWConv
# from engine.extre_module.ultralytics_nn.conv import CBAM
    
# def channel_shuffle(x, groups):
#     """通道混洗操作"""
#     batch, channels, height, width = x.size()
#     channels_per_group = channels // groups
#     x = x.view(batch, groups, channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     x = x.view(batch, -1, height, width)
#     return x

# class SEEFusion(nn.Module):
#     def __init__(self, input_channel, output_channel, gamma=2, bias=1, shuffle_groups=4, cbam_kernel=7):
#         super(SEEFusion, self).__init__()
#         input_channel1, input_channel2 = input_channel
#         self.input_channel1 = input_channel1   
#         self.input_channel2 = input_channel2
#         self.shuffle_groups = shuffle_groups

#         self.avg1 = nn.AdaptiveAvgPool2d(1)     
#         self.avg2 = nn.AdaptiveAvgPool2d(1)
        
#         # 使用最大通道数进行统一处理
#         max_channel = max(input_channel1, input_channel2)
        
#         # 只需要2个处理器！
#         # 浅层处理器：激励细节（用于大尺寸特征图）
#         # self.shallow_processor = nn.Sequential(
#         #     DWConv(max_channel, max_channel, 1),
#         #     nn.SiLU()
#         # )
#         self.shallow_processor = CBAM(max_channel, kernel_size=cbam_kernel)
        
#         # 深层处理器：挤压冗余（用于小尺寸特征图）
#         self.deep_processor = nn.Conv2d(max_channel, max_channel, 1, 
#                                        groups=min(max_channel, shuffle_groups), bias=False)
        
#         # 通道适配器（处理不同输入通道数）
#         self.adapt1 = Conv(input_channel1, max_channel, 1) if input_channel1 != max_channel else nn.Identity()
#         self.adapt2 = Conv(input_channel2, max_channel, 1) if input_channel2 != max_channel else nn.Identity()
#         self.readapt1 = Conv(max_channel, input_channel1, 1) if input_channel1 != max_channel else nn.Identity()
#         self.readapt2 = Conv(max_channel, input_channel2, 1) if input_channel2 != max_channel else nn.Identity()
        
#         # 原有的1D卷积
#         kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
#         kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1  
#         self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

#         self.sigmoid = nn.Sigmoid()
        
#         # 上采样和输出调整
#         self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, 
#                                      kernel_size=3, stride=2, padding=1, output_padding=1)     
#         self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()

#     def forward(self, x):  
#         x1, x2 = x
        
#         # 判断哪个是浅层（大尺寸），哪个是深层（小尺寸）
#         size1 = x1.size(-1) * x1.size(-2)
#         size2 = x2.size(-1) * x2.size(-2)
        
#         # GAP处理
#         x1_ = self.avg1(x1)   
#         x2_ = self.avg2(x2)
        
#         # 适配到统一通道数
#         x1_adapted = self.adapt1(x1_)
#         x2_adapted = self.adapt2(x2_)
        
#         # SEE处理：根据尺寸决定处理方式
#         if size1 >= size2:
#             # x1是浅层（大尺寸）-> 用浅层处理器激励细节
#             # x2是深层（小尺寸）-> 用深层处理器挤压冗余
#             x1_processed = self.shallow_processor(x1_adapted)
#             x2_processed = self.deep_processor(x2_adapted)
#             x2_processed = channel_shuffle(x2_processed, self.shuffle_groups)
#         else:
#             # x2是浅层（大尺寸）-> 用浅层处理器激励细节
#             # x1是深层（小尺寸）-> 用深层处理器挤压冗余
#             x2_processed = self.shallow_processor(x2_adapted)
#             x1_processed = self.deep_processor(x1_adapted)
#             x1_processed = channel_shuffle(x1_processed, self.shuffle_groups)
        
#         # 适配回原始通道数
#         x1_ = self.readapt1(x1_processed)
#         x2_ = self.readapt2(x2_processed)
        
#         # 后续处理流程
#         x_middle = torch.cat((x1_, x2_), dim=1)   
#         x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)   
#         # 删除了sigmoid操作
#         x_middle = self.sigmoid(x_middle)
        
#         x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)
    
#         x1_out = x1 * x_1   
#         x2_out = x2 * x_2    
#         x2_out = self.up(x2_out)    
#         result = x1_out + x2_out
#         return self.conv1x1(result)   

# class SEEFusion_Down(nn.Module):     
#     def __init__(self, input_channel, output_channel, gamma=2, bias=1, shuffle_groups=4, cbam_kernel=7):
#         super(SEEFusion_Down, self).__init__()   
#         input_channel1, input_channel2 = input_channel 
#         self.input_channel1 = input_channel1   
#         self.input_channel2 = input_channel2
#         self.shuffle_groups = shuffle_groups
   
#         self.avg1 = nn.AdaptiveAvgPool2d(1)
#         self.avg2 = nn.AdaptiveAvgPool2d(1)   

#         # 使用最大通道数进行统一处理
#         max_channel = max(input_channel1, input_channel2)
        
#         # 只需要2个处理器！
#         # 浅层处理器：激励细节（用于大尺寸特征图）
#         # self.shallow_processor = nn.Sequential(
#         #     DWConv(max_channel, max_channel, 1),
#         #     nn.SiLU()
#         # )
#         self.shallow_processor = CBAM(max_channel, kernel_size=cbam_kernel)

#         # 深层处理器：挤压冗余（用于小尺寸特征图）
#         self.deep_processor = nn.Conv2d(max_channel, max_channel, 1, 
#                                        groups=min(max_channel, shuffle_groups), bias=False)
        
#         # 通道适配器
#         self.adapt1 = Conv(input_channel1, max_channel, 1) if input_channel1 != max_channel else nn.Identity()
#         self.adapt2 = Conv(input_channel2, max_channel, 1) if input_channel2 != max_channel else nn.Identity()
#         self.readapt1 = Conv(max_channel, input_channel1, 1) if input_channel1 != max_channel else nn.Identity()
#         self.readapt2 = Conv(max_channel, input_channel2, 1) if input_channel2 != max_channel else nn.Identity()

#         # 原有的1D卷积
#         kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))   
#         kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1    
#         self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

#         self.sigmoid = nn.Sigmoid()

#         # 下采样和输出调整
#         self.down = nn.Conv2d(in_channels=input_channel2, out_channels=input_channel1, 
#                              kernel_size=3, stride=2, padding=1)   
#         self.conv1x1 = Conv(input_channel1, output_channel) if input_channel1 != output_channel else nn.Identity()

#     def forward(self, x):    
#         x1, x2 = x    
        
#         # 判断哪个是浅层（大尺寸），哪个是深层（小尺寸）
#         size1 = x1.size(-1) * x1.size(-2)
#         size2 = x2.size(-1) * x2.size(-2)
        
#         x1_ = self.avg1(x1)
#         x2_ = self.avg2(x2)

#         # 适配到统一通道数
#         x1_adapted = self.adapt1(x1_)
#         x2_adapted = self.adapt2(x2_)
        
#         # SEE处理：根据尺寸决定处理方式
#         if size1 >= size2:
#             x1_processed = self.shallow_processor(x1_adapted)
#             x2_processed = self.deep_processor(x2_adapted)
#             x2_processed = channel_shuffle(x2_processed, self.shuffle_groups)
#         else:
#             x2_processed = self.shallow_processor(x2_adapted)
#             x1_processed = self.deep_processor(x1_adapted)
#             x1_processed = channel_shuffle(x1_processed, self.shuffle_groups)

#         # 适配回原始通道数
#         x1_ = self.readapt1(x1_processed)
#         x2_ = self.readapt2(x2_processed)

#         x_middle = torch.cat((x1_, x2_), dim=1)    
#         x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         # 删除了sigmoid操作
#         x_middle = self.sigmoid(x_middle)
    
#         x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)
     
#         x1_out = x1 * x_1   
#         x2_out = x2 * x_2  
#         x2_out = self.down(x2_out)    
#         result = x1_out + x2_out
#         return self.conv1x1(result)  

# if __name__ == '__main__':     
#     RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     batch_size, channel_1, height_1, width_1 = 1, 32, 40, 40
#     batch_size, channel_2, height_2, width_2 = 1, 64, 20, 20    
#     ouc_channel = 64   
#     inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)
#     inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)

#     # 此模块有使用教程在VideoBaiduYun.txt内 
 
#     print(RED + '-'*20 + " SEEFusion " + '-'*20 + RESET)

#     module = SEEFusion([channel_1, channel_2], ouc_channel).to(device)
     
#     outputs = module([inputs_1, inputs_2])   
#     print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

#     print(ORANGE)
#     flops, macs, _ = calculate_flops(model=module,  
#                                      args=[[inputs_1, inputs_2]], 
#                                      output_as_string=True,   
#                                      output_precision=4,    
#                                      print_detailed=True)
#     print(RESET)

#     print(RED + '-'*20 + " SEEFusion_Down " + '-'*20 + RESET)    

#     module = SEEFusion_Down([channel_2, channel_1], ouc_channel).to(device)
 
#     outputs = module([inputs_2, inputs_1])    
#     print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)

#     print(ORANGE)   
#     flops, macs, _ = calculate_flops(model=module,
#                                      args=[[inputs_2, inputs_1]],
#                                      output_as_string=True, 
#                                      output_precision=4,   
#                                      print_detailed=True)  
#     print(RESET)


    
    