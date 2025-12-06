"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""
import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Windows使用这个来指定显卡 多卡的话假设我用的是第一第三第四第五张卡就是"0,2,3,4""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import argparse

from engine.logger_module import get_logger
from engine.extre_module.torch_utils import check_cuda
from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)
debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    check_cuda()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    cfg_str = json.dumps(cfg.__dict__, indent=4, ensure_ascii=False)
    print(GREEN + cfg_str + RESET)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        if args.path:
            solver.val_onnx_engine()
        else:
            solver.val()
    else:
        solver.fit(cfg_str)

    # dist_utils.cleanup()
    # if args.test_only:
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import cv2
    #     import torch
        
    #     # =======================================================
    #     # 1. 初始化与加载 (和之前一样)
    #     # =======================================================
    #     print("🔨 正在初始化模型...")
    #     if hasattr(solver, '_setup'):
    #         solver._setup()
    #     else:
    #         print("⚠️ 警告: 没找到 _setup，尝试直接继续...")

    #     if args.resume:
    #         print(f"📥 正在加载权重: {args.resume}")
    #         solver.load_resume_state(args.resume)
        
    #     solver.model.eval()
    #     # 确保模型在 GPU 上
    #     device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    #     solver.model.to(device)

    #     # =======================================================
    #     # 2. 定义 Hook
    #     # =======================================================
    #     feature_maps = []
    #     def heatmap_hook(module, input, output):
    #         print("⚡️ Hook 触发成功！捕获到数据。") # 调试信息
    #         if isinstance(output, (list, tuple)):
    #             feat = output[0] 
    #         else:
    #             feat = output
    #         feature_maps.append(feat.detach().cpu())

    #     # =======================================================
    #     # 3. [强力修正] 递归搜索并挂载 Hook
    #     # =======================================================
    #     target_layer = None
    #     model_ref = getattr(solver.model, 'module', solver.model)
        
    #     print("🔍 开始深度搜索 Transformer 层...")
        
    #     # 方案 A: 直接按已知路径找 (最准)
    #     # 路径通常是: model -> encoder(HybridEncoder) -> encoder(ModuleList) -> [0](TransformerEncoder)
    #     # 或者: model -> neck -> encoder(ModuleList) -> [0]
        
    #     found = False
    #     try:
    #         # 尝试路径 1: model.encoder.encoder[0]
    #         if hasattr(model_ref, 'encoder') and hasattr(model_ref.encoder, 'encoder'):
    #             # 注意：第二个 encoder 通常是 ModuleList
    #             internal_encoder = model_ref.encoder.encoder
    #             if len(internal_encoder) > 0:
    #                 target_layer = internal_encoder[0]
    #                 print("🎯 成功锁定: model.encoder.encoder[0] (TransformerEncoder)")
    #                 found = True
            
    #         # 尝试路径 2: model.neck.encoder[0]
    #         if not found and hasattr(model_ref, 'neck') and hasattr(model_ref.neck, 'encoder'):
    #             internal_encoder = model_ref.neck.encoder
    #             if len(internal_encoder) > 0:
    #                 target_layer = internal_encoder[0]
    #                 print("🎯 成功锁定: model.neck.encoder[0] (TransformerEncoder)")
    #                 found = True
                    
    #     except Exception as e:
    #         print(f"⚠️ 路径搜索报错: {e}")

    #     # 方案 B: 如果上面没找到，暴力遍历所有子模块，找名字带 'Transformer' 的
    #     if not found:
    #         print("⚠️ 标准路径未找到，尝试暴力搜索...")
    #         for name, module in model_ref.named_modules():
    #             # 找一个像是 Transformer 编码器层的模块
    #             if 'TransformerEncoder' in module.__class__.__name__:
    #                 target_layer = module
    #                 print(f"🎯 暴力搜索锁定: {name} ({module.__class__.__name__})")
    #                 found = True
    #                 break # 找到一个就停
        
    #     # 挂载
    #     if target_layer:
    #         # 移除旧的 hooks 防止重复
    #         target_layer._forward_hooks.clear()
    #         target_layer.register_forward_hook(heatmap_hook)
    #         print(f"✅ Hook 已挂载到对象: {target_layer.__class__.__name__}")
    #     else:
    #         print("❌ 彻底失败: 没能在模型中找到任何 Transformer 层。")
    #         # 打印模型的一级子模块供参考
    #         print("模型一级子模块:", [n for n, _ in model_ref.named_children()])
    #         return

    #     # =======================================================
    #     # 4. [最终修改] 绕过 DataLoader，直接读取单张图片
    #     # =======================================================
    #     print("🔄 正在准备输入图片...")
        
    #     # -------------------------------------------------------
    #     # 方式 A: 读取你电脑里的一张真实图片 (推荐，效果最好)
    #     # -------------------------------------------------------
    #     # 请把这里的 'test.jpg' 改成你目录下任意一张图片的路径
    #     image_path = '/home/featurize/zfy/dataset/SIMD_COCO/val/images/4989.jpg' 
        
    #     import os
    #     import cv2
    #     import torch
    #     import torch.nn.functional as F

    #     if os.path.exists(image_path):
    #         print(f"📸 读取本地图片: {image_path}")
    #         # 读取图片 (H, W, C)
    #         origin_img = cv2.imread(image_path)
    #         # BGR 转 RGB
    #         img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    #         # Resize 到 640x640 (模型标准输入大小)
    #         img = cv2.resize(img, (640, 640))
    #         # 归一化 [0, 255] -> [0, 1] 并转为 Tensor [C, H, W]
    #         input_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    #         # 增加 Batch 维度 -> [1, 3, 640, 640]
    #         input_tensor = input_tensor.unsqueeze(0).to(device)
        
    #     else:
    #         # -------------------------------------------------------
    #         # 方式 B: 如果找不到图片，生成一张随机噪音图 (保底方案)
    #         # -------------------------------------------------------
    #         print(f"⚠️ 没找到 {image_path}，将使用随机噪音进行测试...")
    #         print("注意：噪音图产生的特征图也会是乱的，仅用于测试代码能否跑通。")
    #         input_tensor = torch.rand(1, 3, 640, 640).to(device)

    #     # =======================================================
    #     # 5. 推理与绘图 (带形状修正版)
    #     # =======================================================
    #     try:
    #         print(f"🚀 开始推理... 输入形状: {input_tensor.shape}")
    #         with torch.no_grad():
    #             solver.model(input_tensor)
    #         print("✅ 推理完成！")
    #     except Exception as e:
    #         print(f"❌ 推理报错: {e}")

    #     # =======================================================
    #     # 6. 结果处理：关键在于 Reshape
    #     # =======================================================
    #     if len(feature_maps) > 0:
    #         print(f"🎨 成功捕获特征！开始处理形状...")
            
    #         # feat 的形状通常是 [Batch, Length, Dim] 例如 [1, 400, 256]
    #         # 或者 [Length, Batch, Dim]
    #         feat = feature_maps[0] # 取第一次 Hook 的结果
            
    #         # 1. 统一转为 numpy
    #         if isinstance(feat, torch.Tensor):
    #             feat = feat.detach().cpu().numpy()
            
    #         print(f"原始特征形状: {feat.shape}")
            
    #         # 2. 形状修正逻辑 (关键步骤！！！)
    #         # ---------------------------------------------------
    #         # 情况 A: [Batch, Length, Dim] -> Transformer 的标准输出
    #         if len(feat.shape) == 3:
    #             B, L, C = feat.shape
    #             # 假设特征图是正方形 (H = W)
    #             # L = H * W  =>  H = sqrt(L)
    #             H = int(np.sqrt(L))
    #             W = H
                
    #             if H * W != L:
    #                 # 如果不是正方形，说明可能是 [H*W + token, C] 或者其他情况
    #                 print(f"⚠️ 警告: 序列长度 {L} 不能开平方，无法自动还原为正方形。尝试强制猜测...")
    #                 # 这里可以手动指定，比如 H=20, W=20 (如果 L=400)
                
    #             print(f"🔄 检测到 Transformer 序列格式，正在还原为: [Batch, {C}, {H}, {W}]")
                
    #             # 转换步骤: 
    #             # [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
    #             feat = feat.reshape(B, H, W, C)
    #             feat = feat.transpose(0, 3, 1, 2) # 变成 [B, C, H, W]

    #         # 情况 B: 已经是 [Batch, C, H, W] -> CNN 的标准输出
    #         elif len(feat.shape) == 4:
    #             pass # 不需要动
            
    #         # ---------------------------------------------------

    #         # 取第一张图
    #         final_feat = feat[0] # [C, H, W]
    #         C, H, W = final_feat.shape
    #         print(f"最终绘图形状: {C} 通道, {H}x{W}")

    #         # 3. 挑选通道并绘图
    #         # 计算方差挑选活跃通道
    #         variances = np.var(final_feat.reshape(C, -1), axis=1)
    #         # 降序排列，取前3个
    #         top_indices = np.argsort(variances)[::-1][:3]
            
    #         plt.figure(figsize=(15, 6))
            
    #         for i, idx in enumerate(top_indices):
    #             heatmap = final_feat[idx] # [H, W]
                
    #             # 归一化
    #             heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    #             heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

    #             # Resize 回 640x640
    #             heatmap_resized = cv2.resize(heatmap_norm, (640, 640), interpolation=cv2.INTER_CUBIC)

    #             plt.subplot(1, 3, i+1)
    #             plt.imshow(heatmap_resized, cmap='jet') # 也可以试试 'viridis'
    #             plt.colorbar(fraction=0.046, pad=0.04)
    #             plt.title(f"Channel {idx} (Reshaped)")
    #             plt.axis('off')

    #         save_name = "feature_response_reshaped.png"
    #         plt.savefig(save_name, bbox_inches='tight')
    #         print(f"✅✅✅ 成功！修正后的图片已保存为: {save_name}")
    #     else:
    #         print("❌ 未收集到特征。")
    
    # now
    # if args.test_only:
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import cv2
    #     import torch
    #     import os

    #     # =======================================================
    #     # 1. 初始化模型 (保持不变)
    #     # =======================================================
    #     print("🔨 正在初始化模型...")
    #     if hasattr(solver, '_setup'):
    #         solver._setup()
    #     else:
    #         print("⚠️ 警告: 没找到 _setup，尝试直接继续...")

    #     if args.resume:
    #         print(f"📥 正在加载权重: {args.resume}")
    #         solver.load_resume_state(args.resume)
        
    #     solver.model.eval()
    #     device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    #     solver.model.to(device)

    #     # =======================================================
    #     # 2. 获取 Backbone
    #     # =======================================================
    #     backbone = None
    #     model_ref = getattr(solver.model, 'module', solver.model)
    #     if hasattr(model_ref, 'backbone'):
    #         backbone = model_ref.backbone
    #     else:
    #         print("❌ 错误: 没找到 Backbone")
    #         return

    #     # =======================================================
    #     # 3. 准备图片
    #     # =======================================================
    #     # 替换为你想要分析的图片路径
    #     image_path = '/home/featurize/zfy/dataset/VisDrone2019/VisDrone2019-DET-test-dev/images/0000272_01500_d_0000004.jpg'
        
    #     if os.path.exists(image_path):
    #         print(f"📸 读取图片: {image_path}")
    #         origin_img = cv2.imread(image_path)
    #         img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    #         img = cv2.resize(img, (640, 640))
    #         input_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    #         input_tensor = input_tensor.unsqueeze(0).to(device)
    #     else:
    #         print(f"⚠️ 图片不存在，使用随机噪音测试。")
    #         input_tensor = torch.rand(1, 3, 640, 640).to(device)

    #     # =======================================================
    #     # 4. 运行 Backbone 并提取特征
    #     # =======================================================
    #     try:
    #         print("🚀 运行 Backbone...")
    #         with torch.no_grad():
    #             outputs = backbone(input_tensor)
                
    #         # 获取最后一层特征 [1, C, H, W]
    #         if isinstance(outputs, (list, tuple)):
    #             feat_tensor = outputs[-1]
    #         elif isinstance(outputs, dict):
    #             feat_tensor = list(outputs.values())[-1]
    #         else:
    #             feat_tensor = outputs
            
    #         # 转为 numpy: [C, H, W]
    #         feat = feat_tensor[0].detach().cpu().numpy()
    #         C, H, W = feat.shape
    #         print(f"📦 获取特征图: {C}通道, {H}x{W}")

    #     except Exception as e:
    #         print(f"❌ 运行报错: {e}")
    #         return

    #     # =======================================================
    #     # 5. [核心算法] 计算每个通道的“频率得分”
    #     # =======================================================
    #     print("🧮 正在进行频域分析 (FFT)...")
        
    #     freq_scores = []
        
    #     for c in range(C):
    #         channel_data = feat[c]
            
    #         # 1. 二维傅里叶变换
    #         f = np.fft.fft2(channel_data)
    #         fshift = np.fft.fftshift(f) # 将低频移到中心
            
    #         # 2. 计算幅度谱 (Magnitude Spectrum)
    #         magnitude = np.abs(fshift)
            
    #         # 3. 定义高频掩膜 (High Frequency Mask)
    #         # 中心是低频，四周是高频。我们把中心挖掉，剩下的就是高频。
    #         rows, cols = channel_data.shape
    #         crow, ccol = rows//2 , cols//2
    #         # 掩膜半径，半径越小，包含的高频越多
    #         r = 4 
            
    #         # 计算总能量
    #         total_energy = np.sum(magnitude) + 1e-8
            
    #         # 计算中心低频能量
    #         center_energy = np.sum(magnitude[crow-r:crow+r, ccol-r:ccol+r])
            
    #         # 高频能量占比 = (总能量 - 低频能量) / 总能量
    #         high_freq_ratio = (total_energy - center_energy) / total_energy
            
    #         freq_scores.append((c, high_freq_ratio))

    #     # =======================================================
    #     # 6. 排序与筛选
    #     # =======================================================
    #     # 按高频占比从小到大排序
    #     # score 越小 -> 低频 (Low Freq)
    #     # score 越大 -> 高频 (High Freq)
    #     freq_scores.sort(key=lambda x: x[1])
        
    #     # 挑选索引
    #     low_idx = freq_scores[0][0]           # 最低频 (最平滑)
    #     mid_idx = freq_scores[len(freq_scores)//2][0] # 中频
    #     high_idx = freq_scores[-1][0]         # 最高频 (最锐利/边缘)
        
    #     selected_indices = [low_idx, mid_idx, high_idx]
    #     # labels = ["Low Channel (Background)", "Mid Channel (Texture)", "High Channel (Edges)"]
    #     labels = ["Low Channel", "Mid Channel", "High Channel"]
        
    #     print(f"✅ 筛选结果:")
    #     print(f"   Low  Idx: {low_idx} (Score: {freq_scores[0][1]:.4f})")
    #     print(f"   Mid  Idx: {mid_idx} (Score: {freq_scores[len(freq_scores)//2][1]:.4f})")
    #     print(f"   High Idx: {high_idx} (Score: {freq_scores[-1][1]:.4f})")

    #     # # =======================================================
    #     # # 7. 绘图 (完全模仿论文样式)
    #     # # =======================================================
    #     # plt.figure(figsize=(18, 6))
        
    #     # for i, (idx, label) in enumerate(zip(selected_indices, labels)):
    #     #     heatmap = feat[idx]
            
    #     #     # 归一化 (重要：论文里的图通常对比度很高)
    #     #     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
    #     #     # Resize (让低分辨率特征看起来平滑自然)
    #     #     # 使用 INTER_CUBIC 或 INTER_LANCZOS4 效果更好
    #     #     heatmap_resized = cv2.resize(heatmap_norm, (640, 640), interpolation=cv2.INTER_CUBIC)
            
    #     #     plt.subplot(1, 3, i+1)
    #     #     # 论文用的通常是 jet (蓝-红) 或 turbo
    #     #     plt.imshow(heatmap_resized, cmap='jet') 
    #     #     plt.colorbar(fraction=0.046, pad=0.04)
    #     #     # plt.title(f"{label}\nIndex: {idx}", fontsize=14)
    #     #     plt.title(f"{label}", fontsize=14)
    #     #     plt.axis('off')

    #     # save_name = "paper_style_frequency_response.png"
    #     # plt.savefig(save_name, bbox_inches='tight', dpi=150)
    #     # print(f"✅✅✅ 论文同款图已保存: {save_name}")
    #     # =======================================================
    #     # 7. 绘图 (生成三张单独的图片)
    #     # =======================================================
    #     print("🎨 正在生成三张单独的特征图...")
        
    #     for i, (idx, label) in enumerate(zip(selected_indices, labels)):
    #         heatmap = feat[idx]
            
    #         # 归一化 (重要：论文里的图通常对比度很高)
    #         heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
    #         heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
            
    #         # Resize (让低分辨率特征看起来平滑自然)
    #         heatmap_resized = cv2.resize(heatmap_norm, (640, 640), interpolation=cv2.INTER_CUBIC)
            
    #         # 创建单独的 figure 和 axes
    #         plt.figure(figsize=(6, 6)) # 单张图的尺寸可以小一些
    #         plt.imshow(heatmap_resized, cmap='jet') 
    #         plt.colorbar(fraction=0.046, pad=0.04)
    #         plt.title(f"{label}", fontsize=14) # 单独显示可以把 Index 也放上
    #         plt.axis('off')

    #         # 根据标签生成不同的文件名
    #         if "Low Channel" in label:
    #             save_name = "low_channel_feature.png"
    #         elif "Mid Channel" in label:
    #             save_name = "mid_channel_feature.png"
    #         else: # High Channel
    #             save_name = "high_channel_feature.png"
            
    #         plt.savefig(save_name, bbox_inches='tight', dpi=300)
    #         plt.close() # 关闭当前 figure，释放内存
    #         print(f"✅✅✅ '{label}' 图片已保存: {save_name}")

    #     print("\n所有特征图已单独生成。")
    #     # plt.show() # 如果不需要在显示器上弹出窗口，可以注释掉这行


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    parser.add_argument('-p', '--path', type=str, help='onnx/engine model path in test-only')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
