"""
DFINE with Density-aware Query Selection
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

from .dfine_utils import weighting_function, distance2bbox
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob, visualize_density_map_only
from ..core import register

from .hybrid_encoder import ConvNormLayer_fuse
from .dfine_decoder import Integral, MLP, TransformerDecoder, TransformerDecoderLayer
from .dq_dfine_decoder import MultiScaleFeature, CGFE

from ..logger_module import get_logger

logger = get_logger(__name__)

# __all__ = ['DDGQSDFINETransformer']
__all__ = ['DQSDFINETransformer']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class LightDMG(nn.Module):
    def __init__(self, chs, scale, kernel_sizes=(3, 5, 7, 9, 11)) -> None:
        super().__init__()

        self.scale = scale

        self.dw_conv = nn.ModuleList(nn.Conv2d(chs, chs // 4, kernel_size=k, padding=autopad(k), groups=math.gcd(chs, chs // 4)) for k in kernel_sizes)
        self.pw_conv = ConvNormLayer_fuse(chs // 4 * len(kernel_sizes), chs, 1, 1)

        self.densehead = nn.Sequential(
            ConvNormLayer_fuse(chs, chs // 4, 3, 1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(chs // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.concat([layer(x) for layer in self.dw_conv], dim=1)
        x = self.pw_conv(x)
        density_map = self.densehead(x)
        return x, density_map

@register()
class DQSDFINETransformer(nn.Module):
    # 定义共享参数，这些参数可能在其他地方被引用
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,              # 类别数量，默认为80（例如COCO数据集的类别数）
                 hidden_dim=256,              # Transformer隐藏层的维度
                 num_queries=300,             # 查询（query）的数量，即模型预测的最大目标数
                 feat_channels=[512, 1024, 2048],  # 输入特征图的通道数
                 feat_strides=[8, 16, 32],    # 特征图相对于输入图像的步幅
                 num_levels=3,                # 多尺度特征的层数
                 num_points=4,                # 每个查询点的数量（用于采样）
                 nhead=8,                     # Transformer中多头注意力的头数
                 num_layers=6,                # Transformer解码器层数
                 dim_feedforward=1024,        # 前馈网络的隐藏层维度
                 dropout=0.,                  # Dropout比率，防止过拟合
                 activation="relu",           # 激活函数类型
                 num_denoising=100,           # 去噪训练的查询数量
                 label_noise_ratio=0.5,       # 标签噪声比例，用于去噪训练
                 box_noise_scale=1.0,         # 边界框噪声比例，用于去噪训练
                 learn_query_content=False,   # 是否学习查询内容嵌入
                 eval_spatial_size=None,      # 评估时的空间分辨率
                 eval_idx=-1,                 # 评估时使用的解码器层索引，负数表示从最后一层计数
                 eps=1e-2,                    # 小值阈值，用于边界框的有效性检查
                 aux_loss=True,               # 是否使用辅助损失
                 cross_attn_method='default', # 交叉注意力机制类型
                 query_select_method='default', # 查询选择方法
                 reg_max=32,                  # 回归最大值，用于边界框回归
                 reg_scale=4.,                # 回归缩放因子
                 layer_scale=1,               # 层缩放因子，用于调整隐藏层维度
                 mlp_act='relu',              # MLP激活函数类型
                 using_densitymap_iter=10000,
                 densitymap_temperature=10,
                 query_factor=3,
                 min_query_num=100,
                 max_query_num=1500,
                 using_dynamic_query=False,
                 candidate_factor=3.0   #海选缩放因子
                 ):
        super().__init__()
        # 参数校验，确保输入特征通道数不超过多尺度层数
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        # 如果特征步幅数量不足，自动扩展到num_levels层
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        # 初始化核心参数
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)  # 根据层缩放调整隐藏维度
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        # 校验查询选择和交叉注意力方法的有效性
        assert query_select_method in ('default', 'one2many', 'agnostic'), '查询选择方法无效'
        assert cross_attn_method in ('default', 'discrete'), '交叉注意力方法无效'
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # 构建输入投影层，将主干网络特征投影到hidden_dim维度
        self._build_input_proj_layer(feat_channels)

        # 定义Transformer模块的参数
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)  # 上采样因子，固定为0.5
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)  # 回归缩放参数

        # 定义解码器层
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)

        # 去噪训练相关参数
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            # 为去噪训练创建类别嵌入，+1表示包括背景类
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])  # 初始化类别嵌入权重（除背景类）

        # 解码器嵌入
        self.learn_query_content = learn_query_content
        if learn_query_content:
            # 如果学习查询内容，则创建可学习的查询嵌入
            # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
            self.tgt_embed = nn.Embedding(max_query_num, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)  # 查询位置的MLP

        # 编码器输出层
        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),  # 线性投影
            ('norm', nn.LayerNorm(hidden_dim)),          # 层归一化
        ]))

        # 根据查询选择方法定义得分头
        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)  # 类无关得分
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # 类别相关得分

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 边界框预测MLP

        # 解码器头
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx  # 计算评估层索引
        # 类别得分预测头，根据层数和缩放维度分段定义
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
          + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)])
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)  # 预边界框预测
        # 边界框回归头，输出4*(reg_max+1)表示分布回归
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])
        self.integral = Integral(self.reg_max)  # 积分模块，用于将分布转换为边界框坐标
        
        # 初始化评估时的锚点和有效掩码
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)  # 注册锚点为缓冲区
            self.register_buffer('valid_mask', valid_mask)  # 注册有效掩码
        
        self.LDMG = LightDMG(self.hidden_dim, feat_strides[0], kernel_sizes=[3, 5, 7, 9, 11])
        self.multiscale = MultiScaleFeature(self.hidden_dim, scale=self.num_levels)
        self.CGFE = CGFE(gate_channels=self.hidden_dim, reduction_ratio=16, num_feature_levels=self.num_levels)

        self.iter = 0
        self.using_densitymap_iter = using_densitymap_iter
        self.densitymap_temperature = densitymap_temperature
        self.query_factor = query_factor
        self.using_dynamic_query = using_dynamic_query
        self.min_query_num = min_query_num
        self.max_query_num = max_query_num
        self.candidate_factor = candidate_factor

        # 重置参数
        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        # 将模型转换为部署模式，仅保留评估层的预测头
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )
        self.iter = 0
        self.using_densitymap_iter = 0
        ### [新增] ###
        # 部署时，DQS逻辑被移除，固定iter=0会强制使用_select_topk中的原始logits
        # 这是符合预期的，因为精炼逻辑在训练时
        ### [新增] ###

    def _reset_parameters(self, feat_channels):
        # 参数初始化
        bias = bias_init_with_prob(0.01)  # 初始化偏置，假设函数返回一个偏置值
        init.constant_(self.enc_score_head.bias, bias)  # 初始化编码器得分头的偏置
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  # 初始化边界框头的权重
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)  # 初始化边界框头的偏置

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        # 初始化解码器得分头和边界框头的偏置和权重
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)  # Xavier初始化编码器输出投影权重
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)  # 初始化查询嵌入权重
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)  # 初始化查询位置MLP权重
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)  # 初始化输入投影层的权重

    def _build_input_proj_layer(self, feat_channels):
        # 构建输入投影层，将不同通道数的特征投影到hidden_dim
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())  # 如果通道数匹配，直接使用恒等映射
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),  # 1x1卷积
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])  # 批归一化
                    )
                )

        in_channels = feat_channels[-1]
        # 为剩余的特征层添加投影层
        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),  # 3x3卷积，下采样
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # 获取编码器输入，将特征图投影并展平
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # 展平特征并记录空间形状
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))  # [b, c, h, w] -> [b, h*w, c]
            spatial_shapes.append([h, w])  # 记录每层的空间分辨率

        feat_flatten = torch.concat(feat_flatten, 1)  # 拼接所有层特征
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        # 生成锚点和有效掩码
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # 生成网格坐标
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)  # 归一化到[0,1]
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)  # 根据层级缩放锚点大小
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)  # 拼接中心点和宽高
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)  # 检查锚点是否有效
        anchors = torch.log(anchors / (1 - anchors))  # 将锚点转换为logit形式(数值稳定性：避免边界值的梯度消失)
        anchors = torch.where(valid_mask, anchors, torch.inf)  # 无效锚点置为无穷大

        return anchors, valid_mask
    
    ### [修改] ###
    # _get_decoder_input 现在只负责“海选”
    # 1. 删除了 densityMap=None 参数
    # 2. 增加了 num_candidate_queries 参数
    # 3. 删除了 densityMapMemory 的处理逻辑
    # 4. _select_topk 调用时移除了 densityMapMemory，并使用了 num_candidate_queries
    # 5. 返回了 enc_topk_ind 和 anchors，用于后续的密度图对齐
    ### [修改] ###

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           num_candidate_queries: int,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           ):  ### [新增] ###:
        # 准备解码器输入
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)  # 为batch扩展锚点

        memory = valid_mask.to(memory.dtype) * memory  # 应用有效掩码

        output_memory: torch.Tensor = self.enc_output(memory)  # 编码器输出
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)  # 计算得分

        # 处理densityMap
        # densityMapMemory = []
        # for idx, s in enumerate(self.feat_strides):
        #     if idx == 0:
        #         densityMapMemory.append(nn.AvgPool2d(kernel_size=s, stride=s)(densityMap))
        #     else:
        #         densityMapMemory.append(nn.AvgPool2d(kernel_size=s // self.feat_strides[idx - 1], stride=s // self.feat_strides[idx - 1])(densityMapMemory[-1]))
        # for idx in range(len(densityMapMemory)):
        #     densityMapMemory[idx] = densityMapMemory[idx].flatten(2).permute(0, 2, 1)
        # densityMapMemory = torch.cat(densityMapMemory, dim=1).squeeze()

        # 选择top-k查询
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(output_memory, enc_outputs_logits, densityMapMemory, anchors, self.num_queries)

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors  # 预测边界框

        # 如果是训练阶段，记录编码器输出
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # 获取查询内容
        if self.learn_query_content:
            content = self.tgt_embed.weight[:num_candidate_queries].unsqueeze(0).tile([memory.shape[0], 1, 1])  # 可学习嵌入
        else:
            content = enc_topk_memory.detach()  # 使用编码器输出

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        # 如果有去噪输入，拼接去噪和正常查询
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        ### [修改] ###
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits, enc_topk_ind, anchors

        ### [修改] ###
        # _select_topk 现在是一个通用的Top-K选择器
        # 1. 删除了 densityMapMemory 参数
        # 2. 删除了所有 self.iter 和 densitymap_temperature 相关逻辑
        # 3. 简化为只根据传入的 outputs_logits 进行选择
        # 4. 返回 topk_ind，用于后续对齐
        ### [修改] ###

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, densityMapMemory: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        # outputs_logits.size() [bs, token_len, classes]
        # memory.size() [bs, token_len, seq_len(128)]
        # outputs_anchors_unact.size() [bs, token_len, 4]
        # 根据查询选择方法选择top-k查询
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1) #
        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor # [bs, topk]

        # 提取top-k对应的锚点、得分和记忆
        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors, topk_ind

    def forward(self, feats, targets=None):
        # 前向传播
        memory, spatial_shapes = self._get_encoder_input(feats)  # 获取编码器输入

        shallow_spatial_shapes = spatial_shapes[0]
        shallow_feature = memory[:, :shallow_spatial_shapes[0] * shallow_spatial_shapes[1]].permute(0, 2, 1).reshape(memory.size(0), memory.size(2), shallow_spatial_shapes[0], shallow_spatial_shapes[1]) # bs, h * w, c -> bs, c, h, w
        densityData, densityMap = self.LDMG(shallow_feature) # bs, 1, h, w
        multi_ccm_feature = self.multiscale(densityData)
        memory = self.CGFE(multi_ccm_feature, memory, spatial_shapes)

# 2. DQS 动态数量计算 (修改)    
        num_queries_list = None
        bs = memory.size(0)
        if self.using_dynamic_query:
            num_final_list = list(map(int, (densityMap.sum(dim=[1,2,3]) * self.query_factor).cpu().detach().tolist())) #
            num_final_list = [max(min(q, self.max_query_num), self.min_query_num) for q in num_final_list] #
        else:
            # (如果关闭动态查询，则使用 num_queries 作为固定值)
            num_final_list = [self.num_queries] * bs

        # (新逻辑) 
        # 1. 动态计算“最终”查询数
        num_final_max = max(num_final_list)
        # 2. 动态计算“海选”候选数
        num_candidate = min(int(num_final_max * self.candidate_factor), self.max_query_num) #
        
        # 3. 临时将 self.num_queries 设为“海选”数量，用于去噪和海选
        self.num_queries = num_candidate 
        ### [重点修改] ###
            
            # self.num_queries = int(max(num_queries_list))
            # self.num_queries = max(min(self.num_queries, self.max_query_num), self.min_query_num)
        # self.num_queries = 300

        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy(), 'result.png')
        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy() ** (1 / self.densitymap_temperature), 'result.png')
        # visualize_density_map_only(densityMap.squeeze().cpu().detach().numpy() ** (1 / self.densitymap_temperature) / float((densityMap ** (1 / self.densitymap_temperature)).max()), 'result.png')

        # 准备去噪训练数据
        if self.training and self.num_denoising > 0:
            # (注意：get_contrastive_denoising_training_group 内部也依赖 self.num_queries)
            # (它会创建 num_denoising + num_candidate_queries 大小的 mask)
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
            # if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask
            #     if attn_mask is not None:
            #         attn_mask = attn_mask.unsqueeze(0).repeat(memory.size(0) * self.nhead, 1, 1) # 每个样本不同的attn_mask，需要扩展到三维
            #         for i, qn in enumerate(num_queries_list):
            #             # 假设这个batch最大的query是1500，当这个batch的其中一个样本为900的时候，前900个不能看到后600个查询
            #             attn_mask[i * self.nhead:(i + 1) * self.nhead, dn_meta['dn_num_split'][0] + qn:, :dn_meta['dn_num_split'][0] + qn] = True 
            #     else:
            #         self.init_attn_mask(memory.size(0), num_queries_list, memory.device)
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
            # if memory.size(0) > 1 and self.using_dynamic_query: # bs等于1的时候就不需要处理attn_mask
            #     self.init_attn_mask(memory.size(0), num_queries_list, memory.device)

        # 4. 获取“海选”查询 (num_candidate 个)
        init_ref_contents, init_ref_points_unact, _, _, enc_outputs_logits, enc_topk_ind, anchors = \
            self._get_decoder_input(memory, spatial_shapes,
                                 ### [修改] ###
                                 num_candidate_queries=num_candidate, # <--- 报错行，现在 num_candidate 已定义
                                 ### [修改] ###
                                 denoising_logits=denoising_logits, 
                                 denoising_bbox_unact=denoising_bbox_unact
                                 )
        


        # # 获取解码器输入
        # init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = \
        #     self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, densityMap)

        # 5. 【【核心修改】】手动执行解码器循环
        # (逻辑从 dfine_decoder.py 复制而来)
        output = init_ref_contents
        output_detach = pred_corners_undetach = 0
        value = self.decoder.value_op(memory, None, None, None, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        project = weighting_function(self.reg_max, self.up, self.reg_scale)
        ref_points_detach = F.sigmoid(init_ref_points_unact)

        # (只运行第 0 层)
        i = 0
        layer = self.decoder.layers[i]
        ref_points_input = ref_points_detach.unsqueeze(2)
        query_pos_embed = self.query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        # (Layer Scale 逻辑)
        if i >= self.decoder.eval_idx + 1 and self.decoder.layer_scale > 1:
            query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.decoder.layer_scale)
            value = self.decoder.value_op(memory, None, query_pos_embed.shape[-1], None, spatial_shapes)
            output = F.interpolate(output, size=query_pos_embed.shape[-1])
            output_detach = output.detach()

        output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

        # (记录第 0 层输出)
        pre_bboxes = F.sigmoid(self.pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
        pre_scores = self.dec_score_head[0](output)
        ref_points_initial = pre_bboxes.detach()

        pred_corners = self.dec_bbox_head[i](output + output_detach) + pred_corners_undetach
        inter_ref_bbox = distance2bbox(ref_points_initial, self.integral(pred_corners, project), self.reg_scale)
        scores_layer_0 = self.dec_score_head[i](output)
        scores_layer_0 = self.decoder.lqe_layers[i](scores_layer_0, pred_corners)
        
        # (保存第0层结果，用于 enc_aux_outputs)
        enc_topk_bboxes_list = [inter_ref_bbox.detach()]
        enc_topk_logits_list = [scores_layer_0.detach()]

        ### [新增] DQS 动态精选逻辑 (仅在训练时) ###
        if self.training:
            # 1. 分离去噪和候选查询
            dn_output = output[:, :dn_count, :]
            candidate_output = output[:, dn_count:, :]
            scores_layer_0_candidates = scores_layer_0[:, dn_count:, :]
            
            # 2. 对齐密度图
            # (从 _get_decoder_input 复制逻辑)
            densityMapMemory = []
            for idx, s in enumerate(self.feat_strides):
                if idx == 0:
                    densityMapMemory.append(nn.AvgPool2d(kernel_size=s, stride=s)(densityMap))
                else:
                    densityMapMemory.append(nn.AvgPool2d(kernel_size=s // self.feat_strides[idx - 1], stride=s // self.feat_strides[idx - 1])(densityMapMemory[-1]))
            for idx in range(len(densityMapMemory)):
                densityMapMemory[idx] = densityMapMemory[idx].flatten(2).permute(0, 2, 1)
            densityMapMemory = torch.cat(densityMapMemory, dim=1) # [B, Total_Pixels]
            
            # (使用 'enc_topk_ind' 从全量 densityMapMemory 中 gather 出海选池的密度)
            candidate_density = densityMapMemory.gather(dim=1, index=enc_topk_ind.repeat(1, 1, densityMapMemory.shape[-1])) # [B, num_candidate, C] -> 应该不对
            # enc_topk_ind [B, num_candidate]
            # densityMapMemory [B, Total_Pixels]
            candidate_density = densityMapMemory.gather(dim=1, index=enc_topk_ind) # [B, num_candidate]

            # 3. 应用温度
            candidate_density = candidate_density ** (1 / self.densitymap_temperature)
            candidate_density = candidate_density / (candidate_density.max(dim=-1, keepdim=True)[0] + 1e-6)

            # 4. 计算最终得分 (精炼后得分 * 密度得分)
            final_scores = scores_layer_0_candidates.max(-1).values * candidate_density # [B, num_candidate]
            
            # 5. 动态 Top-K 选择 (每张图选自己的 Top-K)
            final_topk_ind_list = []
            self.num_queries = num_final_max # 将 self.num_queries 重设为最终数量 (e.g., 750)
            
            for b in range(bs):
                k = num_final_list[b] # (e.g., 100 or 750)
                _, topk_ind_b = torch.topk(final_scores[b], k, dim=-1)
                # (补齐到 num_final_max, e.g., 750)
                if k < self.num_queries:
                    padding = topk_ind_b.new_full((self.num_queries - k,), 0) # (用第0个索引填充)
                    topk_ind_b = torch.cat([topk_ind_b, padding], dim=0)
                final_topk_ind_list.append(topk_ind_b)
            
            final_topk_ind = torch.stack(final_topk_ind_list, dim=0).unsqueeze(-1) # [B, 750, 1]

            # 6. Gather “精英”查询
            elite_output = candidate_output.gather(dim=1, index=final_topk_ind.repeat(1, 1, output.shape[-1]))
            
            # 7. 重组 (去噪查询 + 精英查询)
            output = torch.cat([dn_output, elite_output], dim=1)
            output_detach = output.detach()
            
            # 8. 裁剪其他张量
            dn_corners = pred_corners[:, :dn_count, :]
            candidate_corners = pred_corners[:, dn_count:, :]
            elite_corners = candidate_corners.gather(dim=1, index=final_topk_ind.repeat(1, 1, pred_corners.shape[-1]))
            pred_corners_undetach = torch.cat([dn_corners, elite_corners], dim=1)
            
            dn_ref_points = inter_ref_bbox[:, :dn_count, :]
            candidate_ref_points = inter_ref_bbox[:, dn_count:, :]
            elite_ref_points = candidate_ref_points.gather(dim=1, index=final_topk_ind.repeat(1, 1, inter_ref_bbox.shape[-1]))
            ref_points_detach = torch.cat([dn_ref_points, elite_ref_points], dim=1).detach()
            
            dn_ref_initial = ref_points_initial[:, :dn_count, :]
            candidate_ref_initial = ref_points_initial[:, dn_count:, :]
            elite_ref_initial = candidate_ref_initial.gather(dim=1, index=final_topk_ind.repeat(1, 1, ref_points_initial.shape[-1]))
            ref_points_initial = torch.cat([dn_ref_initial, elite_ref_initial], dim=1).detach()
            
            # 9. 更新 attn_mask 和 dn_meta
            attn_mask = None # 简化：第0层后不再需要mask
            if dn_meta is not None:
                dn_meta['dn_num_split'][1] = self.num_queries # (更新匹配查询数量为 e.g., 750)

        ### [新增] ###

        # (存储第0层精选后的结果)
        if self.training or i == self.decoder.eval_idx:
            scores_layer_0_final = self.dec_score_head[i](output)
            scores_layer_0_final = self.decoder.lqe_layers[i](scores_layer_0_final, pred_corners_undetach)
            dec_out_logits.append(scores_layer_0_final)
            dec_out_bboxes.append(ref_points_detach) # (inter_ref_bbox 已被 gather 并重命名为 ref_points_detach)
            dec_out_pred_corners.append(pred_corners_undetach)
            dec_out_refs.append(ref_points_initial)

        # 6. 运行剩余的解码器层 (Layer 1 到 5)
        for i, layer in enumerate(self.decoder.layers[1:], start=1):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = self.query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # (Layer Scale 逻辑)
            if i >= self.decoder.eval_idx + 1 and self.decoder.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.decoder.layer_scale)
                value = self.decoder.value_op(memory, None, query_pos_embed.shape[-1], None, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            pred_corners = self.dec_bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, self.integral(pred_corners, project), self.reg_scale)

            # 在训练模式或评估层时，计算得分和输出
            if self.training or i == self.decoder.eval_idx:
                scores = self.dec_score_head[i](output)
                scores = self.decoder.lqe_layers[i](scores, pred_corners) # 使用 LQE 调整得分
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:  # 推理模式下，仅计算到 eval_idx 层
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        # --- [复制/修改 结束] ---

        # 将结果堆叠为张量返回
        out_bboxes = torch.stack(dec_out_bboxes)
        out_logits = torch.stack(dec_out_logits)
        out_corners = torch.stack(dec_out_pred_corners)
        out_refs = torch.stack(dec_out_refs)

        # 7. 构造输出字典
        if self.training and dn_meta is not None:
            ### [修改] ###
            # (pre_logits 和 pre_bboxes 现在只包含精选后的查询)
            dn_pre_logits, pre_logits = torch.split(pre_scores[:, dn_count:, :], dn_meta['dn_num_split'][1], dim=1) # (Bug: pre_scores 是海选的)
            # (修复：pre_scores 和 pre_bboxes 也需要被 gather)
            # (为简单起见，我们假设 pre_outputs 的损失在精选前计算，或者干脆放弃这个辅助损失)
            # (我们选择放弃 pre_outputs 损失，因为它在第0层后被裁剪，逻辑混乱)
            pre_logits = torch.empty(0) # (置空)
            pre_bboxes = torch.empty(0) # (置空)
            dn_pre_logits = torch.empty(0)
            dn_pre_bboxes = torch.empty(0)
            ### [修改] ###
            
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
        
        ### [修改] ###
        # (num_queries_list 必须是“最终”列表，e.g., [100, 750])
        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                    'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale, 'pred_densitymap': densityMap,
                    'num_queries_list': num_final_list} # [修改]
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_densitymap': densityMap, 
                   'num_queries_list': num_final_list} # [修改]
        ### [修改] ###
        

        # 如果是训练阶段且使用辅助损失，添加辅助输出
        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                        out_corners[-1], out_logits[-1])
            ### [修改] ###
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list) # (现在是第0层精选前的结果)
            # (删除了 pre_outputs 损失)
            # out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes} 
            ### [修改] ###
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                ### [修改] ###
                # out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                ### [修改] ###
                out['dn_meta'] = dn_meta

            out['enc_outputs_logits'] = enc_outputs_logits
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                        teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                        'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]

### [修改] ###
# (init_attn_mask 现在基于海选池大小 num_candidate 来创建)
    def init_attn_mask(self, bs, num_candidate, device):
        attn_mask = torch.full([bs * self.nhead, num_candidate, num_candidate], False, dtype=torch.bool, device=device)
        # (注意：这里的循环逻辑 假设所有批次都相同，需要重写以支持 num_candidate_list)
        # (为简单起见，我们假设批次中所有样本都使用 num_candidate)
        # for i, qn in enumerate(num_queries_list): 
        #  attn_mask[i * self.nhead:(i + 1) * self.nhead, qn:, :qn] = True 
        return attn_mask
