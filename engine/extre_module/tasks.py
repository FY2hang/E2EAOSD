import os, sys, re, yaml, contextlib  
from pathlib import Path     
from functools import partial
from ..core import register
from ..misc.dist_utils import Multiprocess_sync, is_dist_available_and_initialized
from ..backbone.common import FrozenBatchNorm2d

import torch   
import torch.nn as nn  

from engine.backbone.hgnetv2 import StemBlock, HG_Stage  
from engine.deim.hybrid_encoder import RepNCSPELAN4, ConvNormLayer_fuse, SCDown, CSPLayer, TransformerEncoderBlock
from engine.deim.dfine_decoder import DFINETransformer

from engine.extre_module.ultralytics_nn.conv import Concat, Add, Conv
from engine.extre_module.ultralytics_nn.block import Bottleneck, C3_Block, C2f_Block, C3k2_Block, MetaFormer_Block, MetaFormer_Mona, MetaFormer_SEFN, MetaFormer_Mona_SEFN, NCHW2NLC2NCHW

from engine.extre_module.custom_nn.attention.CDFA import ContrastDrivenFeatureAggregation
from engine.extre_module.custom_nn.attention.SEAttention import SEAttention
from engine.extre_module.custom_nn.conv_module.psconv import PSConv    
from engine.extre_module.custom_nn.module.IDWB import InceptionDWBlock
from engine.extre_module.custom_nn.block.RepHMS import RepHMS
from engine.extre_module.custom_nn.block.MANet import MANet
from engine.extre_module.custom_nn.neck_module.HyperCompute import HyperComputeModule  
from engine.extre_module.custom_nn.transformer.DAttention import DAttention
from engine.extre_module.custom_nn.transformer.AdaptiveSparseSA import AdaptiveSparseSA
from engine.extre_module.custom_nn.transformer.PolaLinearAttention import PolaLinearAttention
from engine.extre_module.custom_nn.transformer.CTA import Channel_Transposed_Attention  
from engine.extre_module.custom_nn.transformer.DPBAttention import DPB_Attention
from engine.extre_module.custom_nn.transformer.biformer import BiLevelRoutingAttention
from engine.extre_module.custom_nn.neck_module.HyperACE import HyperACE, FullPAD_Tunnel, DownsampleConv
from engine.extre_module.custom_nn.neck_module.GoldYOLO import *
from engine.extre_module.custom_nn.neck_module.HS_FPN import *
from engine.extre_module.custom_nn.featurefusion.mfm import MFM
from engine.extre_module.custom_nn.featurefusion.mpca import MultiScalePCA, MultiScalePCA_Down
from engine.extre_module.custom_nn.featurefusion.PST import PST
from engine.extre_module.custom_nn.featurefusion.msga import MultiScaleGatedAttn
from engine.extre_module.custom_nn.featurefusion.wfu import WFU
from engine.extre_module.custom_nn.featurefusion.cgfm import ContextGuideFusionModule, SEAttention
from engine.extre_module.custom_nn.featurefusion.emfm import EMFM
from engine.extre_module.custom_nn.featurefusion.seefusion import SEEFusion, SEEFusion_Down
from engine.extre_module.custom_nn.featurefusion.DPCF import DPCF
from engine.extre_module.custom_nn.featurefusion.SCEF import SCEF
from engine.extre_module.custom_nn.neck.FDPN import FocusFeature, ADown     
from engine.extre_module.custom_nn.mlp.ConvolutionalGLU import ConvolutionalGLU
from engine.extre_module.custom_nn.module.nnWNet import *  
from engine.extre_module.custom_nn.module.ARF import ARF   
from engine.extre_module.custom_nn.neck.BDPN import BDBlock, ADown
from engine.extre_module.custom_nn.module.fasterblock import Faster_Block
from engine.extre_module.custom_nn.module.MSCB import MSCB
from engine.extre_module.custom_nn.conv_module.CVPR2023_PartialConv import PartialConv
from engine.extre_module.custom_nn.stem.LoG import LoGStem
from engine.extre_module.custom_nn.stem.SRFD import SRFD
from engine.extre_module.custom_nn.block.rgcspelan import RGCSPELAN
from engine.extre_module.custom_nn.block.psnet import PFE_Net, EnhancedPFE_Net
    
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 

__all__ = ['DEIM_MG']

@register(force=True) # 避免因为导入导致的多次注册
class DEIM_MG(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']
    def __init__(self, \
        yaml_path,  
        pretrained=None,    
        freeze_stem_only=False,     
        freeze_at=-1,    
        freeze_norm=False,     
        num_classes=80,   
        eval_spatial_size=(640, 640)   
    ):
        super().__init__() 
        d = yaml_load(yaml_path)  
        backbone, encoder, decoder, self.save = parse_model(d, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size, verbose=True) 
        self.backbone = backbone    
        self.encoder = encoder    
        self.decoder = decoder   
 
        # print(self.backbone.state_dict().keys())
  
        if freeze_at >= 0:
            self._freeze_parameters(self.backbone[0])
            if not freeze_stem_only:     
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i + 1])
  
        if freeze_norm:     
            self._freeze_norm(self.backbone)
 
        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:     
                state = torch.load(pretrained, map_location='cpu')   
                print(f"Loaded stage1 {pretrained} HGNetV2 from local file.") 

                # need_keep_key_prefix_list = ['0.*', '1.*'] 
                # need_pop_key_list = [] 
                # for key in state.keys():
                #     need_pop = True
                #     for keep in need_keep_key_prefix_list:     
                #         if re.match(keep, key):   
                #             need_pop = False   
                #             break   
                #     if need_pop:   
                #         need_pop_key_list.append(key) 
                # for key in need_pop_key_list:   
                #     state.pop(key)
     
                print(RED + f'Loading Pretrained State Dict Key Names:{state.keys()}' + RESET)    
                    
                self.backbone.load_state_dict(state, strict=False)   
     
            except (Exception, KeyboardInterrupt) as e:
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()):
                    print(f"Loading Backbone Pretrained Weight Error. Message:{str(e)}") 
                exit()
    
    def forward(self, x, targets=None):    
        y = []     
        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):   
            # print(idx, m.f, m.i)   
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers     
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output   
   
        x = self.decoder([y[j] for j in self.decoder.f], targets)
        return x  

    def deploy(self, ):  
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()    
        return self
   
    def _freeze_norm(self, m: nn.Module):     
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:    
            for name, child in m.named_children():
                _child = self._freeze_norm(child)     
                if _child is not child:     
                    setattr(m, name, _child) 
        return m
    
    def _freeze_parameters(self, m: nn.Module):  
        for p in m.parameters():
            p.requires_grad = False   

def yaml_load(file="data.yaml", append_filename=False):     
    """
    Load YAML data from a file.
  
    Args:   
        file (str, optional): File name. Default is 'data.yaml'.     
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"     
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
   
        # Remove special characters
        if not s.isprintable(): 
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
     
        # Add YAML filename to dict and return     
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file) 
        return data 
    
def parse_module(d, i, f, m, args, ch, nc=None, eval_spatial_size=None): 
    import ast
    try:    
        if m == 'node_mode':
            m = d[m] 
            if len(args) > 0:
                if args[0] == 'head_channel':
                    args[0] = int(d[args[0]])
        t = m  
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
    except:
        pass  
 
    selfatt, selfatt_args_index = None, -1    
    if type(args) is list:
        for j, a in enumerate(args):
            if isinstance(a, str):   
                with contextlib.suppress(ValueError):
                    try:    
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                    except:
                        args[j] = a 
            elif type(a) is dict:
                if 'module' in a:
                    module_ = a['module']     
                    try:   
                        module_ = getattr(torch.nn, module_[3:]) if 'nn.' in module_ else globals()[module_]
                    except Exception as e:
                        raise Exception(f'{module_} is maybe not import in task.py, please check. {e}')
                    module_param = a.get('param', {})  
                    for k in module_param:  
                        p = module_param[k] 
                        try:
                            module_param[k] = locals()[p] if p in locals() else (getattr(torch.nn, p[3:]) if 'nn.' in p else ast.literal_eval(p))
                        except:
                            module_param[k] = p   
                    args[j] = partial(module_, **module_param)   
                if 'selfatt' in a:     
                    selfatt = a['selfatt'] 
                    selfatt_args_index = j
    if selfatt_args_index != -1:
        args.pop(j)   
    
    c2 = ch[-1] 
    if m in {StemBlock, HG_Stage}:
        c1, cmid, c2 = ch[f], args[0], args[1]
        args = [c1, cmid, c2, *args[2:]]
    elif m in {RepNCSPELAN4, CSPLayer, ConvNormLayer_fuse, SCDown}:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {TransformerEncoderBlock}:
        c2 = ch[f]     
        args = [c2, *args]     
    elif m is Concat: 
        c2 = sum(ch[x] for x in f)  
    elif m in {ContrastDrivenFeatureAggregation, DownsampleConv, SEAttention}: # attention   
        c2 = ch[f]   
        args = [c2, *args]
    elif m in {PSConv, ADown, Conv}: # Conv   
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {InceptionDWBlock, GlobalBlock, LocalBlock, OPE}: # module
        c1, c2 = ch[f], args[0]    
        args = [c1, c2, *args[1:]]    
    elif m in {RepHMS, MANet, C3_Block, C2f_Block, C3k2_Block, MetaFormer_Block, MetaFormer_Mona, MetaFormer_SEFN, MetaFormer_Mona_SEFN, PFE_Net}: # Block
        c1, c2 = ch[f], args[0]
        if selfatt is not None:
            args = ([c1, c2, *args[1:]], {'selfatt':selfatt})
        else:
            args = [c1, c2, *args[1:]]   
    elif m is NCHW2NLC2NCHW:
        c2 = ch[f]
        args = [c2, *args]
    elif m in {HyperComputeModule}:    
        c2 = ch[f]  
        args = [c2, *args]  
    elif m is HyperACE:
        c1 = ch[f[1]]
        c2 = args[0]     
        args = [c1, c2, *args[1:]]   
    elif m is FullPAD_Tunnel:  
        c2 = ch[f[0]]  
    elif m in {MFM, MultiScalePCA, MultiScalePCA_Down, MultiScaleGatedAttn, WFU, ContextGuideFusionModule, EMFM, SEEFusion, SEEFusion_Down, DPCF, SCEF}: # featurefusion     
        c1 = [ch[i] for i in f]   
        c2 = args[0]
        args = [c1, c2, *args[1:]]
    # elif m is SCEF: # 针对特征融合模块
      
    #     # 1. 提取 SCEF 初始化所需的输入通道数列表（in_features）
    #     # f 是输入特征层的索引（如 [0, 1]），ch 是各层通道数列表 → c1 是 [ch[0], ch[1]]（如 [32, 64]）
    #     in_features = [ch[i] for i in f]   
        
    #     # 2. 提取 SCEF 初始化所需的输出通道数（out_features）
    #     # 假设 args[0] 原本就是配置中指定的输出通道数（如 64），直接取即可
    #     out_features = args[0]
        
    #     # 3. 重组 args：仅保留 SCEF __init__ 所需的参数（in_features, out_features + 其他自定义参数）
    #     # 若 SCEF 有额外初始化参数（如自定义超参），则保留 args[1:]，否则可简化为 [in_features, out_features]
    #     args = [in_features, out_features, *args[1:]]
        
    #     # 4. （关键）更新 ch 列表（模型通道数记录）：SCEF 输出通道数 = out_features
    #     # 因为后续层会依赖当前层的输出通道数，必须同步更新 ch
    #     ch.append(out_features)
    # elif m is WFU:  # WFU单独处理
    #     if len(f) != 2:
    #         raise ValueError(f"WFU requires exactly 2 input features, got {len(f)}")
        
    #     ch1, ch2 = ch[f[0]], ch[f[1]]
        
    #     # 如果需要交换顺序（让较大的在前面）
    #     # 根据你的网络结构判断哪个应该是x_big
    #     if f[0] < f[1]:  # 如果第一个索引更小，可能对应更大的特征图
    #         c1 = [ch1, ch2]  # 正常顺序
    #     else:
    #         c1 = [ch2, ch1]  # 交换顺序
        
    #     c2 = args[0]
    #     args = [c1, c2, *args[1:]]
    elif m is PST:  # PST单独处理
    # 假设f包含两个索引：[当前层索引, 上层索引]
        c1 = ch[f[0]] if len(f) > 0 else ch[-1]
        c_up = ch[f[1]] if len(f) > 1 else ch[-2] 
        c2 = args[0] if len(args) > 0 else c1
        args = [c1, c_up, c2, *args[1:]]
    elif m is FocusFeature:  
        c1 = [ch[i] for i in f]   
        c2 = c1[1]
        args = [c1, *args]
    # --------------GOLD-YOLO--------------   
    elif m in {SimFusion_4in, AdvPoolFusion}: 
        c2 = sum(ch[x] for x in f)     
    elif m is SimFusion_3in:    
        c1 = [ch[x] for x in f]  
        c2 = args[0]
        args = [c1, c2]
    elif m in {IFM, TopBasicLayer}:
        c1 = ch[f]
        c2 = sum(args[0])
        args = [c1, *args]
    elif m in {InjectionMultiSum_Auto_pool, InjectionMultiSum_Auto_pool_Fourier}:   
        c1 = ch[f[0]]
        c2 = args[0] 
        args = [c1, c2, *args[1:]]  
    elif m is PyramidPoolAgg:
        c1 = sum(ch[x] for x in f)   
        c2 = args[0]   
        args = [c1, c2, *args[1:]] 
    # --------------GOLD-YOLO--------------    
    elif m is Add:
        c2 = ch[f[-1]]
    elif m is HFP: 
        c2 = ch[f]
        args = [c2, *args]  
    elif m is SDP:
        c1 = [ch[x] for x in f]    
        c2 = args[0]
        args = [c1, c2, *args[1:]]
    elif m is DFINETransformer:
        args["feat_channels"] = [ch[x] for x in f] 
        args["num_classes"] = nc
        args["eval_spatial_size"] = eval_spatial_size
    else:  
        c2 = ch[f]
    
    if type(args) is dict: 
        m_ = m(**args)
    elif type(args) is list: 
        m_ = m(*args)  # module
    else:
        m_ = m(*args[0], **args[1])
    t = str(m)[8:-2].replace('__main__.', '')  # module type
    m_.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
 
    return m_, c2, t, args

def parse_model(d, ch, nc, eval_spatial_size, verbose=True):   
    if verbose:
        print(ORANGE + f"{'':>3}{'from':>10}{'params':>10}  {'module':<60}{'arguments':<30}" + RESET) 
    layer_index, ch = 0, [ch]   
    backbone_layers, encoder_layers, decoder_model, save, c2 = [], [], None, [], ch[-1]  # layers, savelist, ch out
     
    if verbose:  
        print(BLUE + "-"*40 + "BackBone" + "-"*40 + RESET)
    for f, m, args in d["backbone"]:  
        
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)

        if verbose:     
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print 
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist     
        backbone_layers.append(m_)
        if layer_index == 0:  
            ch = [] 
        ch.append(c2)  

        layer_index += 1
 
    if verbose:     
        print(BLUE + "-"*40 + "Enchoder" + "-"*40 + RESET)
    for f, m, args in d["encoder"]: 
        
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch)   

        if verbose:
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        encoder_layers.append(m_)
        ch.append(c2)   
   
        layer_index += 1
 
    if verbose: 
        print(BLUE + "-"*40 + "Decoder" + "-"*40 + RESET)
    for f, m, args in d["decoder"]:   
        
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, nc, eval_spatial_size)

        if verbose:     
            print(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
  
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 
        decoder_model = m_
        ch.append(c2)
    
    # print(ch)   
    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(save)   
