"""
检查模型参数名 - 基于BaseSolver源码
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import re

from engine.core import YAMLConfig
from engine.solver import TASKS

def main():
    config_path = "configs/deim/deim_hgnetv2_n_custom.yml"
    
    try:
        cfg = YAMLConfig(config_path)
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        
        print("🚀 调用solver._setup()来初始化模型...")
        solver._setup()  # 这会创建 self.model
        
        print(f"✅ 找到模型: {type(solver.model)}")
        
        model = solver.model
        
        print("\n" + "="*60)
        print("🔥 分析优化器参数配置效果")
        print("="*60)
        
        # 优化器配置中的正则表达式
        patterns = {
            'backbone_normal': r'^(?=.*backbone)(?!.*norm|bn).*$',
            'backbone_norm': r'^(?=.*backbone)(?=.*norm|bn).*$', 
            'encoder_decoder_norm': r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
        }
        
        matched_params = {key: [] for key in patterns.keys()}
        other_params = []
        
        print("📝 扫描所有参数...")
        
        for name, param in model.named_parameters():
            matched = False
            for pattern_name, pattern in patterns.items():
                if re.match(pattern, name):
                    matched_params[pattern_name].append(name)
                    matched = True
                    break
            if not matched:
                other_params.append(name)
        
        # 显示结果
        print(f"\n🔥 Backbone普通参数 (lr=0.0004, weight_decay=0.0001): {len(matched_params['backbone_normal'])}个")
        for name in matched_params['backbone_normal'][:3]:
            print(f"   {name}")
        if len(matched_params['backbone_normal']) > 3:
            print(f"   ... 还有{len(matched_params['backbone_normal'])-3}个")
        
        print(f"\n🔥 Backbone归一化参数 (lr=0.0004, weight_decay=0): {len(matched_params['backbone_norm'])}个")
        for name in matched_params['backbone_norm'][:3]:
            print(f"   {name}")
        if len(matched_params['backbone_norm']) > 3:
            print(f"   ... 还有{len(matched_params['backbone_norm'])-3}个")
            
        print(f"\n🔥 Encoder/Decoder归一化参数 (默认lr=0.0002, weight_decay=0): {len(matched_params['encoder_decoder_norm'])}个")
        for name in matched_params['encoder_decoder_norm'][:3]:
            print(f"   {name}")
        if len(matched_params['encoder_decoder_norm']) > 3:
            print(f"   ... 还有{len(matched_params['encoder_decoder_norm'])-3}个")
        
        print(f"\n⚪ 其他参数 (默认lr=0.0002, weight_decay=0.0001): {len(other_params)}个")
        
        total_special = sum(len(v) for v in matched_params.values())
        total_params = len(list(model.named_parameters()))
        
        print(f"\n📊 总结:")
        if total_special > 0:
            print(f"✅ 特殊优化器配置影响 {total_special}/{total_params} 个参数")
            print(f"✅ 那两个0.0004的学习率配置是有效的!")
            
            backbone_total = len(matched_params['backbone_normal']) + len(matched_params['backbone_norm'])
            if backbone_total > 0:
                print(f"   - Backbone部分使用 lr=0.0004 (共{backbone_total}个参数)")
            if len(matched_params['encoder_decoder_norm']) > 0:
                print(f"   - 归一化层使用 weight_decay=0")
        else:
            print(f"❌ 特殊配置不匹配任何参数")
            print(f"❌ 那两个0.0004的配置没有用，所有参数都使用默认配置")
        
        print(f"\n📝 参数名示例 (总共{total_params}个):")
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < 15:  # 显示前15个
                print(f"   {name}")
            else:
                print(f"   ... 还有{total_params-15}个参数")
                break
        
        # 额外：显示按模块分组的参数统计
        print(f"\n🔍 按模块分组统计:")
        module_counts = {}
        for name, param in model.named_parameters():
            module = name.split('.')[0] if '.' in name else name
            module_counts[module] = module_counts.get(module, 0) + 1
        
        for module, count in sorted(module_counts.items()):
            print(f"   {module}: {count}个参数")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()