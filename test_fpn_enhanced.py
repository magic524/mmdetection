#!/usr/bin/env python3
"""
FPN ADown Enhanced 架构测试脚本
用于验证修复后的模型能否正常前向传播
"""

import torch
import torch.nn as nn
from mmdet.models.necks.fpn_adown_enhanced import FPN_ADown_Enhanced

def test_fpn_adown_enhanced():
    """测试增强版FPN的前向传播"""
    print("🔍 开始测试 FPN_ADown_Enhanced...")
    
    # 1. 创建模型 (使用Enhanced配置的参数)
    model = FPN_ADown_Enhanced(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        add_extra_convs=True,
        enhance_bottom_up=True,
        multi_scale_adown=False,
        adown_cfg=dict(
            ks=3,
            use_blur=True,
            gate_temp=0.8,
            use_at=True,
            use_fuse=True,
            learnable_temp=True,
            pre_smooth=True
        ),
        enhance_cfg=dict(
            enable_p2p3=False,
            enable_p3p4=True,
            enable_p4p5=True,
            adown_channels=128
        )
    )
    
    # 2. 创建模拟输入 (ResNet-50 backbone 输出)
    batch_size = 2
    inputs = [
        torch.randn(batch_size, 256, 200, 334),   # C2: /4
        torch.randn(batch_size, 512, 100, 167),   # C3: /8  
        torch.randn(batch_size, 1024, 50, 84),    # C4: /16
        torch.randn(batch_size, 2048, 25, 42),    # C5: /32
    ]
    
    print(f"📥 输入特征形状:")
    for i, inp in enumerate(inputs):
        print(f"   C{i+2}: {inp.shape}")
    
    # 3. 前向传播
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(tuple(inputs))
        
        print(f"\n✅ 前向传播成功！")
        print(f"📤 输出特征形状:")
        for i, out in enumerate(outputs):
            scale = 2**(i+2)
            print(f"   P{i+2}: {out.shape} (1/{scale})")
            
        # 4. 验证输出
        assert len(outputs) == 5, f"期望5个输出，实际得到{len(outputs)}个"
        assert all(out.shape[1] == 256 for out in outputs), "所有输出通道数应为256"
        
        print(f"\n🎉 测试通过！FPN_ADown_Enhanced 工作正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lite_config():
    """测试轻量级配置"""
    print("\n🔍 测试轻量级配置...")
    
    model = FPN_ADown_Enhanced(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        add_extra_convs=True,
        enhance_bottom_up=True,
        multi_scale_adown=False,
        adown_cfg=dict(
            ks=3,
            use_blur=True,
            gate_temp=0.7,
            use_at=False,  # 轻量级：禁用注意力
            use_fuse=True,
            learnable_temp=True,
            pre_smooth=True
        ),
        enhance_cfg=dict(
            enable_p2p3=False,
            enable_p3p4=True,
            enable_p4p5=False,  # 轻量级：减少增强
            adown_channels=96
        )
    )
    
    inputs = [
        torch.randn(1, 256, 100, 167),
        torch.randn(1, 512, 50, 84),
        torch.randn(1, 1024, 25, 42),
        torch.randn(1, 2048, 13, 21),
    ]
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(tuple(inputs))
        print("✅ 轻量级配置测试通过！")
        return True
    except Exception as e:
        print(f"❌ 轻量级配置测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 FPN ADown Enhanced 架构验证")
    print("=" * 60)
    
    # 测试标准配置
    success1 = test_fpn_adown_enhanced()
    
    # 测试轻量级配置  
    success2 = test_lite_config()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 所有测试通过！可以开始训练了")
    else:
        print("❌ 部分测试失败，需要进一步检查")
    print("=" * 60)