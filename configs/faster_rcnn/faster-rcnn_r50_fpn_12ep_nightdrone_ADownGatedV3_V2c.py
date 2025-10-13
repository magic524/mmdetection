_base_ = './faster-rcnn_r50_fpn_12ep_nightdrone.py'

# V2c: 专门替换 FPN 中的下采样操作为 ADownGatedV3
# 策略：在 FPN 需要生成额外 P6 层时，用 ADown 替换 MaxPool2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks import FPN
from mmdet.models.layers import ADownGatedV3

class ADownFPN(FPN):
    """FPN with ADownGatedV3 replacing downsampling operations.
    
    专门替换 FPN 中的下采样操作，主要是生成 P6 层的 MaxPool 操作。
    """
    
    def __init__(self, *args, use_adown_downsample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adown_downsample = use_adown_downsample
        
        # 如果启用 ADown 下采样，为 P6 层准备 ADown 模块
        if self.use_adown_downsample and self.num_outs > 4:  # Faster R-CNN 通常有 P6
            # P6 是从 P5 (256通道) 下采样得到的
            self.adown_p6 = ADownGatedV3(
                c1=256,  # P5 输出通道
                c2=256,  # P6 输出通道
                ks=3,
                use_blur=True,   # 抗锯齿对夜间小目标重要
                gate_temp=2.0,
                use_at=True,     # ECA注意力
                use_fuse=True,   # 特征融合
                learnable_temp=True,
                pre_smooth=False
            )
    
    def forward(self, inputs):
        """Forward function with ADown downsampling."""
        assert len(inputs) == len(self.in_channels)

        # build laterals - 完全保持原始 FPN 逻辑
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path - 完全保持原始 FPN 逻辑
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels - 保持原始逻辑
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        
        # part 2: add extra levels - 这里替换下采样操作
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # 原本用 max_pool2d，现在用 ADown 替换第一个额外层的生成
                for i in range(self.num_outs - used_backbone_levels):
                    if i == 0 and self.use_adown_downsample and hasattr(self, 'adown_p6'):
                        # 用 ADown 生成 P6
                        outs.append(self.adown_p6(outs[-1]))
                    else:
                        # 后续层仍用 max_pool（如果有P7等）
                        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # add conv layers on top of original feature maps (RetinaNet)
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

# 注册自定义 FPN
from mmdet.registry import MODELS
MODELS.register_module()(ADownFPN)

# 使用 ADown 替换 FPN 下采样操作，backbone 保持预训练权重
model = dict(
    neck=dict(
        type='ADownFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        use_adown_downsample=True,  # 启用 ADown 下采样替换
    ),
    # 修正类别数量为 NightDrone 的 8 个类别
    roi_head=dict(
        bbox_head=dict(num_classes=8)
    )
)

# 输出路径调整
val_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2c/val'
)

test_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2c/test'
)