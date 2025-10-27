# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from ..layers import ADownGatedV3


@MODELS.register_module()
class FPN_ADown_Enhanced(BaseModule):
    r"""Enhanced Feature Pyramid Network with comprehensive ADownGatedV3 integration.

    This enhanced version applies ADownGatedV3 in multiple strategic locations:
    1. Extra layers (P6, P7, etc.) - replacing stride=2 convolutions
    2. Bottom-up enhancement path - adaptive feature refinement
    3. Multi-scale downsampling - for robust night vision feature extraction

    Key improvements for night vision:
    - Noise-robust downsampling at multiple scales
    - Adaptive gating for feature selection in low-light conditions
    - Anti-aliasing at critical downsampling points

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level. Defaults to 0.
        end_level (int): Index of the end input backbone level. Defaults to -1.
        add_extra_convs (bool | str): Whether to add extra conv layers. Defaults to True.
        enhance_bottom_up (bool): Whether to add bottom-up enhancement path 
            with ADown. Defaults to True.
        multi_scale_adown (bool): Whether to apply ADown at multiple scales 
            in lateral connections. Defaults to False.
        relu_before_extra_convs (bool): Whether to apply relu before extra conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral. Defaults to False.
        conv_cfg, norm_cfg, act_cfg: Configuration for convolution layers.
        upsample_cfg: Configuration for interpolation layer.
        adown_cfg: Configuration for ADownGatedV3 modules. Enhanced defaults for night vision.
        enhance_cfg: Configuration for bottom-up enhancement path.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = True,  # 默认启用
        enhance_bottom_up: bool = True,  # 新增：底向上增强路径
        multi_scale_adown: bool = False,  # 新增：多尺度ADown
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        adown_cfg: ConfigType = dict(
            ks=3, 
            use_blur=True, 
            gate_temp=0.8,      # 稍微降低温度，更敏感的门控
            use_at=True,        # 启用注意力
            use_fuse=True, 
            learnable_temp=True, 
            pre_smooth=True     # 启用预平滑
        ),
        enhance_cfg: ConfigType = dict(
            enable_p2p3=False,  # P2->P3 enhancement
            enable_p3p4=True,   # P3->P4 enhancement  
            enable_p4p5=True,   # P4->P5 enhancement
            adown_channels=128  # enhancement path 通道数
        ),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.enhance_bottom_up = enhance_bottom_up
        self.multi_scale_adown = multi_scale_adown
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.adown_cfg = adown_cfg.copy()
        self.enhance_cfg = enhance_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        # 1. 标准 lateral 和 fpn convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 2. Extra layers with ADownGatedV3
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels_extra = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels_extra = out_channels
                
                # 确保输入通道数为偶数
                if in_channels_extra % 2 != 0:
                    adjust_conv = ConvModule(
                        in_channels_extra,
                        in_channels_extra + 1,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)
                    adown_in_channels = in_channels_extra + 1
                    extra_fpn_conv = nn.Sequential(
                        adjust_conv,
                        ADownGatedV3(
                            adown_in_channels,
                            out_channels,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            **self.adown_cfg
                        )
                    )
                else:
                    extra_fpn_conv = ADownGatedV3(
                        in_channels_extra,
                        out_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **self.adown_cfg
                    )
                
                self.fpn_convs.append(extra_fpn_conv)

        # 3. Bottom-up enhancement path (新增)
        self.enhance_convs = nn.ModuleList()
        self.enhance_adapters = nn.ModuleList()  # 新增：通道适配层
        if self.enhance_bottom_up:
            enhance_channels = self.enhance_cfg.get('adown_channels', 128)
            
            # P2->P3 enhancement
            if self.enhance_cfg.get('enable_p2p3', False) and self.backbone_end_level >= 2:
                if out_channels % 2 != 0:
                    # 调整到偶数通道
                    adjust_conv = ConvModule(out_channels, out_channels + 1, 1, 
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)
                    enhance_conv = nn.Sequential(
                        adjust_conv,
                        ADownGatedV3(out_channels + 1, enhance_channels, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                    )
                else:
                    enhance_conv = ADownGatedV3(out_channels, enhance_channels,
                                              norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                # 添加通道适配层：enhance_channels -> out_channels
                adapter = ConvModule(enhance_channels, out_channels, 1, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.enhance_convs.append(enhance_conv)
                self.enhance_adapters.append(adapter)
            else:
                self.enhance_convs.append(None)
                self.enhance_adapters.append(None)
            
            # P3->P4 enhancement
            if self.enhance_cfg.get('enable_p3p4', True) and self.backbone_end_level >= 3:
                if out_channels % 2 != 0:
                    adjust_conv = ConvModule(out_channels, out_channels + 1, 1, 
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)
                    enhance_conv = nn.Sequential(
                        adjust_conv,
                        ADownGatedV3(out_channels + 1, enhance_channels, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                    )
                else:
                    enhance_conv = ADownGatedV3(out_channels, enhance_channels,
                                              norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                # 添加通道适配层
                adapter = ConvModule(enhance_channels, out_channels, 1, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.enhance_convs.append(enhance_conv)
                self.enhance_adapters.append(adapter)
            else:
                self.enhance_convs.append(None)
                self.enhance_adapters.append(None)
                
            # P4->P5 enhancement  
            if self.enhance_cfg.get('enable_p4p5', True) and self.backbone_end_level >= 4:
                if out_channels % 2 != 0:
                    adjust_conv = ConvModule(out_channels, out_channels + 1, 1, 
                                           norm_cfg=norm_cfg, act_cfg=act_cfg)
                    enhance_conv = nn.Sequential(
                        adjust_conv,
                        ADownGatedV3(out_channels + 1, enhance_channels, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                    )
                else:
                    enhance_conv = ADownGatedV3(out_channels, enhance_channels,
                                              norm_cfg=norm_cfg, act_cfg=act_cfg, **self.adown_cfg)
                # 添加通道适配层
                adapter = ConvModule(enhance_channels, out_channels, 1, 
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.enhance_convs.append(enhance_conv)
                self.enhance_adapters.append(adapter)
            else:
                self.enhance_convs.append(None)
                self.enhance_adapters.append(None)

        # 4. Multi-scale lateral ADown (可选，默认关闭)
        self.lateral_adowns = nn.ModuleList()
        if self.multi_scale_adown:
            for i in range(self.start_level, self.backbone_end_level):
                if in_channels[i] % 2 == 0 and in_channels[i] >= out_channels:
                    # 只在合适的层应用
                    lateral_adown = ADownGatedV3(
                        in_channels[i],
                        out_channels,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **self.adown_cfg
                    )
                    self.lateral_adowns.append(lateral_adown)
                else:
                    self.lateral_adowns.append(None)
        
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Enhanced forward pass with ADownGatedV3 at multiple locations."""
        assert len(inputs) == len(self.in_channels)

        # 1. Build laterals (标准或增强)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if self.multi_scale_adown and i < len(self.lateral_adowns) and self.lateral_adowns[i] is not None:
                # 使用 ADown 进行 lateral 处理
                lateral_feat = self.lateral_adowns[i](inputs[i + self.start_level])
            else:
                # 标准 lateral 处理
                lateral_feat = lateral_conv(inputs[i + self.start_level])
            laterals.append(lateral_feat)

        # 2. Build top-down path (标准FPN)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # 3. Build outputs from FPN levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # 4. Bottom-up enhancement path (新增核心功能)
        if self.enhance_bottom_up and len(self.enhance_convs) > 0:
            enhanced_feats = []
            
            # P2 增强 (如果启用)
            if len(outs) >= 1 and len(self.enhance_convs) >= 1 and self.enhance_convs[0] is not None:
                p2_enhanced_raw = self.enhance_convs[0](outs[0])  # P2 -> enhanced (128 channels)
                p2_enhanced = self.enhance_adapters[0](p2_enhanced_raw)  # 128 -> 256 channels
                enhanced_feats.append(p2_enhanced)
            else:
                enhanced_feats.append(None)
                
            # P3 增强
            if len(outs) >= 2 and len(self.enhance_convs) >= 2 and self.enhance_convs[1] is not None:
                p3_input = outs[1]
                if enhanced_feats[0] is not None:
                    # 融合来自P2的增强特征 (现在通道数匹配: 256 + 256)
                    p3_input = p3_input + F.interpolate(enhanced_feats[0], size=p3_input.shape[2:], mode='nearest')
                p3_enhanced_raw = self.enhance_convs[1](p3_input)  # -> 128 channels
                p3_enhanced = self.enhance_adapters[1](p3_enhanced_raw)  # 128 -> 256 channels
                enhanced_feats.append(p3_enhanced)
            else:
                enhanced_feats.append(None)
                
            # P4 增强
            if len(outs) >= 3 and len(self.enhance_convs) >= 3 and self.enhance_convs[2] is not None:
                p4_input = outs[2]
                if enhanced_feats[1] is not None:
                    # 融合来自P3的增强特征 (现在通道数匹配: 256 + 256)
                    p4_input = p4_input + F.interpolate(enhanced_feats[1], size=p4_input.shape[2:], mode='nearest')
                p4_enhanced_raw = self.enhance_convs[2](p4_input)  # -> 128 channels
                p4_enhanced = self.enhance_adapters[2](p4_enhanced_raw)  # 128 -> 256 channels
                enhanced_feats.append(p4_enhanced)
                
                # 将增强特征反馈到对应输出 (特征增强, 通道数匹配: 256 + 256)
                if len(outs) >= 4:  # 有P5
                    outs[3] = outs[3] + F.interpolate(p4_enhanced, size=outs[3].shape[2:], mode='nearest')

        # 5. Add extra levels with ADownGatedV3
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # 退化为标准池化 (不推荐)
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # 使用 ADownGatedV3 生成额外层
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                    
                # 第一个额外层
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                
                # 后续额外层
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)