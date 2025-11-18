"""ResNet variant: Replace stride=2 conv in Bottleneck with ADownGatedV3.

This implementation directly replaces the stride-2 conv3x3 in the first bottleneck
of each stage with ADownGatedV3, similar to YOLO's deployment strategy.

Key differences from ResNetADownStages:
- ResNetADownStages: Insert ADown BEFORE stage, keep original Bottleneck
- ResNetADownReplace: Replace conv3x3 IN Bottleneck with ADown
- Result: Lower parameter count (like YOLO)

Author: magic524
Date: 2025-11-08
"""

from typing import Optional

import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmdet.registry import MODELS
from .resnet import ResNet, Bottleneck as _Bottleneck
from ..layers.adown import ADownGatedV3


class BottleneckADown(_Bottleneck):
    """Bottleneck block with ADownGatedV3 replacing stride=2 conv3x3.
    
    Standard Bottleneck:
        Conv1x1 -> Conv3x3(stride=2) -> Conv1x1
    
    BottleneckADown:
        Conv1x1 -> ADownGatedV3 -> Conv1x1
    
    This reduces parameters while improving anti-aliasing.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                 use_adown=False,
                 adown_cfg=None):
        """
        Args:
            use_adown (bool): Whether to use ADown to replace conv2
            adown_cfg (dict): Config for ADownGatedV3
        """
        # Initialize parent without calling build_conv_layer for conv2 yet
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=1,  # We handle stride in ADown or conv2
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            plugins=plugins,
            init_cfg=init_cfg
        )
        
        self.use_adown = use_adown
        self.original_stride = stride
        
        if use_adown and stride == 2:
            # Replace conv2 with ADownGatedV3
            cfg = adown_cfg or {}
            
            # Remove the original conv2 and norm2
            delattr(self, 'conv2')
            delattr(self, self.norm2_name)
            
            # Create ADown with same in/out channels as conv2 would have
            self.conv2 = ADownGatedV3(
                c1=planes,
                c2=planes,
                ks=cfg.get('ks', 3),
                use_blur=cfg.get('use_blur', True),
                gate_temp=cfg.get('gate_temp', 1.0),
                use_at=cfg.get('use_at', True),
                use_fuse=cfg.get('use_fuse', False),
                learnable_temp=cfg.get('learnable_temp', False),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'),
                pre_smooth=cfg.get('pre_smooth', False),
            )
            
            # ADown has built-in norm, so we create a dummy norm2
            # to keep the interface compatible
            self.norm2 = nn.Identity()
        else:
            # Keep original conv2 with original stride
            self.stride = stride


@MODELS.register_module()
class ResNetADownReplace(ResNet):
    """ResNet with ADownGatedV3 replacing stride=2 conv in Bottleneck.
    
    This is similar to YOLO's ADown deployment:
    - Directly replaces stride-2 convolutions
    - Reduces parameter count
    - Improves anti-aliasing
    
    Strategy:
    - Keep Stem unchanged (preserve ImageNet alignment)
    - Replace stride=2 conv3x3 in first Bottleneck of Layer2/3/4
    - Use YOLO11-V13 inspired parameters per layer
    
    Default config per stage:
        adown_stage_cfg = {
            2: dict(ks=3, use_blur=True,  gate_temp=1.5, use_at=False, ...),  # C3
            3: dict(ks=3, use_blur=True,  gate_temp=1.0, use_at=True,  ...),  # C4
            4: dict(ks=3, use_blur=False, gate_temp=0.8, use_at=True,  ...),  # C5
        }
    """

    arch_settings = {
        50: (BottleneckADown, (3, 4, 6, 3)),
        101: (BottleneckADown, (3, 4, 23, 3)),
        152: (BottleneckADown, (3, 8, 36, 3))
    }

    def __init__(self, *args,
                 adown_stage_cfg: Optional[dict] = None,
                 **kwargs):
        """
        Args:
            adown_stage_cfg (dict): Per-stage config for ADown.
                Key: stage index (2, 3, 4)
                Value: dict with ADown parameters
        """
        # Default config (YOLO11-V13 inspired)
        default_cfg = {
            2: dict(ks=3, use_blur=True,  gate_temp=1.5, use_at=False, 
                    use_fuse=False, learnable_temp=False, pre_smooth=False),
            3: dict(ks=3, use_blur=True,  gate_temp=1.0, use_at=True,  
                    use_fuse=False, learnable_temp=False, pre_smooth=False),
            4: dict(ks=3, use_blur=False, gate_temp=0.8, use_at=True,  
                    use_fuse=False, learnable_temp=False, pre_smooth=False),
        }
        
        if adown_stage_cfg:
            default_cfg.update(adown_stage_cfg)
        self._adown_stage_cfg = default_cfg
        
        super().__init__(*args, **kwargs)

    def make_res_layer(self, **kwargs):
        """Override to inject ADown into first Bottleneck of each stage."""
        layer_idx = len(getattr(self, 'res_layers', [])) + 1  # 1-based
        stride = kwargs.get('stride', 1)
        block = kwargs.get('block')
        num_blocks = kwargs.get('num_blocks')
        planes = kwargs.get('planes')
        inplanes = kwargs.get('inplanes')
        
        # Check if this stage needs ADown
        use_adown = (stride == 2) and (layer_idx in self._adown_stage_cfg)
        
        if not use_adown:
            return super().make_res_layer(**kwargs)
        
        # Build stage with ADown in first block
        cfg = self._adown_stage_cfg[layer_idx]
        downsample = None
        
        # Build downsample for shortcut if needed
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    kwargs.get('conv_cfg'),
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                build_norm_layer(kwargs.get('norm_cfg'), planes * block.expansion)[1]
            )
        
        layers = []
        
        # First block with ADown
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=kwargs.get('dilation', 1),
                downsample=downsample,
                style=kwargs.get('style', 'pytorch'),
                with_cp=kwargs.get('with_cp', False),
                conv_cfg=kwargs.get('conv_cfg'),
                norm_cfg=kwargs.get('norm_cfg'),
                dcn=kwargs.get('dcn'),
                plugins=kwargs.get('plugins'),
                init_cfg=kwargs.get('init_cfg'),
                use_adown=True,
                adown_cfg=cfg
            )
        )
        
        # Remaining blocks (standard)
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=kwargs.get('dilation', 1),
                    style=kwargs.get('style', 'pytorch'),
                    with_cp=kwargs.get('with_cp', False),
                    conv_cfg=kwargs.get('conv_cfg'),
                    norm_cfg=kwargs.get('norm_cfg'),
                    dcn=kwargs.get('dcn'),
                    plugins=kwargs.get('plugins'),
                    init_cfg=kwargs.get('init_cfg'),
                    use_adown=False,
                    adown_cfg=None
                )
            )
        
        return nn.Sequential(*layers)
