from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .resnet import ResNet
from ..layers.adown import ADownGatedV3


@MODELS.register_module()
class ResNetADownStem(ResNet):
    """ResNet variant using ADownGatedV3 to replace stem MaxPool.

    Rationale: Minimal, stable change targeting low-light performance by
    replacing the lossy MaxPool with a gated anti-aliased downsampler.

    Args added:
        adown_cfg (dict): Configuration for ADownGatedV3. Keys mirror
            ADownGatedV3 ctor: ks, use_blur, gate_temp, use_at,
            use_fuse, learnable_temp.
    """

    def __init__(self, *args,
                 adown_cfg: Optional[dict] = None,
                 **kwargs):
        self._adown_cfg = adown_cfg or {}
        super().__init__(*args, **kwargs)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        # Build the standard stem (conv1 + bn + relu or deep stem)
        super()._make_stem_layer(in_channels, stem_channels)
        # Replace MaxPool with ADownGatedV3 while keeping the same channel dim
        # Original: conv1 stride=2 then MaxPool stride=2 -> total /4
        # Here: conv1 stride=2 then ADownGatedV3 (downsample) -> total /4
        self.maxpool = ADownGatedV3(
            c1=stem_channels,
            c2=stem_channels,
            ks=self._adown_cfg.get('ks', 3),
            use_blur=self._adown_cfg.get('use_blur', True),
            gate_temp=self._adown_cfg.get('gate_temp', 1.0),
            use_at=self._adown_cfg.get('use_at', True),
            use_fuse=self._adown_cfg.get('use_fuse', True),
            learnable_temp=self._adown_cfg.get('learnable_temp', False),
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'),
        )
