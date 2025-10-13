from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .resnet import ResNet
from ..layers.adown import ADownGatedV3


@MODELS.register_module()
class ResNetADownStages(ResNet):
    """ResNet variant: keep stem unchanged, replace stage downsamplings with ADownGatedV3.

    Strategy:
    - Keep original stem (conv1 + bn + relu + maxpool) to preserve ImageNet-aligned early features
      since replacing maxpool hurt accuracy in your tests.
    - For stages with stride=2 (typical: layer2, layer3, layer4 in ResNet-50), insert an
      ADownGatedV3 before the first block to perform spatial downsampling, then set the
      first block stride to 1 (so ADown handles the downsample). Channels are kept intact.

    Config knobs via `adown_stage_cfg` per stage index (1-based layer index):
        adown_stage_cfg = dict(
            2=dict(ks=3, use_blur=True, gate_temp=1.5, use_at=False, use_fuse=False, learnable_temp=False),
            3=dict(ks=3, use_blur=True, gate_temp=1.0, use_at=True,  use_fuse=False, learnable_temp=False),
            4=dict(ks=3, use_blur=False, gate_temp=0.8, use_at=True, use_fuse=False, learnable_temp=False),
        )
    If a stage is absent in the dict, it falls back to standard stride conv in the block.
    """

    def __init__(self, *args,
                 adown_stage_cfg: Optional[dict] = None,
                 **kwargs):
        self._adown_stage_cfg = adown_stage_cfg or {}
        super().__init__(*args, **kwargs)

    def make_res_layer(self, **kwargs):  # override
        """Wrap parent make_res_layer to inject ADownGatedV3 for stride stages.

        Parent kwargs include: block, inplanes, planes, num_blocks, stride, dilation,
        style, avg_down, with_cp, conv_cfg, norm_cfg, dcn, plugins, init_cfg
        """
        layer_idx = len(getattr(self, 'res_layers', [])) + 1  # 1-based
        stride = kwargs.get('stride', 1)
        inplanes = kwargs.get('inplanes')
        planes = kwargs.get('planes')
        block = kwargs.get('block')
        norm_cfg = kwargs.get('norm_cfg')

        # If this stage would downsample spatially and we have a config, insert ADown
        use_adown = (stride == 2) and (layer_idx in self._adown_stage_cfg)

        if not use_adown:
            return super().make_res_layer(**kwargs)

        # Build a small wrapper nn.Sequential: [ADownGatedV3] + [res_layer(stride=1)]
        cfg = self._adown_stage_cfg[layer_idx]
        # ADown expects equal in/out channels to preserve feature dims; here we use inplanes
        adown = ADownGatedV3(
            c1=inplanes,
            c2=inplanes,
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

        # Now create the res_layer with stride=1 since spatial downsampling is done by ADown
        kwargs2 = kwargs.copy()
        kwargs2['stride'] = 1
        res_layer = super().make_res_layer(**kwargs2)

        return nn.Sequential(adown, res_layer)
