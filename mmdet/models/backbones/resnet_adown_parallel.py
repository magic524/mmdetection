from typing import Optional
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.layers.adown import ADownGatedV3
from mmcv.cnn import ConvModule


@MODELS.register_module()
class ResNetADownParallel(ResNet):
    """ResNet with parallel ADown branches - 保持原 stride conv，ADown 作为辅助分支"""
    
    def __init__(self, *args, adown_stage_cfg: Optional[dict] = None, **kwargs):
        self._adown_stage_cfg = adown_stage_cfg or {}
        super().__init__(*args, **kwargs)
    
    def make_res_layer(self, **kwargs):
        layer_idx = len(getattr(self, 'res_layers', [])) + 1
        stride = kwargs.get('stride', 1)
        inplanes = kwargs.get('inplanes')
        
        # 正常构建 ResNet layer
        res_layer = super().make_res_layer(**kwargs)
        
        # 如果这一层需要 ADown 辅助且通道数为偶数
        if (stride == 2 and layer_idx in self._adown_stage_cfg and inplanes % 2 == 0):
            cfg = self._adown_stage_cfg[layer_idx]
            
            # 创建 ADown 分支
            planes = kwargs.get('planes')
            block = kwargs.get('block')
            out_channels = planes * block.expansion
            
            adown_branch = ADownGatedV3(
                c1=inplanes, c2=inplanes,
                ks=cfg.get('ks', 3), use_blur=cfg.get('use_blur', False),
                gate_temp=cfg.get('gate_temp', 1.0), use_at=cfg.get('use_at', False),
                use_fuse=cfg.get('use_fuse', False), learnable_temp=cfg.get('learnable_temp', False),
                norm_cfg=kwargs.get('norm_cfg'), act_cfg=dict(type='ReLU'),
                pre_smooth=cfg.get('pre_smooth', False)
            )
            
            # 创建并行融合模块
            class ParallelFusion(nn.Module):
                def __init__(self, res_layer, adown_branch, inplanes, out_channels, norm_cfg):
                    super().__init__()
                    self.res_layer = res_layer
                    self.adown_branch = adown_branch
                    # 1x1 conv 将 ADown 输出对齐到 res_layer 输出
                    self.align_conv = ConvModule(
                        inplanes, out_channels, 1, stride=1, padding=0,
                        norm_cfg=norm_cfg, act_cfg=None
                    )
                    # 可学习的融合权重，初始时 ADown 权重很小
                    self.fusion_weight = nn.Parameter(torch.tensor(0.1))
                
                def forward(self, x):
                    # 原 ResNet 分支
                    res_out = self.res_layer(x)
                    # ADown 分支
                    adown_out = self.adown_branch(x)
                    adown_out = self.align_conv(adown_out)
                    # 加权融合
                    weight = torch.sigmoid(self.fusion_weight)
                    fused = (1 - weight) * res_out + weight * adown_out
                    return fused
            
            return ParallelFusion(res_layer, adown_branch, inplanes, out_channels, kwargs.get('norm_cfg'))
        
        return res_layer