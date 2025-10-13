import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class BlurPool(nn.Module):
    """Simple anti-aliasing blur with 3x3 kernel.

    Used before pooling to reduce aliasing artifacts on downsampling.
    """

    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("kernel", k[None, None, ...].repeat(channels, 1, 1, 1))
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.groups)


class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA).

    Lightweight channel attention without dimensionality reduction.
    """

    def __init__(self, c: int, b: int = 1, gamma: int = 2):
        super().__init__()
        k_size = int(abs((math.log(c, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ADownGatedV3(nn.Module):
    """ADownGatedV3 downsampling block.

    Contract:
    - Input: (B, C_in, H, W), C_in must be even.
    - Output: (B, C_out, H/2, W/2)
    - Typically set C_out == C_in to replace MaxPool without changing channels.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        ks: int = 3,
        use_blur: bool = True,
        gate_temp: float = 1.0,
        use_at: bool = True,
        use_fuse: bool = True,
        learnable_temp: bool = False,
        norm_cfg: Optional[dict] = dict(type="BN", requires_grad=True),
        act_cfg: Optional[dict] = dict(type="ReLU"),
        pre_smooth: bool = False,
    ):
        super().__init__()
        assert c1 % 2 == 0, "ADownGatedV3 requires even input channels"
        assert c2 % 2 == 0, "ADownGatedV3 requires even output channels"
        self.c = c2 // 2
        pad = (ks - 1) // 2
        self.pre_smooth = pre_smooth

        # branch1: conv downsample (s=2)
        self.cv1 = ConvModule(
            c1 // 2,
            self.c,
            kernel_size=ks,
            stride=2,
            padding=pad,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # branch2: 1x1 after gated pooling
        self.cv2 = ConvModule(
            c1 // 2,
            self.c,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.use_blur = use_blur
        self.blur = BlurPool(c1 // 2) if use_blur else nn.Identity()

        # local gating between {max, avg}
        self.gate = nn.Conv2d(c1 // 2, 2, kernel_size=1, stride=1, padding=0)
        self.learnable_temp = learnable_temp
        if learnable_temp:
            self._raw_temp = nn.Parameter(torch.log(torch.exp(torch.tensor(float(gate_temp))) - 1.0))
        else:
            self.register_buffer("_fixed_temp", torch.tensor(float(gate_temp)))

        # fuse and attention
        self.fuse = (
            ConvModule(c2, c2, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=act_cfg)
            if use_fuse
            else nn.Identity()
        )
        self.eca = ECAAttention(c2) if use_at else nn.Identity()

    def _temperature(self) -> torch.Tensor:
        if self.learnable_temp:
            return F.softplus(self._raw_temp) + 1e-3
        else:
            return self._fixed_temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional pre-smoothing. For ResNet backbone integration this is often disabled.
        if self.pre_smooth:
            x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0, ceil_mode=False, count_include_pad=True)
        x1, x2 = x.chunk(2, 1)

        # branch1: conv downsample
        x1 = self.cv1(x1)

        # branch2: anti-alias + gated pooling downsample
        x2 = self.blur(x2)
        x2_max = F.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
        x2_avg = F.avg_pool2d(x2, kernel_size=3, stride=2, padding=1)
        temp = self._temperature()
        logits = self.gate(x2_max + x2_avg) / temp
        w = F.softmax(logits, dim=1)  # [w_max, w_avg]
        x2 = w[:, 0:1] * x2_max + w[:, 1:2] * x2_avg
        x2 = self.cv2(x2)

        out = torch.cat((x1, x2), 1)
        out = self.fuse(out)
        out = self.eca(out)
        return out
