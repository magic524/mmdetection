_base_ = [
    './faster-rcnn_r50_fpn_12ep_nightdrone_ADownGatedV3_V2a.py'
]

# 在 V2a 基础上再启用 layer3 的 ADown（温和配置）
model = dict(
    backbone=dict(
        adown_stage_cfg={
            3: dict(
                ks=3,
                use_blur=False,
                gate_temp=2.0,
                use_at=True,
                use_fuse=False,
                learnable_temp=True,
                pre_smooth=False,
            ),
            4: dict(
                ks=3,
                use_blur=False,
                gate_temp=5.0,
                use_at=True,
                use_fuse=True,
                learnable_temp=True,
                pre_smooth=False,
            )
        }
    )
)
