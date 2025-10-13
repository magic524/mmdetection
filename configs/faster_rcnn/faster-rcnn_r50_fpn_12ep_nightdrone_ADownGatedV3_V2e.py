# V2e: 并行融合策略 - ADown 作为辅助分支与原 stride conv 并行

_base_ = './faster-rcnn_r50_fpn_12ep_nightdrone.py'

model = dict(
    backbone=dict(
        type='ResNetADownParallel',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        adown_stage_cfg={
            4: dict(ks=3, use_blur=False, gate_temp=1.0, use_at=False, use_fuse=False, 
                   learnable_temp=False, pre_smooth=False)
        },
    ),
)

# 稍微提高学习率，因为有预训练权重的稳定性
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0018, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=20, norm_type=2)
)

val_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2e/val'
)

test_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2e/test'
)