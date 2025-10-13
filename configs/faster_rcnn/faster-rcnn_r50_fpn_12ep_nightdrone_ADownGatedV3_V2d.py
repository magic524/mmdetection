_base_ = './faster-rcnn_r50_fpn_12ep_nightdrone.py'

# V2d: 渐进式训练策略 - 分阶段解冻 ADown 参数
# 阶段1 (0-4 epoch): 冻结 ADown，只训练检测头
# 阶段2 (5-8 epoch): 解冻 ADown gate，微调门控权重
# 阶段3 (9-12 epoch): 全解冻，端到端训练

# 仅在 layer4 使用 ADown，但采用渐进式训练
model = dict(
    backbone=dict(
        type='ResNetADownStages',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        adown_stage_cfg={
            4: dict(
                ks=3, use_blur=False, gate_temp=1.0,  # 固定温度，避免初期不稳定
                use_at=False, use_fuse=False,  # 最简配置
                learnable_temp=False, pre_smooth=False,
            )
        },
    ),
)

# 自定义训练配置：分阶段冻结/解冻
train_cfg = dict(
    max_epochs=12,
    # 自定义 hook 控制参数冻结（需要在 mmdet/engine/hooks 中实现）
    stage_configs=[
        dict(epochs=(0, 4), freeze_adown=True, freeze_adown_gate=True),
        dict(epochs=(5, 8), freeze_adown=False, freeze_adown_gate=False), 
        dict(epochs=(9, 12), freeze_adown=False, freeze_adown_gate=False),
    ]
)

# 对应的分阶段学习率
param_scheduler = [
    dict(type='ConstantLR', factor=0.5, by_epoch=True, begin=0, end=4),  # 阶段1: 低学习率
    dict(type='MultiStepLR', by_epoch=True, begin=5, milestones=[8, 11], gamma=0.1),  # 阶段2+3: 正常调度
]

# 更保守的优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),  # 更低的初始学习率
    clip_grad=dict(max_norm=10, norm_type=2)  # 更严格的梯度裁剪
)

# 输出路径调整
val_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2d/val'
)

test_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2d/test'
)