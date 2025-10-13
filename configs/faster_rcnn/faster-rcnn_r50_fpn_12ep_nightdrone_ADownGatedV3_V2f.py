_base_ = './faster-rcnn_r50_fpn_12ep_nightdrone.py'

# V2f: 高分辨率层优化 - 专门针对小目标，仅在 P3/P4 对应层使用 ADown
# 策略：避开 P5 (layer4)，专注于检测小目标的关键层 layer2/layer3

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
            # layer2 (P3/8): 小目标的关键层，使用最温和的 ADown
            2: dict(
                ks=3, use_blur=False, gate_temp=2.0,
                use_at=False, use_fuse=True,  # 启用 fuse 减小分布偏移
                learnable_temp=True, pre_smooth=False,
            ),
            # layer3 (P4/16): 中等目标层，稍强的配置
            3: dict(
                ks=3, use_blur=False, gate_temp=1.5,
                use_at=True, use_fuse=True,
                learnable_temp=True, pre_smooth=False,
            ),
            # 跳过 layer4，保持原始下采样以维持预训练特征的完整性
        },
    ),
)

# 针对小目标优化的训练配置
# 更小的学习率和更多的 warmup
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0012, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=15, norm_type=2)
)

# 学习率调度：更平缓的下降
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),  # warmup
    dict(type='MultiStepLR', by_epoch=True, begin=0, milestones=[9, 11], gamma=0.1),  # 延后衰减
]

# 数据增强：对小目标更友好的配置
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        scale=[(1333, 600), (1333, 800), (1333, 1000)],  # 多尺度，包含更高分辨率
        multiscale_mode='value',
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',  # 光度变换，适合夜间数据
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='PackDetInputs')
]

# 重新定义 dataloader 以使用新的 pipeline
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_train.json',
        data_prefix=dict(img='/data/Nighttime_Dataset/NightDrone/images/'),
        metainfo=dict(classes=('car', 'truck', 'motor', 'pedestrian', 'van', 'tricycle', 'people', 'bus')),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False)
    )
)

val_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2f/val'
)

test_evaluator = dict(
    type='CocoMetric', ann_file='/data/Nighttime_Dataset/NightDrone/annotations/instances_val.json', 
    metric='bbox', format_only=False, classwise=True, 
    outfile_prefix='./work_dirs/nightdrone_ADownGatedV3_V2f/test'
)