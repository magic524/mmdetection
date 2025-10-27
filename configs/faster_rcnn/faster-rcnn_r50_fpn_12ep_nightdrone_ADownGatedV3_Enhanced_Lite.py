_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# -------------------- 硬件优化配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# 覆盖默认的 checkpoint hook：快速实验版本
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        save_last=True,
        max_keep_ckpts=1,
    )
)

# -------------------- 快速实验配置：12 epochs --------------------
train_cfg = dict(max_epochs=12)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
data_root = '/data/Nighttime_Dataset/NightDrone/'
img_scale = (1333, 800)

classes = (
    'car',
    'truck',
    'motor',
    'pedestrian',
    'van',
    'tricycle',
    'people',
    'bus'
)

# -------------------- 数据流水线 --------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='PackDetInputs')
]

# -------------------- 数据加载配置 --------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=val_pipeline
    )
)

test_dataloader = val_dataloader

# -------------------- 评估器配置 --------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_enhanced_12ep/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_enhanced_12ep/test'
)

# -------------------- 轻量级增强模型配置 --------------------
model = dict(
    neck=dict(
        type='FPN_ADown_Enhanced',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        add_extra_convs=True,           # ✓ 启用额外层的ADown
        
        # ★ 轻量级增强配置 ★
        enhance_bottom_up=True,         # ✓ 启用底向上增强路径
        multi_scale_adown=False,        # ✗ 暂不启用，避免过度复杂
        
        # ★ ADownGatedV3 激进优化配置 ★
        adown_cfg=dict(
            ks=3,
            use_blur=True,              # ✓ 抗混叠
            gate_temp=0.7,              # ✓ 更低温度，更敏感门控
            use_at=False,               # ✗ 禁用注意力，减少参数 (轻量级)
            use_fuse=True,              # ✓ 保留融合
            learnable_temp=True,        # ✓ 可学习温度
            pre_smooth=True             # ✓ 预平滑
        ),
        
        # ★ 保守的增强路径配置 ★
        enhance_cfg=dict(
            enable_p2p3=False,          # ✗ 不启用浅层
            enable_p3p4=True,           # ✓ 中层增强
            enable_p4p5=False,          # ✗ 暂不启用深层 (简化版)
            adown_channels=96           # ✓ 减少通道数
        )
    )
)

# -------------------- 轻量级优化器配置 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.003,   # 提高学习率，快速实验
        momentum=0.9, 
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)