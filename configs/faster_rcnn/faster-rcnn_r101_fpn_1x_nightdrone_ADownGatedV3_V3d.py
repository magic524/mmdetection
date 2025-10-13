_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# -------------------- 硬件优化配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# 覆盖默认的 checkpoint hook：只保留最佳和最后一轮的权重，以节省磁盘空间
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,  # R101更大，每2个epoch保存一次
        save_best='auto',
        save_last=True,
        max_keep_ckpts=1,
    )
)

# -------------------- 自定义训练配置：1x版本（12 epochs） --------------------
train_cfg = dict(max_epochs=12)
# 对应的学习率调度：在第 8 和 11 轮下降
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
# Windows 路径 E:\Datasets\Nighttime_Dataset\NightDrone\ 已转换为 WSL 格式
data_root = '/data/Nighttime_Dataset/NightDrone/'
img_scale = (1333, 800)

# 根据 NightDrone.yaml 的 names 列表设置类别（保持索引顺序）
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
    batch_size=2,  # R101较大，可能需要调整batch size
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
    outfile_prefix='./work_dirs/nightdrone_r101_fpn_adown_v3d_1x/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_r101_fpn_adown_v3d_1x/test'
)

# -------------------- 模型配置：ResNet-101 + FPN_ADown + V3d配置 --------------------
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,  # 使用ResNet-101
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')  # R101预训练权重
    ),
    neck=dict(
        type='FPN_ADown',
        in_channels=[256, 512, 1024, 2048],  # R101的输出通道与R50相同
        out_channels=256,
        num_outs=5,
        # ADownGatedV3配置：V3d版本 - 简化版本，专注gating机制
        adown_cfg=dict(
            ks=3,
            use_blur=True,       # 启用抗混叠去噪
            gate_temp=1.0,       # 标准温度值
            use_at=False,        # 禁用ECA注意力，减少参数复杂度
            use_fuse=True,       # 保留特征融合
            learnable_temp=True, # 启用可学习温度
            pre_smooth=True      # 启用轻微预平滑，进一步降噪
        )
    )
)

# -------------------- 优化器配置：R101需要更保守的学习率 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),  # R101用标准学习率
    clip_grad=dict(max_norm=35, norm_type=2)  # 梯度裁剪防止发散
)