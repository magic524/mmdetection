_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# -------------------- 硬件优化配置 --------------------
# 启用混合精度训练 + 显存优化
env_cfg = dict(
    cudnn_benchmark=True,  # 加速卷积计算
    mp_cfg=dict(mp_start_method='fork'),  # 多进程优化
    dist_cfg=dict(backend='nccl')
)

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
data_root = '/data/Small_Objects_Dataset/VisDrone/'
img_scale = (640, 640)  # 与YOLO对齐

classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
)

# 简化数据流水线（移除小目标增强）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),  # 方形输入
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# -------------------- 数据加载配置 --------------------
train_dataloader = dict(
    batch_size=12,  # 2080Ti 22GB可支持更大batch
    num_workers=8,  # 根据CPU核心数调整（建议物理核心数的75%）
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False)  # 关闭小目标过滤
    )
)

val_dataloader = dict(
    batch_size=8,  # 验证阶段可用稍大batch
    num_workers=4,
    dataset=dict(
        ann_file=data_root + 'annotations/instances_val.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=img_scale, keep_ratio=False),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader

# -------------------- 模型与优化配置 --------------------
# 学习率策略（batch_size=12时的线性缩放规则）
base_lr = 0.02 * 12 / 16  # 基准lr=0.02对应batch=16，按线性缩放
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 混合精度训练
    optimizer=dict(
        type='SGD',
        lr=base_lr,  # 缩放后的学习率
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic'  # 自动调整损失缩放
)

# 训练调度（200 epochs）
max_epochs = 200
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),  # warmup
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=0,
        end=max_epochs)  # cosine衰减更稳定
]

# -------------------- 模型微调 --------------------
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # 保持COCO预训练统计量
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    # 优化RPN参数（适配640输入）
    rpn_head=dict(
        anchor_generator=dict(
            scales=[2, 4, 8],  # 缩小anchor尺度
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        feat_channels=128  # 减少特征通道数
    )
)

# -------------------- 训练监控 --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # 每5个epoch保存一次
        max_keep_ckpts=2,
        save_best='auto'
    ),
    logger=dict(
        type='LoggerHook',
        interval=20  # 每20迭代记录日志
    ),
    visualization=dict(type='DetVisualizationHook', draw=False)  # 关闭可视化节省资源
)

# 验证频率调整
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5  # 每5个epoch验证一次
)