# retinanet_r50_fpn_1x_visdrone.py
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# -------------------- 模型配置 --------------------
model = dict(
    data_preprocessor=dict(
        pad_size_divisor=32
    ),
    bbox_head=dict(
        num_classes=10,  # VisDrone类别数
        # 针对小目标优化anchor设置
        anchor_generator=dict(
            octave_base_scale=2,  # 减小基础尺度以适应小目标
            scales_per_octave=5,  # 增加尺度密度
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0],  # 更宽的高宽比范围
            strides=[4, 8, 16, 32, 64]  # 减小最小stride
        ),
        # 优化损失函数参数
        loss_cls=dict(
            gamma=3.0,  # 增加困难样本权重
            alpha=0.9    # 提高正样本权重
        )
    )
)

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
data_root = '/data/Small_Objects_Dataset/VisDrone/'
img_scale = (640, 640)  # 与YOLO模型保持一致

classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
)

# 更稳定的数据流水线（移除albumentations依赖）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),  # MMDet内置的增强
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackDetInputs')
]

# -------------------- 数据加载器配置 --------------------
train_dataloader = dict(
    batch_size=16,  # 根据GPU显存调整
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=4)  # 过滤微小无效标注
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline
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
    outfile_prefix='./work_dirs/retinanet_visdrone/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/retinanet_visdrone/test'
)

# -------------------- 训练策略优化 --------------------
# 优化器配置
# 修改优化器配置，加强梯度裁剪
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),  # 降低初始LR
    clip_grad=dict(max_norm=10, norm_type=2)  # 加强梯度裁剪
)

# 学习率调度
# 调整学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=2000  # 延长warmup阶段
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[36, 44],
        gamma=0.1
    )
]

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=48,  # 与Faster R-CNN保持一致
    val_interval=3  # 每3个epoch验证一次
)

# -------------------- 硬件配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# -------------------- 钩子配置 --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # 每3个epoch保存一次
        max_keep_ckpts=3,  # 仅保留最近3个检查点
        save_best='auto'  # 自动保存最佳模型
    ),
    logger=dict(type='LoggerHook', interval=50)  # 每50次迭代记录一次
)