# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# 数据集配置（与YOLO完全一致）
dataset_type = 'CocoDataset'
data_root = '/mmdetection/data/Small_Objects_Dataset/TinyPerson/'

# 类别定义（与YOLO的TinyPerson.yaml严格一致）
metainfo = {
    'classes': ('tiny-people', 'dry-person', 'wet-swimmer'),
    'palette': [
        (220, 20, 60),  # tiny-people (红)
        (119, 11, 32),  # dry-person (深红)
        (0, 0, 142)     # wet-swimmer (蓝)
    ]
}

# 训练配置（对齐YOLO关键参数）
train_dataloader = dict(
    batch_size=8,  # 与您YOLO命令batch=8一致
    num_workers=2,  # 对齐YOLO workers=8
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=4),
        pipeline=[
            # 基础增强（与YOLO默认增强强度对齐）
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=False),  # 强制缩放至640x640
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

# 验证/测试配置
val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=False),  # 与训练一致
            dict(type='PackDetInputs')
        ]
    )
)
test_dataloader = val_dataloader

# 评估配置（mAP50计算逻辑与YOLO一致）
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True  # 显示各类别AP
)
test_evaluator = val_evaluator

# 优化器配置（严格对齐YOLO超参数）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,  # lr0=0.01
        momentum=0.937,  # momentum=0.937
        weight_decay=0.0005  # weight_decay=0.0005
    ),
    clip_grad=None  # 关闭梯度裁剪（YOLO默认无裁剪）
)

# 学习率调度（对齐YOLO的线性warmup）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,  # 模拟YOLO的warmup
        by_epoch=True,
        begin=0,
        end=3  # warmup_epochs=3
    ),
    dict(
        type='MultiStepLR',
        milestones=[8, 11],  # 近似YOLO的cos_lr=False时的阶梯下降
        gamma=0.1,
        by_epoch=True
    )
]

# 模型配置（仅修改类别数，其他保持与Faster R-CNN原始论文一致）
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=3,  # 必须修改项
            loss_cls=dict(  # 保持原始交叉熵损失
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(  # 保持原始L1损失
                type='L1Loss',
                loss_weight=1.0)
        )
    )
)

# 训练周期与YOLO对齐
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,  # 与您YOLO的epochs=200一致
    val_interval=1
)

# 数据增强强度控制（关闭MMDet默认增强，仅保留基础变换）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]