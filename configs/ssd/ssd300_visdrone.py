_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_2x.py',
]

input_size = 300
data_root = '/data/Small_Objects_Dataset/VisDrone/'
classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
)

model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],  # 使用 ImageNet 标准 std
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=len(classes),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.45),  # 适配小目标：减小 anchor
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[1, 2], [1, 2, 3], [1, 2, 3], [1, 2], [1], [1]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=0.6,  # 改小以提高小目标鲁棒性
        allowed_border=0,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.01,  # 更低分数保留小目标
        max_per_img=200)
)

# -------------------- 数据增强（减弱版） --------------------
img_scale = (300, 300)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand', mean=[123.675, 116.28, 103.53], to_rgb=True, ratio_range=(1, 1.5)),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3), min_crop_size=0.7),  # 保留更多 GT
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackDetInputs'),
]

# -------------------- 数据加载 --------------------
dataset_type = 'CocoDataset'

train_dataloader = dict(
    batch_size=16,  # 减小 batch size 保守训练
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True)  # 避免空框
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

# -------------------- 评估器 --------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    classwise=True,
    outfile_prefix='./work_dirs/ssd300_visdrone/val'
)

test_evaluator = val_evaluator

# -------------------- 优化器和调度器 --------------------
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4),  # 更保守的初始 lr
    accumulative_counts=1
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),  # warmup
    dict(type='MultiStepLR', by_epoch=True, milestones=[30, 38], gamma=0.1)
]

train_cfg = dict(
    max_epochs=120,
    val_interval=10  # ✅ 每 1 个 epoch 验证一次
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5),
    logger=dict(type='LoggerHook', interval=50)
)
