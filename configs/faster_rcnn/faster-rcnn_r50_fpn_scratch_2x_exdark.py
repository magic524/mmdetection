_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# ================================================================================
# 实验配置：从头训练 48 epochs（与ResNetADownStem对比）
# 目的：验证在相同训练时长下，无预训练权重的标准Faster R-CNN表现
# 预期：mAP 约 15-20%（比ResNetADownStem的15.1%稍好）
# ================================================================================

# -------------------- 硬件优化配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# -------------------- 训练配置：48 epochs --------------------
max_epochs = 48

train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[36, 44], gamma=0.1)
]

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
data_root = '/data/Nighttime_Dataset/ExDark/'
img_scale = (1333, 800)

classes = (
    'Bicycle',
    'Boat',
    'Bottle',
    'Bus',
    'Car',
    'Cat',
    'Chair',
    'Cup',
    'Dog',
    'Motorbike',
    'People',
    'Table'
)

# -------------------- 从头训练模型配置 --------------------
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    backbone=dict(
        frozen_stages=-1,  # 不冻结
        zero_init_residual=False,
        norm_cfg=norm_cfg,
        norm_eval=False,
        init_cfg=None  # ❌ 无预训练权重
    ),
    neck=dict(
        norm_cfg=norm_cfg
    ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=12
        )
    )
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

test_pipeline = [
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
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=val_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        data_prefix=dict(img=data_root + 'images/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline
    )
)

# -------------------- 评估器配置 --------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/exdark_scratch_2x/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/exdark_scratch_2x/test'
)

# -------------------- 优化器配置 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.)
)

# -------------------- 默认运行时配置 --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_best='auto',
        rule='greater'
    )
)

# -------------------- 训练/验证配置 --------------------
# 每 6 个 epoch 验证一次，最后一个 epoch 也会验证
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=6  # 🔥 从每轮验证改为每 6 轮验证一次
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -------------------- 可视化配置 --------------------
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

# -------------------- 工作目录 --------------------
work_dir = './work_dirs/exdark_scratch_2x'
