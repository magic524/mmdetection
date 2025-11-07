_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# ================================================================================
# 实验配置：ResNetADownStages + 从头训练 48 epochs
# 设计理念：参考 YOLO11-ADownGatedV3-V13 的参数设计
# 
# 改动点：
#   1. Stem层保持不变（保留MaxPool，避免ResNetADownStem的问题）
#   2. Layer2/3/4 的 stride=2 下采样替换为 ADownGatedV3
#   3. 参数设计（参考YOLO11-V13）：
#      - Layer2 (C3): 浅层，强平滑抑噪 (use_blur=True, gate_temp=1.5, 无注意力)
#      - Layer3 (C4): 中层，均衡 (use_blur=True, gate_temp=1.0, 开启注意力)
#      - Layer4 (C5): 深层，边缘保留 (use_blur=False, gate_temp=0.8, 开启注意力)
# 
# 对比基线：faster-rcnn_r50_fpn_scratch_2x_exdark.py (标准ResNet，无预训练)
# 预期：mAP 约 18-23% (应该好于标准从头训练的15-20%，因为ADown更适合夜间)
# ================================================================================

# -------------------- 硬件优化配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# -------------------- 训练配置：48 epochs（与基线一致）--------------------
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

# -------------------- ResNetADownStages 模型配置 --------------------
# 使用 GroupNorm（对从头训练更稳定）
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    backbone=dict(
        type='ResNetADownStages',  # 🔥 使用ADown替换stage下采样
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # 不冻结任何层
        zero_init_residual=False,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=None,  # ❌ 无预训练权重（从头训练）
        # 🎯 ADown配置（参考YOLO11-V13的参数设计）
        # 默认已经内置，这里显式写出以便调整
        adown_stage_cfg={
            2: dict(  # Layer2 (C3, H/8): 浅层 - 强平滑抑噪
                ks=3, 
                use_blur=True,      # ✅ 抗混叠
                gate_temp=1.5,      # 高温度，更平滑的门控
                use_at=False,       # ❌ 浅层不用注意力
                use_fuse=False, 
                learnable_temp=False, 
                pre_smooth=False
            ),
            3: dict(  # Layer3 (C4, H/16): 中层 - 均衡
                ks=3, 
                use_blur=True,      # ✅ 抗混叠
                gate_temp=1.0,      # 中等温度
                use_at=True,        # ✅ 开启注意力
                use_fuse=False, 
                learnable_temp=False, 
                pre_smooth=False
            ),
            4: dict(  # Layer4 (C5, H/32): 深层 - 边缘保留
                ks=3, 
                use_blur=False,     # ❌ 不过度平滑，保留边缘
                gate_temp=0.8,      # 低温度，更锐利的门控
                use_at=True,        # ✅ 开启注意力
                use_fuse=False, 
                learnable_temp=False, 
                pre_smooth=False
            ),
        }
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
    outfile_prefix='./work_dirs/exdark_adown_stages_scratch_2x/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/exdark_adown_stages_scratch_2x/test'
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
work_dir = './work_dirs/exdark_adown_stages_scratch_2x'
