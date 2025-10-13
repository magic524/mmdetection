_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',  # 继承基础优化器与通用设定（下面会覆盖 epochs 与 scheduler）
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
        interval=1,
        save_best='auto',
        save_last=True,
        max_keep_ckpts=1,
    )
)

"""V3e 关键点说明：
1. 同时在 backbone 的 stride stages 与 FPN extra level 使用 ADown。
2. 分层差异化：layer2 强去噪、layer3 平衡、layer4 保结构；FPN 保守。
3. 采用 Linear warmup + MultiStep，避免 gate 温度/权重早期震荡。
4. 修正：去除重复 optim_wrapper 定义；评估输出目录改为 v3e；
"""

# -------------------- 自定义训练配置 --------------------
# 12 epoch 快速验证
train_cfg = dict(max_epochs=12)

# 线性 warmup 500 it + milestones (8,11)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=False, begin=0, end=500),
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
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_v3e/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_v3e/test'
)

# -------------------- 模型配置：V3e - backbone和neck --------------------
model = dict(
        backbone=dict(
                _delete_=True,
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
                 # layer2: 强去噪 + 稳定温度
                 2: dict(ks=3, use_blur=True, gate_temp=1.3, use_at=True,
                     use_fuse=True, learnable_temp=False, pre_smooth=True),
                 # layer3: 允许温度学习
                 3: dict(ks=3, use_blur=True, gate_temp=1.0, use_at=True,
                     use_fuse=True, learnable_temp=True, pre_smooth=False),
                 # layer4: 保结构，去掉 blur，较高温度 + learnable
                 4: dict(ks=3, use_blur=False, gate_temp=1.5, use_at=True,
                     use_fuse=False, learnable_temp=True, pre_smooth=False),
             }
        ),
        neck=dict(
                type='FPN_ADown',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5,
                adown_cfg=dict(
                        ks=3, use_blur=True, gate_temp=1.05, use_at=True,
                        use_fuse=True, learnable_temp=True, pre_smooth=False
                )
        )
)

# 统一的优化器（去除重复定义）
optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.0024, momentum=0.9, weight_decay=0.0001),
        clip_grad=dict(max_norm=35, norm_type=2)
)

# 可选：显式指定 work_dir（也可由命令行 --work-dir 覆盖）
work_dir = './work_dirs/nightdrone_fpn_adown_v3e'