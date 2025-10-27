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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,
        save_best='auto',
        save_last=True,
        max_keep_ckpts=3,  # 保留更多版本便于分析
    )
)

# -------------------- 极致训练配置：2x版本 --------------------
train_cfg = dict(max_epochs=48)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[36, 44], gamma=0.1)
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
    batch_size=1,  # 极致版本可能需要更大显存，减少batch size
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
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_enhanced_ultimate_2x/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_fpn_adown_enhanced_ultimate_2x/test'
)

# -------------------- 🚀 极致增强模型配置：最大化ADownGatedV3部署 🚀 --------------------
model = dict(
    neck=dict(
        type='FPN_ADown_Enhanced',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=6,  # ✓ 增加到P6 (更多尺度用于小目标检测)
        add_extra_convs=True,
        
        # ★★★ 极致增强配置 ★★★
        enhance_bottom_up=True,         # ✓ 启用底向上增强路径
        multi_scale_adown=True,         # ✓ 启用多尺度lateral ADown (极致版)
        
        # ★ ADownGatedV3 极致夜间优化配置 ★
        adown_cfg=dict(
            ks=3,
            use_blur=True,              # ✓ 抗混叠模糊
            gate_temp=0.6,              # ✓ 更低温度，极致敏感门控
            use_at=True,                # ✓ 启用ECA注意力
            use_fuse=True,              # ✓ 特征融合
            learnable_temp=True,        # ✓ 可学习温度参数
            pre_smooth=True             # ✓ 预平滑处理
        ),
        
        # ★ 极致底向上增强路径配置 ★
        enhance_cfg=dict(
            enable_p2p3=True,           # ✓ P2->P3启用 (极致版包含浅层)
            enable_p3p4=True,           # ✓ P3->P4启用
            enable_p4p5=True,           # ✓ P4->P5启用
            adown_channels=160          # ✓ 增加增强路径通道数
        )
    )
)

# -------------------- 极致优化器配置：适应复杂架构 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',  # 使用 AdamW 优化器，更适合复杂模型
        lr=0.0001,     # 降低学习率
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8
    ),
    clip_grad=dict(max_norm=50, norm_type=2)  # 更强的梯度裁剪
)

# -------------------- 自定义学习率调度 --------------------
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Main LR schedule  
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[32, 40, 46],  # 更细致的学习率衰减
        gamma=0.1,
        begin=0,
        end=48
    )
]