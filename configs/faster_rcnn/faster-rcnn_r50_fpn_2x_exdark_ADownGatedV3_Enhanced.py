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

# -------------------- 默认运行时配置 --------------------
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,  # 每个 epoch 保存一次
        max_keep_ckpts=1,  # 只保留最近的权重
        save_best='auto',  # 自动保存验证集表现最好的模型
        rule='greater'  # 指标越高越好（针对 mAP）
    )
)

# -------------------- 自定义训练配置：2x 版本（48 epochs） --------------------
train_cfg = dict(max_epochs=48)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[36, 44], gamma=0.1)
]

# -------------------- 数据集配置 --------------------
dataset_type = 'CocoDataset'
# Windows 路径 E:\Datasets\Nighttime_Dataset\ExDark\ 已通过 docker 挂载为 /data/Nighttime_Dataset/ExDark/
data_root = '/data/Nighttime_Dataset/ExDark/'
img_scale = (1333, 800)

# 根据 Dark.yaml 的 names 列表设置类别（保持索引顺序）
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
    outfile_prefix='./work_dirs/exdark_adown_enhanced_2x/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/exdark_adown_enhanced_2x/test'
)

# -------------------- 增强模型配置：全面部署 ADownGatedV3 --------------------
model = dict(
    neck=dict(
        type='FPN_ADown_Enhanced',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        add_extra_convs=False,  # ✓ 是否启用额外层的ADown
        
        # ★★★ 核心增强配置 ★★★
        enhance_bottom_up=True,     # ✓ 启用底向上增强路径
        multi_scale_adown=False,    # ✗ 暂不启用多尺度lateral ADown (可能过度复杂)
        
        # ★ ADownGatedV3 夜间优化配置 ★
        adown_cfg=dict(
            ks=3,
            use_blur=True,          # ✓ 抗混叠模糊，减少夜间噪声
            gate_temp=0.8,          # ✓ 降低温度，增强门控敏感性
            use_at=True,            # ✓ 启用ECA注意力，提升特征选择
            use_fuse=True,          # ✓ 特征融合
            learnable_temp=True,    # ✓ 可学习温度参数
            pre_smooth=True         # ✓ 预平滑处理，进一步降噪
        ),
        
        # ★ 底向上增强路径配置 ★ 
        enhance_cfg=dict(
            enable_p2p3=False,      # ✗ P2->P3不启用 (避免过拟合浅层特征)
            enable_p3p4=True,       # ✓ P3->P4启用 (中层特征增强)
            enable_p4p5=True,       # ✓ P4->P5启用 (深层特征增强)
            adown_channels=128      # ✓ 增强路径通道数
        )
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes))  # ExDark有12个类别
    )
)

# -------------------- 优化器配置：针对增强架构调整 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=0.0018,  # 稍微降低学习率，因为模型更复杂
        momentum=0.9, 
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=40, norm_type=2)  # 增加梯度裁剪强度
)

# -------------------- 可视化配置（可选） --------------------
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
