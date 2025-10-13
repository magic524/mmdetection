_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# ==================== 数据集与路径 ====================
dataset_type = 'CocoDataset'
data_root = '/data/Small_Objects_Dataset/VisDrone/'
# 多尺度训练,保持宽高比
img_scale = (1333, 800)  # (width, height) 标准 COCO 尺度

classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor'
)

# ==================== 模型配置(小目标优化) ====================
model = dict(
    # 修改类别数
    roi_head=dict(
        bbox_head=dict(num_classes=10)),
    # RPN 针对小目标优化
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8],  # 更小的 anchor scales
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        feat_channels=256),
    # 训练配置优化
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3),
            sampler=dict(num=256, pos_fraction=0.5)),
        rpn_proposal=dict(
            nms_pre=3000,  # 增加 proposal 数量
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),  # 允许更小的 bbox
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5),
            sampler=dict(num=512, pos_fraction=0.25))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300))  # 增加每张图最大检测数
)

# ==================== 数据增强流水线(强化版) ====================
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # 多尺度训练
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576), 
                (1333, 608), (1333, 640), (1333, 672), (1333, 704), 
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # 增加颜色抖动
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=img_scale, keep_ratio=True),  # 保持宽高比
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# ==================== DataLoader ====================
# ==================== 数据加载器配置 ====================
train_dataloader = dict(
    batch_size=4,  # 从12降低到4,配合更大的输入尺寸
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = val_dataloader

# ==================== 评估器 ====================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/visdrone_det/faster_rcnn_r50_2x_val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',  # 若无单独test集则仍用val
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/visdrone_det/faster_rcnn_r50_2x_test'
)

# ==================== 训练与验证节奏 ====================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ==================== 学习率配置(根据 batch_size=4 调整) ====================
# 基础学习率: 0.02 * (4/16) = 0.005
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))

# 学习率调度策略
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False, 
        begin=0, 
        end=500),  # warmup 500 iterations
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# ==================== Checkpoint 策略（只保留 best & last） ====================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,              # 每个 epoch 产出一个最新的 last.pth
        max_keep_ckpts=1,        # 只保留最近1个 checkpoint
        save_last=True,          # 保存 last 权重
        save_best='auto',        # 自动保存 best 权重(基于 bbox_mAP)
        rule='greater'           # bbox_mAP 越大越好
    )
)

# ==================== 环境优化 ====================
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# ==================== 自动缩放学习率(可选) ====================
# 如果自动调整学习率,取消注释
# auto_scale_lr = dict(enable=True, base_batch_size=16)
