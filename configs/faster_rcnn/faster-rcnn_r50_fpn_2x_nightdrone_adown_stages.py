_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

train_cfg = dict(max_epochs=48)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[36, 44], gamma=0.1)
]

dataset_type = 'CocoDataset'
data_root = '/data/Nighttime_Dataset/NightDrone/'
img_scale = (1333, 800)

classes = (
    'car', 'truck', 'motor', 'pedestrian', 'van', 'tricycle', 'people', 'bus'
)

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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_adown_stages/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_adown_stages/test'
)

# Use ResNetADownStages: keep stem MaxPool, replace stage downsamplings with ADownGatedV3
model = dict(
    backbone=dict(
        type='ResNetADownStages',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # Per-stage ADown parameters (stage indices: 1..4). Only apply to stride stages 2-4.
        adown_stage_cfg={
            # P3/8 counterpart -> layer2: stronger smoothing, no attention/fuse
            2: dict(ks=3, use_blur=True, gate_temp=1.5, use_at=False, use_fuse=False, learnable_temp=False),
            # P4/16 counterpart -> layer3: balanced, enable attention
            3: dict(ks=3, use_blur=True, gate_temp=1.0, use_at=True,  use_fuse=False, learnable_temp=False),
            # P5/32 counterpart -> layer4: less blur, edge-biased, enable attention
            4: dict(ks=3, use_blur=False, gate_temp=0.8, use_at=True, use_fuse=False, learnable_temp=False),
        },
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))
