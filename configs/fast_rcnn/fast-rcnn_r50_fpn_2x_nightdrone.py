_base_ = [
    '../_base_/models/fast-rcnn_r50_fpn.py',
    # 不继承 coco_detection.py，全部数据配置自行定义
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# -------------------- 硬件优化配置 --------------------
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork'),
    dist_cfg=dict(backend='nccl')
)

# -------------------- 自定义训练配置 --------------------
train_cfg = dict(max_epochs=48)
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[36, 44], gamma=0.1)
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
# Fast R-CNN 需要预先生成的 proposals
# 注意：需要先使用 RPN 或 Faster R-CNN 生成 proposal 文件
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=img_scale, keep_ratio=True),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = val_pipeline

# -------------------- 数据加载配置 --------------------
# 注意：proposal_file 需要使用工具提前生成
# 可以使用 tools/misc/browse_dataset.py 中的相关脚本生成
# 或者先训练 RPN 模型生成 proposals
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
        # TODO: 需要先生成 proposal 文件，路径示例：
        proposal_file=data_root + 'proposals/rpn_r50_fpn_nightdrone_train.pkl',
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
        # TODO: 需要先生成 proposal 文件，路径示例：
        proposal_file=data_root + 'proposals/rpn_r50_fpn_nightdrone_val.pkl',
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
    outfile_prefix='./work_dirs/nightdrone_fastrcnn/val'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    outfile_prefix='./work_dirs/nightdrone_fastrcnn/test'
)

# -------------------- 优化器配置 --------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001))

# -------------------- 说明 --------------------
# Fast R-CNN 需要预先生成的 region proposals
# 生成 proposals 的步骤：
# 1. 先训练一个 RPN 模型或使用已有的 Faster R-CNN 模型
# 2. 使用 tools/analysis_tools/get_flops.py 或自定义脚本生成 proposals
# 3. 将生成的 .pkl 文件放在上述 proposal_file 指定的路径
# 
# 或者，如果只想进行目标检测训练，建议直接使用 Faster R-CNN 配置文件：
# configs/faster_rcnn/faster-rcnn_r50_fpn_2x_nightdrone.py
