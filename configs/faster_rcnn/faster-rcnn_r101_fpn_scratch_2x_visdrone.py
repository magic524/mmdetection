_base_ = './faster-rcnn_r50_fpn_scratch_2x_visdrone.py'

# ================================================================================
# 实验配置：ResNet101 从头训练 48 epochs（VisDrone 数据集）
# 对比：faster-rcnn_r50_fpn_scratch_2x_visdrone.py
# 目的：验证更深的网络在从头训练时的表现
# 预期：mAP 约 16-21%（略高于 ResNet50 的 15-20%）
# ================================================================================

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None  # ❌ 无预训练权重
    )
)

work_dir = './work_dirs/visdrone_r101_scratch_2x'
