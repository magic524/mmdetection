_base_ = './faster-rcnn_r50-adown-stages_fpn_scratch_2x_nightdrone.py'

# ================================================================================
# 实验配置：ResNet101-ADownStages + 从头训练 48 epochs（NightDrone 数据集）
# 对比：faster-rcnn_r50-adown-stages_fpn_scratch_2x_nightdrone.py
# 目的：验证更深的网络 + ADown 在从头训练时的表现
# 预期：mAP 约 19-24%（应优于 ResNet101 标准版和 ResNet50-ADown）
# ================================================================================

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None  # ❌ 无预训练权重
    )
)

work_dir = './work_dirs/nightdrone_r101_adown_stages_scratch_2x'
