_base_ = './faster-rcnn_r50-adown-replace_fpn_scratch_2x_exdark.py'

# ================================================================================
# 实验配置：ResNet101-ADownReplace (方案2) + 从头训练 48 epochs（ExDark 数据集）
# 对比：faster-rcnn_r50-adown-replace_fpn_scratch_2x_exdark.py
# 目的：验证更深网络 + ADown直接替换的效果
# 预期：mAP 约 19-24%，参数量低于方案1的ResNet101-ADownStages
# ================================================================================

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None
    )
)

work_dir = './work_dirs/exdark_r101_adown_replace_scratch_2x'
