_base_ = [
    './faster-rcnn_r50_fpn_2x_visdrone.py'  # 复用 R50 VisDrone 2x 的所有优化配置
]

# ==================== 替换主干为 ResNet-101 ====================
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    )
)

# ==================== 显存优化配置 (2080Ti 22G) ====================
# ResNet-101 比 R50 参数量大,显存占用约增加 30-40%
# 根据实际测试,batch_size=3 在训练后期会 OOM (峰值超过 22G)
# 降低到 batch_size=2 确保稳定运行 (显存峰值约 15-17G)

train_dataloader = dict(
    batch_size=2,  # 从 3 降到 2 以避免 OOM
    num_workers=4,
    persistent_workers=True
)

# ==================== 学习率线性缩放 ====================
# 基准: lr=0.02 对应 batch_size=16
# R101 配置: lr = 0.02 * (2/16) = 0.0025
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

# ==================== 备选方案(如果显存仍不足) ====================
# 方案1: 进一步降低 batch_size=2, lr=0.0025
# 方案2: 降低输入分辨率,修改 img_scale = (1000, 600)
# 方案3: 启用梯度检查点 (略微降速但节省显存)
#   model = dict(
#       backbone=dict(
#           with_cp=True  # 启用 checkpoint,节省约 30% 显存
#       )
#   )