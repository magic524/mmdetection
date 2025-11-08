#!/bin/bash

#######      chmod +x script.sh
#######      ./script.sh

# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_12ep_nightdrone_ADownGatedV3_V3b.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_nightdrone_ADownGatedV3_V3d.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_visdrone.py

###251001

# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_visdrone.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r101_fpn_2x_visdrone.py

# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_exdark_ADownGatedV3_V3d.py --work-dir work_dirs/faster_rcnn_exdark_adown_v3d
# python tools/train.py \
#     configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_exdark.py \
#     --work-dir work_dirs/exdark_adown_stages_scratch_2x

# ================================================================================
# ResNet50 从头训练实验（两个数据集）
# ================================================================================
# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r50_fpn_scratch_2x_exdark.py \
#   --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_scratch_2x

# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_exdark.py \
#   --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_adown_stages_scratch_2x

# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r50_fpn_scratch_2x_nightdrone.py \
#   --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_scratch_2x

# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_nightdrone.py \
#   --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_adown_stages_scratch_2x

# ================================================================================
# ResNet101 从头训练实验（方案1：ADownStages - 参数增加）
# ================================================================================
# # ExDark 
# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r101_fpn_scratch_2x_exdark.py \
#   --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_r101_scratch_2x

# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r101-adown-stages_fpn_scratch_2x_exdark.py \
#   --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_r101_adown_stages_scratch_2x

# # NightDrone 
# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r101_fpn_scratch_2x_nightdrone.py \
#   --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_r101_scratch_2x

# python tools/train.py \
#   configs/faster_rcnn/faster-rcnn_r101-adown-stages_fpn_scratch_2x_nightdrone.py \
#   --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_r101_adown_stages_scratch_2x

# ================================================================================
# 方案2：ResNetADownReplace - ADown替换Bottleneck（参数减少）
# ================================================================================
# ExDark 数据集
python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r50-adown-replace_fpn_scratch_2x_exdark.py \
  --work-dir work_dirs/faster_rcnn/ExDark/backbone_v2/exdark_r50_adown_replace_scratch_2x

python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101-adown-replace_fpn_scratch_2x_exdark.py \
  --work-dir work_dirs/faster_rcnn/ExDark/backbone_v2/exdark_r101_adown_replace_scratch_2x

# NightDrone 数据集
python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r50-adown-replace_fpn_scratch_2x_nightdrone.py \
  --work-dir work_dirs/faster_rcnn/NightDrone/backbone_v2/nightdrone_r50_adown_replace_scratch_2x

python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101-adown-replace_fpn_scratch_2x_nightdrone.py \
  --work-dir work_dirs/faster_rcnn/NightDrone/backbone_v2/nightdrone_r101_adown_replace_scratch_2x

#!/bin/bash

# #!/bin/bash

# # ================================================================================
# # 测试模型复杂度（使用 GPU）
# # ================================================================================

# # 测试标准 ResNet50
# python tools/analysis_tools/get_flops.py \
#     configs/faster_rcnn/faster-rcnn_r50_fpn_scratch_2x_nightdrone.py \
#     --cfg-options model.test_cfg.rcnn.max_per_img=100

# # ==============================
# # Use size divisor set input shape from (1080, 1920) to (768, 1344)
# # ==============================
# # Compute type: dataloader: load a picture from the dataset
# # Input shape: (768, 1344)
# # Flops: 0.324T
# # Params: 42.698M
# # ==============================

# # 测试标准 ResNet101
# python tools/analysis_tools/get_flops.py \
#     configs/faster_rcnn/faster-rcnn_r101_fpn_scratch_2x_nightdrone.py \
#     --cfg-options model.test_cfg.rcnn.max_per_img=100

# # ==============================
# # Use size divisor set input shape from (1080, 1920) to (768, 1344)
# # ==============================
# # Compute type: dataloader: load a picture from the dataset
# # Input shape: (768, 1344)
# # Flops: 0.401T
# # Params: 61.69M
# # ==============================

# # 测试 ResNet50-ADownStages
# python tools/analysis_tools/get_flops.py \
#     configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_nightdrone.py \
#     --cfg-options model.test_cfg.rcnn.max_per_img=100

# # ==============================
# # Use size divisor set input shape from (1080, 1920) to (768, 1344)
# # ==============================
# # Compute type: dataloader: load a picture from the dataset
# # Input shape: (768, 1344)
# # Flops: 0.327T
# # Params: 46.144M
# # ==============================

# # 测试 ResNet101-ADownStages
# python tools/analysis_tools/get_flops.py \
#     configs/faster_rcnn/faster-rcnn_r101-adown-stages_fpn_scratch_2x_nightdrone.py \
#     --cfg-options model.test_cfg.rcnn.max_per_img=100

# # ==============================
# # Use size divisor set input shape from (1080, 1920) to (768, 1344)
# # ==============================
# # Compute type: dataloader: load a picture from the dataset
# # Input shape: (768, 1344)
# # Flops: 0.404T
# # Params: 65.136M
# # ==============================
