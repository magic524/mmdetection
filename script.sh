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
# ResNet101 
# ================================================================================
# ExDark 
python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101_fpn_scratch_2x_exdark.py \
  --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_r101_scratch_2x

python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101-adown-stages_fpn_scratch_2x_exdark.py \
  --work-dir work_dirs/faster_rcnn/ExDark/backbone/exdark_r101_adown_stages_scratch_2x

# NightDrone 
python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101_fpn_scratch_2x_nightdrone.py \
  --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_r101_scratch_2x

python tools/train.py \
  configs/faster_rcnn/faster-rcnn_r101-adown-stages_fpn_scratch_2x_nightdrone.py \
  --work-dir work_dirs/faster_rcnn/NightDrone/backbone/nightdrone_r101_adown_stages_scratch_2x
