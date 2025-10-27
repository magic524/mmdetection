#!/bin/bash

#######      chmod +x script.sh
#######      ./script.sh

# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_12ep_nightdrone_ADownGatedV3_V3b.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_nightdrone_ADownGatedV3_V3d.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_visdrone.py

###251001

# python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_visdrone.py
# python tools/train.py configs/faster_rcnn/faster-rcnn_r101_fpn_2x_visdrone.py

python tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_2x_exdark_ADownGatedV3_V3d.py --work-dir work_dirs/faster_rcnn_exdark_adown_v3d