#!/bin/bash

# ================================================================================
# ResNetADownStages 实验脚本 - ExDark数据集
# 目的：验证在Stage层使用ADown（保留Stem MaxPool）的效果
# ================================================================================

echo "================================================================"
echo "🔬 实验：ResNetADownStages vs 标准ResNet（从头训练对比）"
echo "================================================================"
echo ""
echo "实验设计参考：YOLO11-ADownGatedV3-V13 参数策略"
echo "  - Layer2 (C3, 浅层): 强平滑 (blur=True, temp=1.5, 无注意力)"
echo "  - Layer3 (C4, 中层): 均衡   (blur=True, temp=1.0, 有注意力)"
echo "  - Layer4 (C5, 深层): 边缘   (blur=False, temp=0.8, 有注意力)"
echo ""
echo "================================================================"

# 配置变量
GPU_ID=0

# ================================================================================
# 实验：ResNetADownStages（48 epochs，无预训练）
# ================================================================================
echo ""
echo "🚀 开始训练：ResNetADownStages (Stem不变，Stage用ADown)"
echo "预期：mAP 约 18-23% (优于标准从头训练)"
echo "================================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py \
    configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_exdark.py \
    --work-dir work_dirs/exdark_adown_stages_scratch_2x

echo ""
echo "✅ 训练完成！"
echo ""

# ================================================================================
# 结果对比提示
# ================================================================================
echo "================================================================"
echo "📊 实验结果对比（预期）："
echo "================================================================"
echo ""
echo "| 模型配置                    | Epochs | 预训练 | Stem | Stage下采样 | 预期 mAP |"
echo "|-----------------------------|--------|--------|------|-------------|----------|"
echo "| 标准Faster R-CNN            | 48     | ✅ 是  | MaxPool | Conv stride=2 | ~30%     |"
echo "| ResNetADownStem             | 48     | ⚠️ 部分| ADown   | Conv stride=2 | 15.1%    |"
echo "| 标准ResNet (从头训练)       | 48     | ❌ 否  | MaxPool | Conv stride=2 | 15-20%   |"
echo "| ResNetADownStages (从头训练)| 48     | ❌ 否  | MaxPool | ADown         | 18-23%   |"
echo ""
echo "================================================================"
echo ""
echo "🔍 关键对比点："
echo "  1. vs ResNetADownStem (15.1%):"
echo "     ✅ Stem保留MaxPool（避免破坏预训练基础）"
echo "     ✅ 但Stage仍用ADown（夜间特性优化）"
echo ""
echo "  2. vs 标准从头训练 (15-20%):"
echo "     ✅ 相同训练条件（48 epochs，无预训练）"
echo "     ✅ ADown的夜间优化参数（参考YOLO11-V13）"
echo "     ✅ 逐层调优的门控策略"
echo ""
echo "================================================================"
echo ""
echo "📂 查看结果："
echo "  训练日志: work_dirs/exdark_adown_stages_scratch_2x/"
echo "  最佳模型: work_dirs/exdark_adown_stages_scratch_2x/best_coco_bbox_mAP_epoch_*.pth"
echo ""
echo "🧪 测试模型："
echo "  python tools/test.py \\"
echo "    configs/faster_rcnn/faster-rcnn_r50-adown-stages_fpn_scratch_2x_exdark.py \\"
echo "    work_dirs/exdark_adown_stages_scratch_2x/best_coco_bbox_mAP_epoch_*.pth"
echo ""
echo "📊 对比训练曲线："
echo "  tensorboard --logdir work_dirs/"
echo ""
echo "================================================================"
