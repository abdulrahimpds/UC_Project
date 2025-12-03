#!/bin/bash
# Full training script for SpaceNet7 experiments  
# 9 runs total: 3 models × 3 runs each
# UNet2D-CLSTM excluded due to legacy dependency chain (torchfcn->keras->chainer)
# Estimated time: ~460 hours (19 days) on T4 16GB
# Estimated cost: $87.40 ($0.19/hr)

echo "Starting full training pipeline..."
echo "Total experiments: 9 (3 models × 3 runs)"
echo "Models: TSViT, TSViT-ST, UNet3D"
echo "Estimated time: 460 hours (19 days)"
echo ""

# TSViT runs
echo "[1/9] Training TSViT run1..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT_run1.yaml --device 0

echo "[2/9] Training TSViT run2..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT_run2.yaml --device 0

echo "[3/9] Training TSViT run3..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT_run3.yaml --device 0

# TSViT-ST runs (batch_size=4 due to higher memory usage)
echo "[4/9] Training TSViT-ST run1..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT-ST_run1.yaml --device 0

echo "[5/9] Training TSViT-ST run2..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT-ST_run2.yaml --device 0

echo "[6/9] Training TSViT-ST run3..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/TSViT-ST_run3.yaml --device 0

# UNet3D runs
echo "[7/9] Training UNet3D run1..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/UNet3D_run1.yaml --device 0

echo "[8/9] Training UNet3D run2..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/UNet3D_run2.yaml --device 0

echo "[9/9] Training UNet3D run3..."
python train_and_eval/segmentation_training_transf.py --config configs/SpaceNet7/UNet3D_run3.yaml --device 0

echo ""
echo "All training complete!"
echo "Trained models saved to: models/saved_models/SpaceNet7/"
echo ""
echo "NOTE: UNet2D-CLSTM excluded due to legacy dependencies"
echo "Dependencies: torchfcn->keras->chainer (Python 2.x era, incompatible)"
echo "Refer to .plan/limitations.md for full details"
