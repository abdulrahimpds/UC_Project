#!/bin/bash
# run test set evaluation for all trained models

# set device
DEVICE="0"
OUTPUT_CSV="results/test_metrics.csv"

# remove old results file if it exists
if [ -f "$OUTPUT_CSV" ]; then
    echo "removing old results file: $OUTPUT_CSV"
    rm "$OUTPUT_CSV"
fi

echo "starting evaluation of all models on test set..."
echo "results will be saved to: $OUTPUT_CSV"
echo ""

# tsvit models (temporal-first)
echo "evaluating tsvit models..."
python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT_run1.yaml \
    --checkpoint models/saved_models/TSViT/run1/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT_run1

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT_run2.yaml \
    --checkpoint models/saved_models/TSViT/run2/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT_run2

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT_run3.yaml \
    --checkpoint models/saved_models/TSViT/run3/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT_run3

# tsvit-st models (spatial-first)
echo ""
echo "evaluating tsvit-st models..."
python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT-ST_run1.yaml \
    --checkpoint models/saved_models/TSViT-ST/run1/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT-ST_run1

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT-ST_run2.yaml \
    --checkpoint models/saved_models/TSViT-ST/run2/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT-ST_run2

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/TSViT-ST_run3.yaml \
    --checkpoint models/saved_models/TSViT-ST/run3/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name TSViT-ST_run3

# unet3d models (baseline)
echo ""
echo "evaluating unet3d models..."
python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/UNet3D_run1.yaml \
    --checkpoint models/saved_models/UNet3D/run1/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name UNet3D_run1

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/UNet3D_run2.yaml \
    --checkpoint models/saved_models/UNet3D/run2/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name UNet3D_run2

python scripts/evaluate_on_test.py \
    --config configs/SpaceNet7/UNet3D_run3.yaml \
    --checkpoint models/saved_models/UNet3D/run3/best.pth \
    --device $DEVICE \
    --output $OUTPUT_CSV \
    --model_name UNet3D_run3

echo ""
echo "all evaluations complete!"
echo "results saved to: $OUTPUT_CSV"
echo ""
echo "="*80
echo "running analysis on results..."
echo "="*80

# run analysis script
python scripts/analyze_results.py \
    --input $OUTPUT_CSV \
    --output_dir results/analysis

echo ""
echo "="*80
echo "pipeline complete!"
echo "="*80
echo ""
echo "outputs saved to:"
echo "  - raw metrics: $OUTPUT_CSV"
echo "  - summary tables: results/analysis/summary_table.csv"
echo "  - statistical tests: results/analysis/statistical_tests.csv"
echo "  - visualizations: results/analysis/*.png"
echo ""