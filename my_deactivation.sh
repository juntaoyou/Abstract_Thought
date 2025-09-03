#!/bin/bash

# 模型列表
MODELS=(
  Qwen3-1.7B
)


RATIO="0.01"
NEURONS_PATH="./neuron_deactivation"
SAVE_PATH="./deactivate_model_param"
MODEL_PATH="/NAS/yjt/models"

for MODEL in "${MODELS[@]}"; do
  echo "🔧 Processing model: $MODEL"

  CUDA_VISIBLE_DEVICES=2 python my_deactivation.py \
    --model_name "$MODEL" \
    --ratio "$RATIO" \
    --neurons_path "$NEURONS_PATH" \
    --save_path "$SAVE_PATH" \
    --model_path "$MODEL_PATH"

  echo "✅ Done: $MODEL"
done

echo "🎉 All models processed."
