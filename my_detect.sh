#!/bin/bash

RATIO="0.01"
SAMPLE_SIZE=100
TASK="detect"


MODELS=(
  "Qwen3-1.7B"
)


MODEL_BASE_PATH="/NAS/yjt/models"


for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_BASE_PATH}/${MODEL}"
  echo "🔍 Detecting neurons for: $MODEL"

  CUDA_VISIBLE_DEVICES=7 python my_detect.py \
    --base "$MODEL_PATH" \
    --atten_ratio "$RATIO" \
    --ffn_ratio "$RATIO" \
    --task_name "news_headline_generation" \
    --sample_size "$SAMPLE_SIZE" \
    --task "$TASK" \
    --corpus_path "/NAS/yjt/Abstract_Thought/dataset/LaMP"

  echo "✅ Detection complete for: $MODEL"
done

echo "🎉 All models processed for neuron detection."
