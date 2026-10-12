#!/usr/bin/env bash
# ============================================================================
# 配置方案 ②  正式推荐：Qwen2.5-7B + LoRA
#   显存参考：约 24 GB（bs=1 + 梯度累积 + gradient checkpointing）
#   适用：把金融知识灌注进 7B 模型的主力方案
# 用法： bash configs/qwen2.5-7b-lora.sh
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="${OUT:-out/qwen2.5-7b-lora}"

python3 train/train_cpt.py \
  --model_name_or_path "$MODEL" \
  --data_dir data \
  --output_dir "$OUT" \
  --use_lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --max_seq_len 2048 \
  --max_steps 30000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 8e-6 \
  --warmup_ratio 0.03 \
  --eval_steps 1000 --save_steps 1000 --save_total_limit 3 \
  --bf16 --gradient_checkpointing
