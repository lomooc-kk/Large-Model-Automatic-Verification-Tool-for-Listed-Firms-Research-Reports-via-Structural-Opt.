#!/usr/bin/env bash
# ============================================================================
# 配置方案 ①  小试 / 单卡低显存：Qwen2.5-1.5B + LoRA
#   显存参考：约 8 ~ 16 GB（开启 gradient checkpointing）
#   适用：先跑通、验证数据与流程、快速迭代
# 用法： bash configs/qwen2.5-1.5b-lora.sh
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
OUT="${OUT:-out/qwen2.5-1.5b-lora}"

python3 train/train_cpt.py \
  --model_name_or_path "$MODEL" \
  --data_dir data \
  --output_dir "$OUT" \
  --use_lora --lora_r 64 --lora_alpha 128 \
  --max_seq_len 2048 \
  --max_steps 20000 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --eval_steps 500 --save_steps 500 \
  --bf16 --gradient_checkpointing
