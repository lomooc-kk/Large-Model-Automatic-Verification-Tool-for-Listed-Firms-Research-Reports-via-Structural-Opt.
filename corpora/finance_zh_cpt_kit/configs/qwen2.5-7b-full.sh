#!/usr/bin/env bash
# ============================================================================
# 配置方案 ③  大规模：Qwen2.5-7B 全参继续预训练（多卡）
#   显存参考：7B 全参约需 8 × A100-80G（可用 ZeRO / FSDP 进一步优化）
#   适用：有集群、追求最强领域能力时使用
# 用法： bash configs/qwen2.5-7b-full.sh
#      多卡请用： accelerate launch --num_processes=8 train/train_cpt.py ...
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="${OUT:-out/qwen2.5-7b-full}"
NPROC="${NPROC:-8}"

if command -v accelerate >/dev/null 2>&1 && [ "$NPROC" -gt 1 ]; then
  LAUNCH=(accelerate launch --num_processes "$NPROC")
else
  LAUNCH=(python3)
fi

"${LAUNCH[@]}" train/train_cpt.py \
  --model_name_or_path "$MODEL" \
  --data_dir data \
  --output_dir "$OUT" \
  --max_seq_len 4096 \
  --max_steps 40000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.02 \
  --weight_decay 0.1 \
  --eval_steps 1000 --save_steps 1000 --save_total_limit 3 \
  --bf16 --gradient_checkpointing
