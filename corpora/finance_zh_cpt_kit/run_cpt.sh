#!/usr/bin/env bash
# ============================================================================
# 金融领域语料库 · 一键继续预训练（CPT）入口
#
# 最简用法（自动装依赖 + 自动从 ModelScope 下载模型 + 自检 + 开训）：
#     bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --lora
#
# 常用参数：
#     --model   <路径或ModelScopeID>  基座模型（必填，默认 Qwen/Qwen2.5-1.5B）
#     --out     <目录>                输出目录（默认 out/cpt）
#     --lora                          用 LoRA 微调（低显存推荐）
#     --seq-len <int>                 序列长度（默认 2048）
#     --steps   <int>                 训练步数（默认 20000）
#     --install                       自动安装 Python 依赖（含 torch）
#     --no-dry-run                    跳过训练前的数据管道自检
#     --dry-run                       只做自检，不训练
#     --model-dir <目录>              下载的模型存放目录（默认 models/）
#
# 示例：
#     # ① 先自检（不装 torch 也能跑数据管道，需分词器）
#     bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --dry-run --install
#
#     # ② 单卡 LoRA 正式训练
#     bash run_cpt.sh --model Qwen/Qwen2.5-7B --lora --install \
#                     --seq-len 2048 --steps 30000 --out out/qwen7b-lora
#
#     # ③ 用本地已有模型
#     bash run_cpt.sh --model ./models/Qwen2.5-7B --lora
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
OUT="${OUT:-out/cpt}"
MODEL_DIR="${MODEL_DIR:-models}"
USE_LORA=0
SEQ_LEN=2048
STEPS=20000
DO_INSTALL=0
DO_DRY=1
DRY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --seq-len)   SEQ_LEN="$2"; shift 2 ;;
    --steps)     STEPS="$2"; shift 2 ;;
    --lora)      USE_LORA=1; shift ;;
    --install)   DO_INSTALL=1; shift ;;
    --no-dry-run) DO_DRY=0; shift ;;
    --dry-run)   DRY_ONLY=1; shift ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[error] 未知参数：$1（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

echo "======================================================================"
echo "金融领域语料库 · 一键 CPT"
echo "基座模型 : $MODEL"
echo "输出目录 : $OUT"
echo "序列长度 : $SEQ_LEN   训练步数: $STEPS   LoRA: $USE_LORA"
echo "======================================================================"

# ---- 0. Python ----
if ! command -v python3 >/dev/null 2>&1; then
  echo "[error] 未找到 python3，请先安装 Python 3.9+" >&2
  exit 2
fi
echo "[1/6] Python: $(python3 -V 2>&1)"

# ---- 1. 数据检查 ----
TRAIN="data/finance_zh.train.jsonl.gz"
[ -f "$TRAIN" ] || TRAIN="data/finance_zh.train.jsonl"
VAL="data/finance_zh.val.jsonl.gz"
[ -f "$VAL" ] || VAL="data/finance_zh.val.jsonl"

if [ ! -s "$TRAIN" ] || [ ! -s "$VAL" ]; then
  cat >&2 <<'EOF'
[error] 找不到训练/验证数据文件。
        请先获取数据（二选一）：
          ① 直接使用本包 data/ 下的文件（本包已内置）；
          ② 自行构建：
               python3 scripts/download_corpus.py
               python3 scripts/build_corpus.py --zip <官方zip路径> --out-dir .
EOF
  exit 2
fi
echo "[2/6] 数据就绪：$TRAIN / $VAL"

# ---- 2. 依赖安装 ----
if [ "$DO_INSTALL" -eq 1 ]; then
  echo "[3/6] 安装依赖（清华源）…"
  PIPT=(python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple)
  if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "      - torch 未安装，安装 CPU 版（GPU 机器请按需换 cu121 源）"
    python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch \
      || python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  "${PIPT[@]}" -r train/requirements-train.txt
else
  echo "[3/6] 跳过依赖安装（如需自动安装请加 --install）"
fi

# ---- 3. 模型准备 ----
if [ -d "$MODEL" ]; then
  MODEL_PATH="$MODEL"
  echo "[4/6] 使用本地模型：$MODEL_PATH"
else
  SAFE="$(echo "$MODEL" | tr '/' '_')"
  MODEL_PATH="$MODEL_DIR/$SAFE"
  if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo "[4/6] 复用已下载模型：$MODEL_PATH"
  else
    echo "[4/6] 从 ModelScope 下载模型：$MODEL -> $MODEL_PATH"
    python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope >/dev/null 2>&1 || true
    python3 - "$MODEL" "$MODEL_PATH" <<'PY'
import sys
model_id, local_dir = sys.argv[1], sys.argv[2]
try:
    from modelscope import snapshot_download
except Exception:
    print("[error] modelscope 安装失败，请手动下载模型后使用 --model <本地路径>", file=sys.stderr)
    sys.exit(3)
snapshot_download(model_id, local_dir=local_dir)
print("[ok] 模型已下载到", local_dir)
PY
  fi
fi

# ---- 4. 数据管道自检 ----
if [ "$DO_DRY" -eq 1 ] || [ "$DRY_ONLY" -eq 1 ]; then
  echo "[5/6] 数据管道自检 …"
  python3 train/train_cpt.py \
    --dry_run --data_dir data \
    --tokenizer_name_or_path "$MODEL_PATH" \
    --max_seq_len "$SEQ_LEN"
fi

if [ "$DRY_ONLY" -eq 1 ]; then
  echo "[6/6] 仅自检模式，结束。"
  exit 0
fi

# ---- 5. 训练 ----
echo "[6/6] 开始训练 …"
ARGS=(--model_name_or_path "$MODEL_PATH" --data_dir data --output_dir "$OUT"
      --max_seq_len "$SEQ_LEN" --max_steps "$STEPS")
[ "$USE_LORA" -eq 1 ] && ARGS+=(--use_lora)
ARGS+=(--bf16 --gradient_checkpointing)

echo "      命令：python3 train/train_cpt.py ${ARGS[*]}"
python3 train/train_cpt.py "${ARGS[@]}"

echo "======================================================================"
echo "[done] 训练结束，模型保存在：$OUT"
echo "       推理试玩： python3 - <<'PY'"
echo "           from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "           m = AutoModelForCausalLM.from_pretrained('$OUT', device_map='auto')"
echo "           t = AutoTokenizer.from_pretrained('$OUT')"
echo "           print(t.decode(m.generate(**t('2024年A股市场', return_tensors='pt'), max_new_tokens=64)[0]))"
echo "       PY"
echo "======================================================================"
