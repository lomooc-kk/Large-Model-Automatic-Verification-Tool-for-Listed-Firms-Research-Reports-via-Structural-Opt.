# LLaMA-Factory 接入说明（可选路径）

> 如果你不想用自带的 `train/train_cpt.py`，也可以用国内最流行的 LLaMA-Factory 训练。
> **注意：LLaMA-Factory 不直接读 `.jsonl.gz`，需要先把数据解压成 `.jsonl`。**

## 一、准备数据

```bash
# 在 LLaMA-Factory 仓库根目录下操作
mkdir -p data/finance_zh

# 解压（保留原文件用 -k；磁盘不足可先只解压一个子集）
gunzip -k /path/to/金融领域语料库_开箱即训版/data/finance_zh.train.jsonl.gz
gunzip -k /path/to/金融领域语料库_开箱即训版/data/finance_zh.val.jsonl.gz

mv finance_zh.train.jsonl data/finance_zh/finance_zh.train.jsonl
mv finance_zh.val.jsonl   data/finance_zh/finance_zh.val.jsonl
```

> 磁盘紧张时，可只取前 N 条做初次验证：
> `zcat xxx.jsonl.gz | head -200000 > data/finance_zh/finance_zh.train.jsonl`

## 二、注册数据集

把本目录的 `dataset_info.json` 内容**合并**到 LLaMA-Factory 的 `data/dataset_info.json`，
或把 `file_name` 改成你实际的相对路径（相对于 `dataset_dir`，默认是 `data/`）。

预训练（pt）所需的格式即为 `columns: {"prompt": "text"}`，本包已按官方规范写好：

```json
{
  "finance_zh_cpt": {
    "file_name": "finance_zh/finance_zh.train.jsonl",
    "columns": {"prompt": "text"}
  }
}
```

## 三、开始训练

```bash
llamafactory-cli train /path/to/金融领域语料库_开箱即训版/examples/llamafactory/llamafactory_cpt.yaml
```

或改用命令行模式：

```bash
llamafactory-cli train \
  --stage pt \
  --model_name_or_path Qwen/Qwen2.5-1.5B \
  --dataset finance_zh_cpt \
  --dataset_dir data \
  --cutoff_len 2048 \
  --finetuning_type lora --lora_target all \
  --output_dir saves/qwen2.5-1.5b/finance-cpt \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 --num_train_epochs 1.0 --lr_scheduler_type cosine \
  --bf16 --gradient_checkpointing --logging_steps 10 --save_steps 1000
```

## 四、注意事项

1. **防遗忘**：请把通用中文语料按 30%~50% 一起注册并用逗号并列，
   例如 `--dataset finance_zh_cpt,general_zh_mixed`；可用本包的
   `train/mix_general_corpus.py` 先生成混合文件。
2. **本地模型**：`model_name_or_path` 也可以直接写 ModelScope 下载好的本地目录。
3. **国内下载**：`export USE_MODELSCOPE_HUB=1` 可让 LLaMA-Factory 走 ModelScope 下载模型。
4. 本包的 `train/train_cpt.py` 与 LLaMA-Factory 产出的都是标准权重，二选一即可，**不要同时用**。
