# -*- coding: utf-8 -*-
"""金融领域语料库 · 继续预训练（Continued Pre-Training, CPT）训练脚本。

目标：把本语料库的金融领域知识继续预训练进一个基座模型（base model）。

设计要点
--------
1. 流式读取：语料 1.1 GB / 约 11.8 亿 token，全量载入内存会 OOM。
   本脚本用「流式读 -> 在线 tokenize -> 定长 packing」，内存占用与语料规模无关。
2. 定长 packing：把连续文档 token 拼成固定长度（默认 2048）的训练序列，
   避免 padding 浪费；跨文档边界插入 EOS。
3. 泛化：全参微调 / LoRA 微调二选一；bf16 / fp16 / 纯 CPU 均可。
4. 可自检：--dry_run 只跑数据管道（不需要 GPU），用于上线前验证。

最小用法
--------
    # 0) 先自检数据管道（不需要 GPU）
    python train/train_cpt.py --dry_run --data_dir data --tokenizer_name_or_path ./models/Qwen2.5-1.5B

    # 1) 正式训练（单卡 LoRA，约 24 GB 显存）
    python train/train_cpt.py \
        --model_name_or_path ./models/Qwen2.5-1.5B \
        --data_dir data --output_dir out/cpt-qwen1.5b-lora \
        --use_lora --max_seq_len 2048 --max_steps 20000

环境依赖见 train/requirements-train.txt。
"""
from __future__ import annotations

import argparse
import gzip
import inspect
import json
import math
import os
import random
import sys
from typing import Dict, Iterator, List, Optional

# ----------------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------------
TRAIN_FILE = "finance_zh.train.jsonl.gz"
VAL_FILE = "finance_zh.val.jsonl.gz"


# ----------------------------------------------------------------------------------
# 数据读取（标准库实现，先于 torch 导入，便于单独调试）
# ----------------------------------------------------------------------------------
def _open_text(path: str):
    """按后缀自动选择 gzip / 普通文本打开。"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def iter_records(path: str, max_docs: Optional[int] = None) -> Iterator[Dict]:
    """流式产出 JSONL 记录，跳过空行与坏行（坏行计数后跳过，不中断）。"""
    if not os.path.exists(path):
        raise FileNotFoundError("找不到数据文件：%s" % path)
    n = 0
    bad = 0
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            yield rec
            n += 1
            if max_docs is not None and n >= max_docs:
                break
    if bad:
        print("[warn] %s 跳过 %d 行非法 JSON" % (os.path.basename(path), bad), file=sys.stderr)


def resolve_files(data_dir: str, which: str) -> str:
    """定位数据文件；兼容 .jsonl.gz 与 .jsonl 两种形态。"""
    stem = TRAIN_FILE if which == "train" else VAL_FILE
    cands = [
        os.path.join(data_dir, stem),
        os.path.join(data_dir, stem[:-3]),          # 去掉 .gz
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "在 %s 下找不到 %s（尝试过：%s）" % (data_dir, stem, ", ".join(cands))
    )


def build_train_text(rec: Dict, join_title: bool = True) -> str:
    """把一条语料记录转成训练文本。"""
    text = (rec.get("text") or "").strip()
    title = (rec.get("title") or "").strip()
    if join_title and title:
        return title + "\n" + text
    return text


def tokenizer_upper_bound(args, tokenizer) -> int:
    """计算合法的 token id 上界（开区间）。

    注意：token id 的合法范围并不等于 tokenizer.vocab_size——
    Qwen 系分词器的 eos（<|endoftext|>）等特殊 token 的 id 会等于甚至略大于 vocab_size，
    而模型词表（config.vocab_size）通常是更大的值。因此取三者的最大值，
    若给了 --model_name_or_path 则进一步用模型词表作为上界。
    """
    upper = max(
        tokenizer.vocab_size,
        len(tokenizer),
        (tokenizer.eos_token_id or 0) + 1,
    )
    path = args.model_name_or_path
    if path and os.path.exists(path):
        try:
            from transformers import AutoConfig
            upper = max(upper, AutoConfig.from_pretrained(path, trust_remote_code=True).vocab_size)
        except Exception:
            pass
    return upper


# ----------------------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="金融领域语料库 CPT 训练（流式 + packing）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 模型与数据
    p.add_argument("--model_name_or_path", default=None,
                   help="基座模型路径或 ModelScope/HF 模型 ID（--dry_run 时可省略）")
    p.add_argument("--tokenizer_name_or_path", default=None,
                   help="分词器路径；默认与 --model_name_or_path 相同")
    p.add_argument("--data_dir", default="data", help="数据目录（含 train/val 两个文件）")
    p.add_argument("--output_dir", default="out/cpt", help="输出目录（模型权重与日志）")
    p.add_argument("--packed_dir", default=None,
                   help="可选：tokenize_and_pack.py 产出的预打包目录，提供后跳过在线分词")

    # 序列与打包
    p.add_argument("--max_seq_len", type=int, default=2048, help="训练序列长度")
    p.add_argument("--no_join_title", action="store_true",
                   help="默认会拼接「标题 + 换行 + 正文」，加此开关则只用正文")
    p.add_argument("--shuffle_buffer", type=int, default=2000,
                   help="流式洗牌缓冲区大小（条），0 表示不洗牌")

    # 训练规模
    p.add_argument("--max_steps", type=int, default=20000, help="总训练步数（流式训练下必须给定）")
    p.add_argument("--max_docs", type=int, default=None, help="最多读取多少篇文档（调试用，默认全量）")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5, help="CPT 建议 1e-5 ~ 2e-5")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--eval_docs", type=int, default=2000,
                   help="验证集最多取多少篇文档用于算 loss / perplexity")
    p.add_argument("--seed", type=int, default=42)

    # 精度与显存
    p.add_argument("--bf16", action="store_true", help="Ampere 及以上推荐")
    p.add_argument("--fp16", action="store_true", help="较老显卡使用")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="省显存（约再省 30%%~40%% 激活显存），会略降速")

    # LoRA
    p.add_argument("--use_lora", action="store_true", help="改用 LoRA 微调（显存友好）")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="all",
                   help="all=自动匹配 q/k/v/o/gate/up/down 等线性层")

    # 运行模式
    p.add_argument("--dry_run", action="store_true",
                   help="只验证数据管道（读数据 / 分词 / packing），不加载模型、不训练")
    p.add_argument("--dry_run_docs", type=int, default=50, help="--dry_run 读取的文档数")
    p.add_argument("--report_to", default="none", help="none / tensorboard / wandb")

    args = p.parse_args(argv)

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    if not args.dry_run and not args.model_name_or_path:
        p.error("非 --dry_run 模式下必须提供 --model_name_or_path")
    return args


# ----------------------------------------------------------------------------------
# 数据集：流式 packing（训练）
# ----------------------------------------------------------------------------------
def build_stream_dataset(args, tokenizer):
    """构造流式定长 packing 数据集（内存占用恒定，与语料规模无关）。"""
    import torch

    IterableDataset = torch.utils.data.IterableDataset
    seq_len = args.max_seq_len
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("分词器缺少 eos_token，无法做文档边界标记；请检查 tokenizer 配置")
    join_title = not args.no_join_title
    buf_size = int(args.shuffle_buffer or 0)

    class PackedStream(IterableDataset):
        """把语料流式拼成固定长度序列；每条样本含 input_ids / attention_mask / labels。

        注意：本数据集不做多 worker 分片，请在 DataLoader 中使用 num_workers=0
        （本脚本的 TrainingArguments 已固定为 0），否则各 worker 会重复读全量数据。
        """

        def __init__(self):
            super().__init__()
            self.path = resolve_files(args.data_dir, "train")

        def _texts(self):
            for rec in iter_records(self.path, max_docs=args.max_docs):
                t = build_train_text(rec, join_title=join_title)
                if t:
                    yield t

        def __iter__(self):
            buf: List[int] = []

            def consume(text: str):
                """把一条文本切成 token、补 EOS、填入缓冲，产出所有满长序列。"""
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                ids.append(eos_id)
                buf.extend(ids)
                while len(buf) >= seq_len:
                    chunk = buf[:seq_len]
                    del buf[:seq_len]
                    yield {
                        "input_ids": chunk,
                        "attention_mask": [1] * seq_len,
                        "labels": list(chunk),
                    }

            if buf_size > 0:
                rng = random.Random(args.seed)
                window: List[str] = []
                for text in self._texts():
                    window.append(text)
                    if len(window) < buf_size:
                        continue
                    yield from consume(window.pop(rng.randrange(len(window))))
                # 冲刷缓冲区，避免尾部文档被静默丢弃
                while window:
                    yield from consume(window.pop(rng.randrange(len(window))))
            else:
                for text in self._texts():
                    yield from consume(text)

            # 末尾不足一整段时，右侧补 EOS 凑满，避免浪费（仅当已积累过半段）
            if len(buf) >= seq_len // 2:
                chunk = (buf + [eos_id] * seq_len)[:seq_len]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * seq_len,
                    "labels": list(chunk),
                }

    return PackedStream()


# ----------------------------------------------------------------------------------
# 数据集：验证集（map-style，便于 Trainer 计算 eval loss）
# ----------------------------------------------------------------------------------
def build_eval_dataset(args, tokenizer):
    """从 val 文件取有限条数，tokenize + packing，构造 map-style 数据集。"""
    import torch
    from torch.utils.data import Dataset

    path = resolve_files(args.data_dir, "val")
    seq_len = args.max_seq_len
    eos_id = tokenizer.eos_token_id
    join_title = not args.no_join_title

    pack: List[List[int]] = []
    buf: List[int] = []
    n_doc = 0
    for rec in iter_records(path, max_docs=args.eval_docs):
        text = build_train_text(rec, join_title=join_title)
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)
        buf.extend(ids)
        n_doc += 1
        while len(buf) >= seq_len:
            pack.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(pack) >= 200:
            break

    if not pack:
        raise ValueError(
            "验证集未产出任何 %d 长序列，请调小 --max_seq_len 或增大 --eval_docs" % seq_len
        )

    class ListDataset(Dataset):
        def __len__(self):
            return len(pack)

        def __getitem__(self, i):
            ids = pack[i]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor([1] * seq_len, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long),
            }

    print("[eval] 验证集：%d 篇文档 -> %d 条 %d 长序列" % (n_doc, len(pack), seq_len))
    return ListDataset()


# ----------------------------------------------------------------------------------
# Trainer（兼容新旧版 transformers 的参数名差异）
# ----------------------------------------------------------------------------------
def build_training_arguments(args):
    """构造 TrainingArguments，并兼容不同 transformers 版本的参数差异。

    已知差异（实测）：
      - transformers 5.x 移除了 warmup_ratio / overwrite_output_dir，
        只保留 warmup_steps；本函数会自动把比例换算成步数。
      - transformers 4.x 用 evaluation_strategy，5.x 用 eval_strategy。
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__).parameters

    # 预热：优先用比例，其次换算成步数
    if "warmup_ratio" in sig:
        warmup_kw = {"warmup_ratio": args.warmup_ratio}
    elif "warmup_steps" in sig:
        warmup_kw = {"warmup_steps": max(1, int(args.max_steps * args.warmup_ratio))}
    else:
        warmup_kw = {}

    kw = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=0,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
        remove_unused_columns=False,
    )
    kw.update(warmup_kw)

    # 老版本 transformers 用 evaluation_strategy
    if "eval_strategy" not in sig and "evaluation_strategy" in sig:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")

    # 丢弃当前版本不支持的参数（如 5.x 的 overwrite_output_dir），并提示
    dropped = [k for k in kw if k not in sig]
    for k in dropped:
        kw.pop(k)
    if dropped:
        print("[warn] 当前 transformers 版本不支持以下训练参数，已忽略：%s" % ", ".join(dropped))

    return TrainingArguments(**kw)


def build_trainer(args, model, tokenizer, train_ds, eval_ds):
    from transformers import DataCollatorForLanguageModeling, Trainer

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    sig = inspect.signature(Trainer.__init__).parameters
    common = dict(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    # transformers 5.x 起，Trainer 用 processing_class 取代 tokenizer 参数
    if "processing_class" in sig:
        return Trainer(processing_class=tokenizer, **common)
    return Trainer(tokenizer=tokenizer, **common)


# ----------------------------------------------------------------------------------
# LoRA
# ----------------------------------------------------------------------------------
def attach_lora(args, model):
    from peft import LoraConfig, TaskType, get_peft_model

    target = None
    if args.lora_target_modules and args.lora_target_modules != "all":
        target = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    if target is None:
        # 通用规则：匹配常见注意力与 MLP 线性层，覆盖 Llama / Qwen / Mistral 等架构
        names = set()
        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.__class__.__name__ == "Linear":
                names.add(n.split(".")[-1])
        target = sorted(n for n in names if n in {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "c_attn", "c_proj", "c_fc", "query", "key", "value", "dense",
        })
    if not target:
        raise RuntimeError("未能自动识别 LoRA 目标层，请用 --lora_target_modules 显式指定")

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target,
        bias="none",
    )
    print("[lora] target_modules = %s" % ", ".join(target))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------------
def run_dry_run(args, tokenizer):
    """只验证数据管道：读取 -> 分词 -> packing -> 形状与数值断言。"""
    import itertools

    print("=" * 68)
    print("[dry_run] 数据管道自检（不加载模型、不训练）")
    print("=" * 68)

    # dry_run 只做小规模验证，避免扫描全量数据
    args.max_docs = args.dry_run_docs
    args.shuffle_buffer = 0
    args.eval_docs = min(args.eval_docs, 200)

    train_path = resolve_files(args.data_dir, "train")
    val_path = resolve_files(args.data_dir, "val")
    print("训练文件：%s" % train_path)
    print("验证文件：%s" % val_path)

    eos = tokenizer.eos_token_id
    upper = tokenizer_upper_bound(args, tokenizer)
    print("[tok] vocab_size=%d  len(tokenizer)=%d  eos_token_id=%s  合法上界=%d"
          % (tokenizer.vocab_size, len(tokenizer), eos, upper))
    assert eos is not None and eos < upper, (
        "分词器 eos_token_id=%s 超出合法上界 %d，分词器与模型可能不匹配" % (eos, upper))

    n_doc = 0
    n_char = 0
    for rec in iter_records(train_path, max_docs=args.dry_run_docs):
        t = build_train_text(rec, join_title=not args.no_join_title)
        assert t, "第 %d 条正文为空" % n_doc
        n_doc += 1
        n_char += len(t)
    assert n_doc == args.dry_run_docs, "期望读到 %d 篇，实际 %d 篇" % (args.dry_run_docs, n_doc)
    print("[ok] 训练集读取 %d 篇，共 %d 字" % (n_doc, n_char))

    ds = build_stream_dataset(args, tokenizer)
    it = iter(ds)
    seqs = [next(it) for _ in range(5)]
    L = args.max_seq_len
    for i, s in enumerate(seqs):
        assert len(s["input_ids"]) == L, "第 %d 条序列长度 %d != %d" % (i, len(s["input_ids"]), L)
        assert len(s["attention_mask"]) == L
        assert len(s["labels"]) == L
        assert all(isinstance(x, int) for x in s["input_ids"]), "input_ids 必须为 int"
        assert min(s["input_ids"]) >= 0, "出现负数 token id"
        assert max(s["input_ids"]) < upper, (
            "第 %d 条序列出现越界 token id（最大 %d >= 上界 %d），分词器与模型可能不匹配"
            % (i, max(s["input_ids"]), upper))
    print("[ok] 流式 packing 产出 5 条 %d 长序列，token id 全部合法" % L)
    preview = tokenizer.decode(seqs[0]["input_ids"], skip_special_tokens=True)
    print("     首条序列解码预览（前 60 字）：%s" % preview[:60].replace("\n", " "))

    ev = build_eval_dataset(args, tokenizer)
    assert len(ev) > 0
    one = ev[0]
    assert tuple(one["input_ids"].shape) == (L,)
    print("[ok] 验证集构造完成，共 %d 条序列" % len(ev))

    print("=" * 68)
    print("[dry_run] 全部通过，可以开始正式训练。")
    print("=" * 68)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from transformers import AutoTokenizer, set_seed
    set_seed(args.seed)

    tok_path = args.tokenizer_name_or_path
    if not tok_path:
        print("[error] 需要 --tokenizer_name_or_path（--dry_run 亦需要分词器）", file=sys.stderr)
        return 2
    print("[tok] 加载分词器：%s" % tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("分词器缺少 eos_token")

    if args.dry_run:
        run_dry_run(args, tokenizer)
        return 0

    # 模型
    import torch
    from transformers import AutoModelForCausalLM

    print("[model] 加载基座模型：%s" % args.model_name_or_path)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    # transformers 5.x 把 torch_dtype 改名为 dtype，这里做双向兼容
    _fp_sig = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    _dtype_kw = {"dtype": dtype} if "dtype" in _fp_sig else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, **_dtype_kw
    )
    model.config.use_cache = not args.gradient_checkpointing

    # 前置检查：分词器与模型词表必须匹配，否则训练时会报 index out of range
    vocab = getattr(model.config, "vocab_size", None)
    if vocab and (tokenizer.eos_token_id is None or tokenizer.eos_token_id >= vocab):
        raise ValueError(
            "分词器 eos_token_id=%s 超出模型词表大小 %s，请使用与该模型配套的分词器"
            % (tokenizer.eos_token_id, vocab)
        )

    if args.use_lora:
        model = attach_lora(args, model)

    # 数据
    if args.packed_dir and os.path.isdir(args.packed_dir):
        from datasets import load_from_disk
        print("[data] 使用预打包数据集：%s" % args.packed_dir)
        train_ds = load_from_disk(args.packed_dir)
    else:
        if args.packed_dir:
            print("[warn] --packed_dir 不存在，回退为在线流式分词：%s" % args.packed_dir,
                  file=sys.stderr)
        train_ds = build_stream_dataset(args, tokenizer)
    eval_ds = build_eval_dataset(args, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer = build_trainer(args, model, tokenizer, train_ds, eval_ds)

    print("=" * 68)
    print("[train] max_steps=%d  seq_len=%d  bs=%d x ga=%d  lr=%g  lora=%s"
          % (args.max_steps, args.max_seq_len,
             args.per_device_train_batch_size, args.gradient_accumulation_steps,
             args.learning_rate, args.use_lora))
    print("[train] 输出目录：%s" % args.output_dir)
    print("=" * 68)
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[done] 模型已保存到 %s" % args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
