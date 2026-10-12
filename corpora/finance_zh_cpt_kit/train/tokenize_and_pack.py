# -*- coding: utf-8 -*-
"""（可选 · 加速）预分词并打包成固定长度序列，训练时直接加载，省去每次在线分词。

适用场景
--------
- 需要反复训练 / 调参，想省掉每次启动时的分词开销；
- 多卡训练时希望数据已就绪。

不适用：只是想跑一次、磁盘紧张时。留空即可，train_cpt.py 默认在线流式分词。

用法
----
    python train/tokenize_and_pack.py \
        --data_dir data \
        --tokenizer_name_or_path ./models/Qwen2.5-1.5B \
        --out_dir packed/train \
        --max_seq_len 2048 \
        --max_seqs 200000

产出：一个 HuggingFace Dataset（save_to_disk 目录），字段
    input_ids / attention_mask / labels（均为定长 max_seq_len）。

训练时配合使用：
    python train/train_cpt.py ... --packed_dir packed/train
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

# 复用主脚本的数据读取逻辑，保证两处行为完全一致
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cpt import build_train_text, iter_records, resolve_files  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="预分词 + 定长 packing（可选加速）")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--which", default="train", choices=["train", "val"])
    p.add_argument("--tokenizer_name_or_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--max_seqs", type=int, default=200000, help="最多产出多少条序列（控制磁盘与内存）")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--no_join_title", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from transformers import AutoTokenizer
    from datasets import Dataset

    path = resolve_files(args.data_dir, args.which)
    print("[tok] 加载分词器：%s" % args.tokenizer_name_or_path)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError("分词器缺少 eos_token")

    L = args.max_seq_len
    join_title = not args.no_join_title

    ids_all: List[List[int]] = []
    mask_all: List[List[int]] = []
    buf: List[int] = []
    n_doc = 0

    print("[run] 分词 + packing（%s）…" % path)
    for rec in iter_records(path, max_docs=args.max_docs):
        text = build_train_text(rec, join_title=join_title)
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids.append(eos)
        buf.extend(ids)
        n_doc += 1
        while len(buf) >= L:
            chunk = buf[:L]
            buf = buf[L:]
            ids_all.append(chunk)
            mask_all.append([1] * L)
            if len(ids_all) >= args.max_seqs:
                break
        if len(ids_all) >= args.max_seqs:
            break
        if n_doc % 20000 == 0:
            print("      已处理 %d 篇 / 产出 %d 条序列" % (n_doc, len(ids_all)))

    if not ids_all:
        print("[error] 未产出任何序列，请调小 --max_seq_len 或增大 --max_docs", file=sys.stderr)
        return 2

    ds = Dataset.from_dict({
        "input_ids": ids_all,
        "attention_mask": mask_all,
        "labels": ids_all,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out_dir)) or ".", exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print("[done] 产出 %d 条 %d 长序列，已保存到 %s" % (len(ids_all), L, args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
