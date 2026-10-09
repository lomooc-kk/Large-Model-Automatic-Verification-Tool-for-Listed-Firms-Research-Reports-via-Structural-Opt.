# -*- coding: utf-8 -*-
"""把金融领域语料与「通用中文语料」按比例混合，用于防止灾难性遗忘。

为什么需要它
------------
只用金融语料做 CPT，模型会「金融变强、通用能力退化」（灾难性遗忘）。
业界经验做法：通用语料按 **30% ~ 50%** 的比例混入。

用法
----
    # 通用语料占 40%：金融 60% + 通用 40%，输出混合文件
    python train/mix_general_corpus.py \
        --finance data/finance_zh.train.jsonl.gz \
        --general /path/to/general_zh.jsonl.gz \
        --general_ratio 0.4 \
        --out data/mixed.train.jsonl.gz

    # 只混一小批先跑通（例如总输出 20 万篇）
    python train/mix_general_corpus.py ... --max_docs 200000

支持的通用语料格式
------------------
1) JSONL（.jsonl / .jsonl.gz）：自动识别 text / content / 正文 等字段名；
2) 纯文本（.txt / .txt.gz）：**每个非空行**视为一篇。

字段名兼容表（自动探测，命中即用）：
    text, content, document, 正文, contentText, body, raw

输出格式与金融语料对齐，便于直接喂给 train_cpt.py：
    {"id": ..., "title": ..., "text": ..., "source": "mix:finance|mix:general",
     "date": null, "year": null, "n_chars": ..., "split": "train"}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from typing import Dict, Iterator, List, Optional

TEXT_KEYS = ["text", "content", "document", "正文", "contentText", "body", "raw"]


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def open_write(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "wt", encoding="utf-8")


def guess_text(rec: Dict, path: str) -> str:
    for k in TEXT_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise KeyError(
        "文件 %s 的记录里找不到可用正文字段（已尝试：%s）；"
        "实际字段：%s" % (path, ", ".join(TEXT_KEYS), ", ".join(list(rec.keys())[:12]))
    )


def iter_general(path: str) -> Iterator[Dict]:
    """产出通用语料记录，统一为 {title, text, source}。"""
    base = os.path.basename(path)
    is_txt = base.endswith(".txt") or base.endswith(".txt.gz")
    if is_txt:
        with open_text(path) as f:
            for i, line in enumerate(f):
                s = line.strip()
                if s:
                    yield {"title": "", "text": s, "source": "mix:general", "_key": "g%d" % i}
        return
    with open_text(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield {
                "title": (rec.get("title") or "").strip(),
                "text": guess_text(rec, path),
                "source": "mix:general",
                "_key": "g%d" % i,
            }


def iter_finance(path: str) -> Iterator[Dict]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield {
                "title": (rec.get("title") or "").strip(),
                "text": (rec.get("text") or "").strip(),
                "source": "mix:finance",
                "_key": rec.get("id", ""),
                "_orig_source": rec.get("source"),
                "_date": rec.get("date"),
                "_year": rec.get("year"),
            }


def counts(finance_path: str, general_path: str, max_scan: Optional[int]):
    """快速统计两边可用条数（用于把「比例」换算成「各自取多少条」）。"""
    nf = ng = 0
    for _ in iter_finance(finance_path):
        nf += 1
        if max_scan and nf >= max_scan:
            break
    for _ in iter_general(general_path):
        ng += 1
        if max_scan and ng >= max_scan:
            break
    return nf, ng


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="金融语料 + 通用语料按比例混合（防灾难性遗忘）")
    p.add_argument("--finance", required=True, help="金融语料 jsonl(.gz)")
    p.add_argument("--general", required=True, help="通用语料 jsonl(.gz) 或 txt(.gz)")
    p.add_argument("--general_ratio", type=float, default=0.4,
                   help="通用语料的目标占比，0~1，推荐 0.3~0.5")
    p.add_argument("--out", required=True, help="输出 jsonl(.gz)")
    p.add_argument("--max_docs", type=int, default=None, help="输出总条数上限（默认按金融语料全量推算）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flag_id", action="store_true",
                   help="输出时在 id 前加来源前缀，便于事后按来源筛选")
    args = p.parse_args(argv)

    if not (0.0 < args.general_ratio < 1.0):
        p.error("--general_ratio 必须落在 (0, 1) 开区间内")

    rng = random.Random(args.seed)

    print("[scan] 统计两边条数…")
    nf_avail, ng_avail = counts(args.finance, args.general, args.max_docs)
    print("       金融可用 %d 条，通用可用 %d 条" % (nf_avail, ng_avail))
    if ng_avail == 0:
        return 2

    if args.max_docs:
        total = args.max_docs
    else:
        total = nf_avail
    n_gen_target = int(round(total * args.general_ratio))
    n_fin_target = total - n_gen_target
    n_gen_target = min(n_gen_target, ng_avail)

    print("[plan] 目标：金融 %d 条 + 通用 %d 条 = %d 条（通用占比 %.1f%%）"
          % (n_fin_target, n_gen_target, n_fin_target + n_gen_target,
             100.0 * n_gen_target / max(1, n_fin_target + n_gen_target)))

    # 信用卡式按比例采样：每条按概率决定是否写出，保证流式、内存恒定
    p_fin = min(1.0, n_fin_target / max(1, nf_avail))
    p_gen = min(1.0, n_gen_target / max(1, ng_avail))

    written_f = written_g = 0
    with open_write(args.out) as w:
        for rec in iter_finance(args.finance):
            if written_f >= n_fin_target:
                break
            if rng.random() > p_fin:
                continue
            out = {
                "id": ("mix-fin-" + str(rec["_key"])) if args.flag_id else rec["_key"],
                "title": rec["title"],
                "text": rec["text"],
                "source": rec["_orig_source"] or "mix:finance",
                "date": rec.get("_date"),
                "year": rec.get("_year"),
                "n_chars": len(rec["text"]),
                "split": "train",
                "origin": "finance",
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            written_f += 1

        for rec in iter_general(args.general):
            if written_g >= n_gen_target:
                break
            if rng.random() > p_gen:
                continue
            out = {
                "id": ("mix-gen-" + str(rec["_key"])) if args.flag_id else ("gen-" + str(rec["_key"])),
                "title": rec["title"],
                "text": rec["text"],
                "source": rec["source"],
                "date": None,
                "year": None,
                "n_chars": len(rec["text"]),
                "split": "train",
                "origin": "general",
            }
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            written_g += 1

    total_out = written_f + written_g
    ratio = written_g / total_out if total_out else 0.0
    print("[done] 写出 %d 条 -> %s" % (total_out, args.out))
    print("       金融 %d 条（%.1f%%）+ 通用 %d 条（%.1f%%）"
          % (written_f, 100 * (1 - ratio), written_g, 100 * ratio))
    if written_g < n_gen_target * 0.9:
        print("[warn] 通用语料条数不足，实际只写出 %d 条（目标 %d）" % (written_g, n_gen_target),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
