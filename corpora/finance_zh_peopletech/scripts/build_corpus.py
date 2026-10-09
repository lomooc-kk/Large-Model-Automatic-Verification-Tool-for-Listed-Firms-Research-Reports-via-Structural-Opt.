#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金融领域语料库（人民网科技 / ModelScope peopletech/Financial）→ 学习版中文金融语料构建。

上游：ModelScope peopletech/Financial，Apache License 2.0；revision 与 zip 哈希见 MANIFEST.json。

能力
- 直接从官方 zip 流式读取，不落地 4 GB 中间文件
- 自动修正 macOS 打包导致的 CP437 文件名乱码
- 清洗：HTML 去标签 / 实体解码 / 全角空白与多余空白归一，保留段落
- 质检：非法 UTF-8 字节（按 replace 解码后检出 U+FFFD）、非法 JSON、字段完整性、
        过短正文、跨文件精确去重，逐项计入 stats.json
- 输出：stats.json + 小样本（默认始终产出）；
        --no-full 可跳过 train/val 大文件，仅做画像与抽样
- 自检：构建结束后回读全部产物并严格按 UTF-8 校验，任何一行非法即非零退出
- 幂等：固定规则、无随机数，相同输入 → 相同输出

用法
    python3 scripts/build_corpus.py                       # 全量（含 train/val 大文件）
    python3 scripts/build_corpus.py --no-full             # 只出 stats.json + 小样本
    python3 scripts/build_corpus.py --max-rows 3000       # 冒烟测试
    python3 scripts/build_corpus.py --zip <path> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import datetime
import gzip
import hashlib
import html
import io
import json
import os
import re
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

# ---- 冻结的上游信息（改动需同步更新 MANIFEST.json） ----
ZIP_SHA256 = "9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43"
UPSTREAM_REVISION = "558891a9acb2b56fa3d85f1820724dc515ad2fd9"
ZIP_FILENAME = "金融领域语料库.zip"

MIN_CHARS = 100      # 正文最小字符数（低于此视为噪声/无效）
VAL_PERMILLE = 50    # 5% 作为 val
SAMPLE_N = 1000      # 小样本条数
REPL = "\ufffd"

TAG_RE = re.compile(r"<[^>]{0,200}?>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
WS_RE = re.compile(r"[ \t\u3000\xa0\u200b\ufeff]+")


def fix_zip_name(name: str) -> str:
    """macOS 打包时文件名未置 UTF-8 标志位，Python 会按 CP437 解码；此处还原。"""
    try:
        return name.encode("cp437").decode("utf-8")
    except Exception:
        return name


def norm_date(v):
    """兼容毫秒时间戳 / 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM:SS' 等写法，返回 YYYY-MM-DD 或 None。"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v <= 0:
            return None
        ts = v / 1000.0
        if ts > 4e9:          # 防止把“秒”当“毫秒”
            ts = v
        try:
            return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return None
    s = str(v).strip()
    if not s or s == "0":
        return None
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return "%04d-%02d-%02d" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def clean_text(raw: str) -> str:
    """HTML / 实体清洗 + 空白归一，保留段落结构。"""
    if not raw:
        return ""
    t = SCRIPT_RE.sub(" ", raw)
    t = TAG_RE.sub("\n", t)            # 标签当换行，避免文字粘连
    t = html.unescape(t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(ln.strip() for ln in (WS_RE.sub(" ", x).strip() for x in t.split("\n")) if ln.strip())


def stable_split(doc_id: str) -> str:
    """按 id 的哈希稳定划分为 train / val（无随机性，可复现）。"""
    h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < VAL_PERMILLE else "train"


def resolve_zip(explicit):
    cands = ([explicit] if explicit else []) + [
        os.path.join(ROOT, "_cache", ZIP_FILENAME),
        os.path.join(ROOT, ZIP_FILENAME),
        os.path.join(ROOT, "corpus.zip"),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("找不到上游 zip。请先运行：python3 scripts/download_corpus.py --out-dir ./_cache")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建清洗后的中文金融语料")
    ap.add_argument("--zip", default=None, help="官方 zip 路径（默认自动查找）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认脚本上级目录）")
    ap.add_argument("--no-full", action="store_true", help="不写 train/val 大文件，只出 stats.json + 小样本")
    ap.add_argument("--max-rows", type=int, default=0, help="只处理前 N 行原始数据（0=全量，用于冒烟测试）")
    ap.add_argument("--sample-n", type=int, default=SAMPLE_N, help="小样本条数（默认 %d）" % SAMPLE_N)
    args = ap.parse_args(argv)

    root = os.path.abspath(args.out_dir) if args.out_dir else ROOT
    zip_path = resolve_zip(args.zip)
    full = not args.no_full
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "samples"), exist_ok=True)

    t0 = time.time()
    z = zipfile.ZipFile(zip_path)
    members = [i for i in z.infolist()
               if fix_zip_name(i.filename).endswith(".json") and "__MACOSX" not in fix_zip_name(i.filename)]
    members.sort(key=lambda i: fix_zip_name(i.filename))
    print("[*] 待处理文件 %d 个（full=%s）" % (len(members), full), flush=True)

    stats = {
        "dataset": "金融领域语料库 (ModelScope peopletech/Financial, 人民网科技公司)",
        "license": "Apache License 2.0",
        "upstream_revision": UPSTREAM_REVISION,
        "zip_sha256": ZIP_SHA256,
        "build_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {"min_chars": MIN_CHARS, "val_permille": VAL_PERMILLE,
                   "sample_n": args.sample_n, "max_rows": args.max_rows, "full": full},
        "per_file": {}, "totals": {}, "quality": {},
    }

    seen = set()
    src_counter = collections.Counter()
    year_counter = collections.Counter()
    lens = []
    total_raw = total_kept = dup = malformed = short = empty = badchar = missing_date = 0
    per_file = {}
    sample_pool = []

    ftrain = fval = None
    if full:
        ftrain = gzip.open(os.path.join(root, "data/finance_zh.train.jsonl.gz"), "wt", encoding="utf-8", compresslevel=6)
        fval = gzip.open(os.path.join(root, "data/finance_zh.val.jsonl.gz"), "wt", encoding="utf-8", compresslevel=6)

    stop = False
    for info in members:
        fn = fix_zip_name(info.filename).split("/")[-1]
        n_raw = n_kept = n_dup = n_bad = n_short = 0
        with z.open(info) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                n_raw += 1; total_raw += 1
                if args.max_rows and total_raw > args.max_rows:
                    stop = True
                    break
                # 关键：按 replace 解码，非法字节 -> U+FFFD，随后被丢弃，保证产物始终是合法 UTF-8
                try:
                    o = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    n_bad += 1; malformed += 1
                    continue
                text = o.get("contentText")
                if text is None:
                    text = o.get("content")
                title = (o.get("title") or "").strip()
                src = (o.get("publishSource") or "").replace(REPL, "").strip()
                date = norm_date(o.get("dataTime"))
                text = clean_text(text or "")
                n_chars = len(text)
                lens.append(n_chars)
                if n_chars == 0:
                    empty += 1; n_short += 1; continue
                if n_chars < MIN_CHARS:
                    short += 1; n_short += 1; continue
                if REPL in text or REPL in title:
                    badchar += 1; continue
                fp = hashlib.sha1((title + "\x01" + text[:2000]).encode("utf-8", "ignore")).hexdigest()
                if fp in seen:
                    dup += 1; n_dup += 1; continue
                seen.add(fp)

                doc_id = "fmzh-%07d" % total_kept
                rec = {"id": doc_id, "title": title, "text": text, "source": src,
                       "date": date, "year": int(date[:4]) if date else None,
                       "n_chars": n_chars, "split": stable_split(doc_id)}
                line = json.dumps(rec, ensure_ascii=False)

                if full:
                    (fval if rec["split"] == "val" else ftrain).write(line + "\n")
                if src:
                    src_counter[src] += 1
                if date:
                    year_counter[date[:4]] += 1
                else:
                    missing_date += 1
                total_kept += 1; n_kept += 1
                if len(sample_pool) < args.sample_n:
                    sample_pool.append(line)
                else:
                    j = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % total_kept
                    if j < args.sample_n:
                        sample_pool[j] = line
                if total_kept % 100000 == 0:
                    print("  ... kept=%d raw=%d (%.1fs)" % (total_kept, total_raw, time.time() - t0), flush=True)
        per_file[fn] = {"rows_raw": n_raw, "rows_kept": n_kept, "dup": n_dup,
                        "malformed": n_bad, "too_short": n_short}
        print("文件 %s 完成: raw=%d kept=%d dup=%d bad=%d short=%d" % (fn, n_raw, n_kept, n_dup, n_bad, n_short), flush=True)
        if stop:
            print("[i] 达到 --max-rows 上限，提前结束", flush=True)
            break

    if full:
        ftrain.close(); fval.close()
    sample_path = os.path.join(root, "samples/sample_%d.jsonl" % args.sample_n)
    with io.open(sample_path, "w", encoding="utf-8") as fo:
        for ln in sample_pool:
            fo.write(ln + "\n")

    train_lines = val_lines = 0
    if full:
        train_lines = sum(1 for _ in gzip.open(os.path.join(root, "data/finance_zh.train.jsonl.gz"), "rt", encoding="utf-8"))
        val_lines = sum(1 for _ in gzip.open(os.path.join(root, "data/finance_zh.val.jsonl.gz"), "rt", encoding="utf-8"))

    lens.sort()
    pct = (lambda p: lens[min(len(lens) - 1, int(len(lens) * p))]) if lens else (lambda p: 0)
    total_chars = sum(lens)

    stats["per_file"] = per_file
    stats["totals"] = {"documents_kept": total_kept, "documents_raw": total_raw,
                       "train": train_lines, "val": val_lines, "chars": total_chars,
                       "est_tokens": int(total_chars * 0.85),
                       "distinct_publish_sources": len(src_counter)}
    stats["quality"] = {"malformed_json": malformed, "empty_text": empty,
                        "too_short_dropped": short, "duplicate_dropped": dup,
                        "replacement_char_dropped": badchar, "missing_or_invalid_date": missing_date,
                        "dup_ratio": round(dup / total_raw, 6) if total_raw else 0,
                        "keep_ratio": round(total_kept / total_raw, 6) if total_raw else 0}
    stats["text_len"] = {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                         "max": lens[-1] if lens else 0,
                         "mean": int(total_chars / total_kept) if total_kept else 0}
    stats["date_year_hist"] = dict(sorted(year_counter.items()))
    stats["top20_sources"] = src_counter.most_common(20)
    stats["elapsed_sec"] = round(time.time() - t0, 1)

    with io.open(os.path.join(root, "stats.json"), "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=2)

    # ---- 自检：严格按 UTF-8 回读全部产物 ----
    def check_jsonl(path, opener=io.open):
        bad = 0
        with opener(path, "rt", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    json.loads(ln)
                except Exception:
                    bad += 1
        return bad

    problems = []
    for p, op in ([(sample_path, io.open)] +
                  ([(os.path.join(root, "data/finance_zh.train.jsonl.gz"), gzip.open),
                    (os.path.join(root, "data/finance_zh.val.jsonl.gz"), gzip.open)] if full else [])):
        try:
            b = check_jsonl(p, op)
            if b:
                problems.append("%s: %d 行非法" % (p, b))
        except Exception as e:
            problems.append("%s: %s" % (p, e))

    print(json.dumps(stats["totals"], ensure_ascii=False))
    print(json.dumps(stats["quality"], ensure_ascii=False))
    if problems:
        print("[✗] 自检失败：")
        for p in problems:
            print("   ", p)
        return 2
    print("[✓] 自检通过：全部产物均为合法 UTF-8 JSONL")
    print("BUILD OK", stats["elapsed_sec"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
