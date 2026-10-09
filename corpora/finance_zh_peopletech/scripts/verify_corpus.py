#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金融领域语料库成品校验（断言式；退出码 0 = 全部通过）。

校验对象：
  1) 始终校验 samples/sample_1000.jsonl（已入库）
  2) 若存在 data/*.jsonl.gz，一并校验（本地构建后才有）

覆盖：JSON 合法性、字段规范、ID 唯一、乱码(U+FFFD)、HTML 残留、
     正文长度、日期格式、划分合法、train/val 不重叠、计数与 stats.json 一致。

用法：python3 scripts/verify_corpus.py
"""
import collections
import gzip
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "sample_1000.jsonl")
DATA_DIR = os.path.join(ROOT, "data")
STATS = os.path.join(ROOT, "stats.json")

REQ = ["id", "title", "text", "source", "date", "year", "n_chars", "split"]
MIN_CHARS = 100
TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,200}>")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REPL = "\ufffd"

fails, warns, checks = [], [], 0


def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)
        print("  [FAIL]", msg)
    return cond


def warn(msg):
    warns.append(msg)
    print("  [WARN]", msg)


def read_jsonl(path, opener=io.open):
    with opener(path, "rt", encoding="utf-8") as fh:
        for i, ln in enumerate(fh, 1):
            yield i, json.loads(ln)


def scan(path, opener, label):
    ids, splits, n = set(), collections.Counter(), 0
    for i, r in read_jsonl(path, opener):
        n += 1
        for k in REQ:
            check(k in r, "%s:%d 缺少字段 %s" % (label, i, k))
        check(r["id"] not in ids, "%s:%d ID 重复 %s" % (label, i, r["id"]))
        ids.add(r["id"])
        check(r["n_chars"] == len(r["text"]), "%s:%d n_chars 与实际不符" % (label, i))
        check(r["n_chars"] >= MIN_CHARS, "%s:%d 正文过短 %d" % (label, i, r["n_chars"]))
        check(REPL not in r["text"] and REPL not in r["title"], "%s:%d 含替换符 U+FFFD" % (label, i))
        check(not TAG_RE.search(r["text"]), "%s:%d 正文残留 HTML 标签" % (label, i))
        if r["date"] is not None:
            check(bool(DATE_RE.match(r["date"])), "%s:%d 日期格式异常 %r" % (label, i, r["date"]))
        splits[r["split"]] += 1
    return {"ids": ids, "splits": splits, "n": n}


def main():
    if not os.path.exists(SAMPLE):
        print("[FAIL] 缺少样本文件：%s" % SAMPLE)
        return 1

    print("[1] 校验样本 samples/sample_1000.jsonl ...")
    s = scan(SAMPLE, io.open, "sample")
    check(s["n"] > 0, "样本为空")
    print("    样本条数 =", s["n"], "| split 分布 =", dict(s["splits"]))

    if os.path.isdir(DATA_DIR) and os.listdir(DATA_DIR):
        print("[2] 发现 data/，校验 train/val ...")
        tr = scan(os.path.join(DATA_DIR, "finance_zh.train.jsonl.gz"), gzip.open, "train")
        va = scan(os.path.join(DATA_DIR, "finance_zh.val.jsonl.gz"), gzip.open, "val")
        inter = tr["ids"] & va["ids"]
        check(not inter, "train/val 存在 %d 个重叠 ID" % len(inter))
        print("    train=%d val=%d 重叠=%d" % (tr["n"], va["n"], len(inter)))
        if os.path.exists(STATS):
            st = json.load(io.open(STATS, encoding="utf-8"))
            t = st.get("totals", {})
            check(t.get("train") in (0, tr["n"]), "stats.train=%s 与实测 %d 不符" % (t.get("train"), tr["n"]))
            check(t.get("val") in (0, va["n"]), "stats.val=%s 与实测 %d 不符" % (t.get("val"), va["n"]))
    else:
        print("[2] 未发现 data/（正常：大文件不入库，需本地构建）")

    print("\n执行断言 %d 项 | 失败 %d | 警告 %d" % (checks, len(fails), len(warns)))
    if fails:
        print("[✗] 校验未通过")
        return 1
    print("[✓] 校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
