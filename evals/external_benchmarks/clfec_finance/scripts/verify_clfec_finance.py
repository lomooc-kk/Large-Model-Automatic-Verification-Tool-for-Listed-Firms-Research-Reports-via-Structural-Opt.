#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLFEC-Finance 子集自检脚本。

退出码 0 = 全部断言通过。任何一项不通过立即报错退出（非 0），
便于接入 CI，把"交付的数据不能有错"变成机器可校验的约束。
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(ROOT, "raw", "CLFEC.official.json")
SOURCE = os.path.join(ROOT, "raw", "SOURCE.json")

SPLITS = ["mix", "fec_only", "lec_only", "no_error"]
FINANCE_TOTAL = 268
SPLIT_COUNTS = {"mix": 113, "fec_only": 57, "lec_only": 66, "no_error": 32}
FINANCE_EDITS = 534
FINANCE_CHARS = 97329

INPUT_KEYS = {"sample_id", "input_text"}
GOLD_KEYS = {"sample_id", "source_id", "domain", "split",
             "corrected_text", "num_edits", "error_type_counts", "cors"}
ANSWER_ONLY = {"corrected_text", "cors", "candidate_word", "error_word",
               "error_type", "num_edits", "label", "answer", "gold"}

failures = []


def check(cond, msg):
    if cond:
        print(f"  [PASS] {msg}")
    else:
        print(f"  [FAIL] {msg}")
        failures.append(msg)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_cors(text, cors):
    buf = list(text)
    for c in sorted(cors, key=lambda x: x["start"], reverse=True):
        buf[c["start"]:c["end"]] = list(c["candidate_word"])
    return "".join(buf)


def main():
    src = json.load(open(SOURCE, encoding="utf-8"))

    print("1) 原始来源冻结")
    raw_sha = hashlib.sha256(open(RAW, "rb").read()).hexdigest()
    check(raw_sha == src["file_sha256"], f"raw/CLFEC.official.json sha256 == 冻结值 ({raw_sha[:16]}…)")
    check(src["license"] == "MIT", "许可为 MIT")

    inputs, golds = {}, {}
    print("\n2) 文件齐备且字段合规")
    for s in SPLITS:
        ip = os.path.join(DATA, f"inputs.{s}.jsonl")
        gp = os.path.join(DATA, f"gold.{s}.jsonl")
        check(os.path.exists(ip) and os.path.exists(gp), f"{s}: inputs/gold 均存在")
        inputs[s] = read_jsonl(ip)
        golds[s] = read_jsonl(gp)
        check(all(set(r) == INPUT_KEYS for r in inputs[s]),
              f"{s}: inputs 字段恰为 {sorted(INPUT_KEYS)}")
        check(all(set(r) == GOLD_KEYS for r in golds[s]),
              f"{s}: gold 字段恰为约定集合")

    print("\n3) 规模与官方口径一致")
    total = sum(len(v) for v in inputs.values())
    check(total == FINANCE_TOTAL, f"金融段落总数 == {FINANCE_TOTAL}（实际 {total}）")
    for s in SPLITS:
        check(len(inputs[s]) == SPLIT_COUNTS[s], f"{s}: 段落数 == {SPLIT_COUNTS[s]}（实际 {len(inputs[s])}）")
    edits = sum(g["num_edits"] for v in golds.values() for g in v)
    check(edits == FINANCE_EDITS, f"编辑总数 == {FINANCE_EDITS}（实际 {edits}）")
    chars = sum(len(r["input_text"]) for v in inputs.values() for r in v)
    check(chars == FINANCE_CHARS, f"字符总数 == {FINANCE_CHARS}（实际 {chars}）")

    print("\n4) 输入与答案严格分离（无泄漏）")
    for s in SPLITS:
        leaked = [r for r in inputs[s] if set(r) & ANSWER_ONLY]
        check(not leaked, f"{s}: inputs 不含任何答案字段")
    # sample_id 不得泄漏拆分/正误含义
    import re
    pat = re.compile(r"(clean|wrong|right|error|no_error|mix|_ok|__)", re.I)
    all_ids = [r["sample_id"] for v in inputs.values() for r in v]
    check(not any(pat.search(i) for i in all_ids), "sample_id 不含拆分/正误/答案语义")
    check(all(i.startswith("clfec-fin-") for i in all_ids), "sample_id 统一前缀 clfec-fin-")

    print("\n5) ID 全局唯一且 inputs/gold 一一对应")
    check(len(all_ids) == len(set(all_ids)), f"sample_id 全局唯一（{len(set(all_ids))}/{len(all_ids)}）")
    for s in SPLITS:
        a = [r["sample_id"] for r in inputs[s]]
        b = [r["sample_id"] for r in golds[s]]
        check(a == b, f"{s}: inputs 与 gold 的 sample_id 顺序完全一致")
        check(len(a) == len(set(a)), f"{s}: 拆分内 id 无重复")

    print("\n6) 数据正确性：编辑可精确还原标准答案")
    bad_apply, span_bad, noerr_bad, unchanged = [], [], [], []
    for s in SPLITS:
        imap = {r["sample_id"]: r["input_text"] for r in inputs[s]}
        for g in golds[s]:
            text = imap[g["sample_id"]]
            for c in g["cors"]:
                if text[c["start"]:c["end"]] != c["error_word"]:
                    span_bad.append((g["sample_id"], c))
            if apply_cors(text, g["cors"]) != g["corrected_text"]:
                bad_apply.append(g["sample_id"])
            if g["split"] == "no_error":
                if text != g["corrected_text"] or g["cors"]:
                    noerr_bad.append(g["sample_id"])
            else:
                if text == g["corrected_text"]:
                    unchanged.append(g["sample_id"])
    check(not span_bad, f"所有编辑的 [start:end] 与 error_word 一致（违规 {len(span_bad)}）")
    check(not bad_apply, f"所有样本的 cors 可还原 corrected_text（违规 {len(bad_apply)}）")
    check(not noerr_bad, f"no_error 样本 input==gold 且无编辑（违规 {len(noerr_bad)}）")
    check(not unchanged, f"有错样本的 input != gold（违规 {len(unchanged)}）")

    print("\n7) 类别不均衡基线提示")
    n_err = sum(len(inputs[s]) for s in SPLITS if s != "no_error")
    print(f"  有错样本 {n_err} / {total}；全答“有错”的多数类基线 = {n_err/total*100:.2f}%")

    print("\n8) 与 MANIFEST 一致性")
    man = json.load(open(os.path.join(DATA, "MANIFEST.json"), encoding="utf-8"))
    check(man["filter"]["kept"] == total, "MANIFEST kept == 实际段落数")
    check(man["totals"]["edits"] == edits, "MANIFEST 编辑数 == 实际编辑数")

    print("\n" + "=" * 56)
    if failures:
        print(f"自检失败：{len(failures)} 项未通过")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("自检通过：全部断言成立（退出码 0）")


if __name__ == "__main__":
    main()
