#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLFEC-Finance 评分器（字符级精确跨度匹配）。

用法:
    python3 score_clfec.py --split fec_only --pred pred.jsonl
    # pred.jsonl 每行: {"sample_id": "...", "corrected_text": "..."}

指标:
  - 段落级检测  : 预测是否"有错"(pred != input) vs 标注是否"有错"(split != no_error)
                  → Precision / Recall / F1 / 误报率(对 no_error 段)
  - 编辑级纠正  : 由 (input -> pred) 抽取字符编辑跨度，与 gold 的 (start,end,candidate)
                  做精确匹配 → Precision / Recall / F1
  - 段落精确匹配: pred == gold 的比例（EM）
"""

import argparse
import difflib
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
SPLITS = ["mix", "fec_only", "lec_only", "no_error"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_edits(src, dst):
    """用 SequenceMatcher 从 src->dst 抽取字符级编辑，返回 {(start,end,replacement)}。"""
    edits = set()
    sm = difflib.SequenceMatcher(a=src, b=dst, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.add((i1, i2, dst[j1:j2]))
    return edits


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--pred", required=True, help="预测文件 {sample_id, corrected_text}")
    args = ap.parse_args()

    inputs = {r["sample_id"]: r["input_text"]
              for r in read_jsonl(os.path.join(DATA, f"inputs.{args.split}.jsonl"))}
    golds = {r["sample_id"]: r for r in read_jsonl(os.path.join(DATA, f"gold.{args.split}.jsonl"))}
    preds = {r["sample_id"]: r["corrected_text"] for r in read_jsonl(args.pred)}

    missing = [i for i in inputs if i not in preds]
    if missing:
        print(f"[警告] 缺少 {len(missing)} 条预测，例如 {missing[:3]}")

    det_tp = det_fp = det_fn = 0
    edit_tp = edit_fp = edit_fn = 0
    em = 0
    n = 0
    for sid, text in inputs.items():
        if sid not in preds:
            continue
        n += 1
        pred = preds[sid]
        g = golds[sid]
        gold_has_err = g["split"] != "no_error"
        pred_has_err = pred != text
        det_tp += (gold_has_err and pred_has_err)
        det_fp += (not gold_has_err and pred_has_err)
        det_fn += (gold_has_err and not pred_has_err)

        g_edits = {(c["start"], c["end"], c["candidate_word"]) for c in g["cors"]}
        p_edits = extract_edits(text, pred)
        edit_tp += len(g_edits & p_edits)
        edit_fp += len(p_edits - g_edits)
        edit_fn += len(g_edits - p_edits)

        em += (pred == g["corrected_text"])

    p, r, f = prf(det_tp, det_fp, det_fn)
    ep, er, ef = prf(edit_tp, edit_fp, edit_fn)
    fpr = det_fp / n if n else 0.0

    print(f"拆分 = {args.split}   样本 = {n}")
    print("-" * 46)
    print(f"[段落级检测]  P={p:.4f}  R={r:.4f}  F1={f:.4f}  误报率={fpr:.4f}")
    print(f"             TP={det_tp} FP={det_fp} FN={det_fn}")
    print(f"[编辑级纠正]  P={ep:.4f}  R={er:.4f}  F1={ef:.4f}")
    print(f"             TP={edit_tp} FP={edit_fp} FN={edit_fn}")
    print(f"[段落精确匹配] EM = {em}/{n} = {em/n*100:.2f}%" if n else "")
    print("-" * 46)
    print("提示：类别不均衡时不要只看准确率。全答“有错”在 mix/fec_only/lec_only")
    print("      上的检测基线接近 100%，必须结合误报率(no_error)与编辑级 P/R/F1。")


if __name__ == "__main__":
    main()
