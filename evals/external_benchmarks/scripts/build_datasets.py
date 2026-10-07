#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建"金融文本一致性 / 纠错"数据集。

输入 : sources/ 下三个基准的原始下载（FinanceBench / FinVerBench / FinBen）
输出 : data/  下的标准化 JSONL

运行 : python3 scripts/build_datasets.py
依赖 : pandas, pyarrow （仅 FinBen 部分需要）
"""
import glob
import json
import os
import re
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sources")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def jdump(path, rows, gzip_threshold_mb=20):
    """把 list[dict] 写成 JSONL（UTF-8，每行一条）。
    若内容超过 gzip_threshold_mb，则写成 .jsonl.gz 以控制体积。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    if len(payload) > gzip_threshold_mb * 1024 * 1024:
        import gzip
        path = path + ".gz"
        with gzip.open(path, "wb") as f:
            f.write(payload)
    else:
        with open(path, "wb") as f:
            f.write(payload)
    print("[write] {:<58s} {:>6d} rows  {:>7.1f} MB".format(
        os.path.relpath(path, ROOT), len(rows), (os.path.getsize(path)) / 1e6))


def to_jsonable(v):
    """把 numpy / pandas 类型递归转换为可 JSON 序列化的原生类型。"""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    try:  # numpy 标量
        return v.item()
    except Exception:
        pass
    try:  # numpy 数组 / pandas Series
        return [to_jsonable(x) for x in list(v)]
    except Exception:
        return str(v)


# ----------------------------------------------------------------------------
# 1. FinVerBench —— 财务报表内部一致性判定 + 纠错对
#    原始实例已自带 干净(clean) / 注入错误(arithmetic, cross-statement,
#    year-over-year, magnitude) 标签，是最贴近"一致性纠错"任务的现成样本。
# ----------------------------------------------------------------------------
def build_finverbench():
    path = os.path.join(SRC, "FinVerBench/data/benchmark/benchmark.json")
    if not os.path.exists(path):
        print("[skip] FinVerBench benchmark.json 不存在")
        return [], []
    instances = json.load(open(path, encoding="utf-8"))

    clean_by_company = {}
    for it in instances:
        if it.get("instance_id", "").endswith("__clean"):
            clean_by_company[it.get("company")] = it

    detection, correction = [], []
    for it in instances:
        gt = it.get("ground_truth") or {}
        has_err = bool(gt.get("has_error"))
        det = OrderedDict()
        det["sample_id"] = "finverbench/" + it["instance_id"]
        det["source"] = "FinVerBench"
        det["task"] = "financial_statement_consistency_detection"
        det["language"] = "en"
        det["company"] = it.get("company")
        det["period"] = it.get("period")
        det["instruction"] = it.get("question")
        det["context"] = it.get("formatted_statements")
        det["label"] = 1 if has_err else 0
        det["label_text"] = "inconsistent" if has_err else "consistent"
        det["error_category"] = gt.get("error_category") or (None if has_err else "none")
        det["error_category_cn"] = ERROR_CAT_CN.get(gt.get("error_category"), None)
        det["error_type"] = gt.get("error_type")
        det["error_location"] = gt.get("error_location")
        det["error_magnitude_pct"] = gt.get("error_magnitude_pct")
        det["original_value"] = gt.get("original_value")
        det["modified_value"] = gt.get("modified_value")
        det["error_description"] = gt.get("description")
        det["difficulty"] = it.get("difficulty")
        detection.append(det)

        if has_err:
            cl = clean_by_company.get(it.get("company"))
            if cl and cl.get("period") == it.get("period"):
                cor = OrderedDict()
                cor["sample_id"] = "finverbench_corr/" + it["instance_id"]
                cor["source"] = "FinVerBench"
                cor["task"] = "financial_statement_correction"
                cor["language"] = "en"
                cor["company"] = it.get("company")
                cor["period"] = it.get("period")
                cor["instruction"] = (
                    "The financial statements below contain internal inconsistencies. "
                    "Identify the incorrect line item(s) and output the corrected statements."
                )
                cor["corrupted_text"] = it.get("formatted_statements")
                cor["corrected_text"] = cl.get("formatted_statements")
                cor["error_category"] = gt.get("error_category")
                cor["error_category_cn"] = ERROR_CAT_CN.get(gt.get("error_category"))
                cor["error_type"] = gt.get("error_type")
                cor["error_location"] = gt.get("error_location")
                cor["original_value"] = gt.get("original_value")
                cor["modified_value"] = gt.get("modified_value")
                cor["error_description"] = gt.get("description")
                cor["difficulty"] = it.get("difficulty")
                correction.append(cor)

    jdump(os.path.join(OUT, "finverbench_consistency_detection.jsonl"), detection)
    jdump(os.path.join(OUT, "finverbench_statement_correction.jsonl"), correction)
    return detection, correction


ERROR_CAT_CN = {
    "AE": "算术错误 (合计/小计不等于各项之和)",
    "CL": "跨表勾稽错误 (同一指标在不同报表间不一致)",
    "YOY": "同比连续性错误 (期初/期末与上期对不上)",
    "MR": "量级扰动 (数值被整体放大或缩小)",
    "multi": "多重错误 (同时注入多类错误)",
    "none": "无错误 (干净样本)",
}


# ----------------------------------------------------------------------------
# 2. FinanceBench —— 基于证据的事实一致性判定 + 数值断言纠错对
# ----------------------------------------------------------------------------
NUM_TOKEN = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?%?")


def perturb_number(tok, factor):
    """把数值字符串按 factor 比例改写，尽量保持原有格式（$ 前缀、% 后缀、千分位、小数位）。"""
    m = re.match(r"^(\$?)(-?\d[\d,]*(?:\.(\d+))?)(%?)$", tok)
    if not m:
        return None
    pre, num, dec, post = m.group(1), m.group(2), m.group(3), m.group(4)
    has_comma = "," in num
    try:
        val = float(num.replace(",", ""))
    except ValueError:
        return None
    new = val * (1 + factor)
    nd = len(dec) if dec else 0
    s = "{:,.{}f}".format(abs(new), nd) if has_comma else "{:.{}f}".format(abs(new), nd)
    if new < 0:
        s = "-" + s
    return pre + s + post


def pick_perturbable(answer):
    """挑一个"看起来是金额/比例/精确实数"的数值 token，避免误改年份等普通整数。"""
    for tok in NUM_TOKEN.findall(answer or ""):
        if tok.startswith("$") or tok.endswith("%") or ("," in tok) or ("." in tok):
            return tok
    return None


def build_financebench():
    qpath = os.path.join(SRC, "FinanceBench/data/financebench_open_source.jsonl")
    mpath = os.path.join(SRC, "FinanceBench/data/financebench_document_information.jsonl")
    if not os.path.exists(qpath):
        print("[skip] FinanceBench 数据不存在")
        return [], []
    questions = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    meta = {}
    if os.path.exists(mpath):
        for l in open(mpath, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                meta[d.get("doc_name")] = d

    fact, pairs = [], []
    for q in questions:
        ev = q.get("evidence") or []
        ctx = "\n\n".join(
            (e.get("evidence_text_full_page") or e.get("evidence_text") or "") for e in ev
        )
        m = meta.get(q.get("doc_name"), {})
        s = OrderedDict()
        s["sample_id"] = "financebench/" + str(q.get("financebench_id"))
        s["source"] = "FinanceBench"
        s["task"] = "evidence_grounded_fact_consistency"
        s["language"] = "en"
        s["company"] = q.get("company")
        s["doc_name"] = q.get("doc_name")
        s["doc_type"] = m.get("doc_type")
        s["doc_period"] = m.get("doc_period")
        s["gics_sector"] = m.get("gics_sector")
        s["question_type"] = q.get("question_type")
        s["question_reasoning"] = q.get("question_reasoning")
        s["question"] = q.get("question")
        s["claim"] = q.get("answer")
        s["justification"] = q.get("justification")
        s["evidence_text"] = "\n\n".join((e.get("evidence_text") or "") for e in ev)
        s["evidence_page_num"] = [e.get("evidence_page_num") for e in ev]
        s["context"] = ctx
        fact.append(s)

        ans = q.get("answer") or ""
        tok = pick_perturbable(ans)
        if tok:
            wrong = perturb_number(tok, 0.08)
            if wrong and wrong != tok:
                p = OrderedDict()
                p["sample_id"] = "financebench_corr/" + str(q.get("financebench_id"))
                p["source"] = "FinanceBench"
                p["task"] = "numeric_claim_correction"
                p["language"] = "en"
                p["company"] = q.get("company")
                p["doc_name"] = q.get("doc_name")
                p["question"] = q.get("question")
                p["context"] = ctx
                p["correct_statement"] = ans
                p["wrong_statement"] = ans.replace(tok, wrong, 1)
                p["edit"] = {"from": tok, "to": wrong}
                p["perturbation_rule"] = "rule_based_factor=+0.08"
                p["gold_answer"] = ans
                pairs.append(p)

    jdump(os.path.join(OUT, "financebench_fact_consistency.jsonl"), fact)
    jdump(os.path.join(OUT, "financebench_correction_pairs.jsonl"), pairs)
    return fact, pairs


# ----------------------------------------------------------------------------
# 3. FinBen —— 信息抽取 / 文本分析 / 数值推理子任务标准化
#    只覆盖可公开下载的子集；受限(gated)子集见 SOURCES.md
# ----------------------------------------------------------------------------
FINBEN_TASKS = {
    "finben-finer-ord": ("named_entity_recognition", "信息抽取：金融文本命名实体识别 (PER/LOC/ORG)"),
    "finben-fomc": ("central_bank_stance_classification", "文本分析：央行(FOMC)立场分类 (hawkish/dovish/neutral)"),
    "flare-fomc": ("central_bank_stance_classification", "文本分析：央行(FOMC)立场分类（与 finben-fomc 同源）"),
    "flare-finred": ("relation_extraction", "信息抽取：金融关系抽取"),
    "flare-fnxl": ("xbrl_numeric_tagging", "信息抽取：财报数字/XBRL 标签抽取"),
    "flare-convfinqa": ("conversational_numeric_reasoning", "数值推理：财报多轮数值问答"),
    "flare-tatqa": ("table_text_numeric_reasoning", "数值推理：表格+文本数值问答"),
}


def build_finben():
    base = os.path.join(SRC, "FinBen")
    if not os.path.isdir(base):
        print("[skip] FinBen 数据不存在")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("[skip] 需要 pandas/pyarrow 才能解析 FinBen parquet")
        return []

    # 清理旧的输出，保证幂等
    for old in glob.glob(os.path.join(OUT, "finben", "*.jsonl")) + \
               glob.glob(os.path.join(OUT, "finben", "*.jsonl.gz")):
        os.remove(old)

    # 先按子任务把所有 split 累积到内存，最后每个子任务只写一次（保证幂等）
    bucket = {}
    files = sorted(glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True))
    for f in files:
        ds = os.path.relpath(f, base).split(os.sep)[0]
        if ds not in FINBEN_TASKS:
            continue
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print("[warn] 解析失败 {}: {}".format(os.path.relpath(f, ROOT), e))
            continue
        task, task_cn = FINBEN_TASKS[ds]
        split = os.path.basename(f).split("-")[0]
        bucket.setdefault(ds, [])
        for rec in df.to_dict(orient="records"):
            rec = to_jsonable(rec)
            row = OrderedDict()
            row["sample_id"] = "finben/{}/{}".format(ds, rec.get("id"))
            row["source"] = "FinBen"
            row["subset"] = ds
            row["split"] = split
            row["task"] = task
            row["task_cn"] = task_cn
            row["language"] = "en"
            row["instruction"] = rec.get("query")
            row["input"] = rec.get("text") if rec.get("text") is not None else rec.get("query")
            row["output"] = rec.get("answer")
            # 其余原始字段放进 extra，避免与上面字段重复
            used = {"id", "query", "answer", "text"}
            extra = {k: v for k, v in rec.items() if k not in used}
            if extra:
                row["extra"] = extra
            bucket[ds].append(row)

    summary = []
    for ds, rows in bucket.items():
        rows = sorted(rows, key=lambda r: str(r["sample_id"]))
        jdump(os.path.join(OUT, "finben", "{}.jsonl".format(ds)), rows)
        summary.append((ds, FINBEN_TASKS[ds][1], len(rows)))
    return summary


# ----------------------------------------------------------------------------
# 4. 统一视图 —— 把可用于"一致性判定"的样本汇总成一份
# ----------------------------------------------------------------------------
def build_unified(fv_det, fb_fact, fb_pairs):
    rows = []
    for d in fv_det:
        rows.append(OrderedDict([
            ("sample_id", d["sample_id"]),
            ("source", "FinVerBench"),
            ("task", "consistency_detection"),
            ("language", "en"),
            ("context", d["context"]),
            ("instruction", d["instruction"]),
            ("claim", None),
            ("label", d["label"]),
            ("label_text", d["label_text"]),
            ("error_category", d["error_category"]),
            ("gold_output", None),
            ("meta", {"company": d["company"], "period": d["period"], "error_type": d["error_type"]}),
        ]))
    for p in fb_pairs:
        # 错误版本 -> label 0；正确版本 -> label 1
        rows.append(OrderedDict([
            ("sample_id", p["sample_id"] + "_wrong"),
            ("source", "FinanceBench"),
            ("task", "consistency_detection"),
            ("language", "en"),
            ("context", p["context"]),
            ("instruction", p["question"]),
            ("claim", p["wrong_statement"]),
            ("label", 0),
            ("label_text", "inconsistent"),
            ("error_category", "MR"),
            ("gold_output", p["correct_statement"]),
            ("meta", {"company": p["company"], "doc_name": p["doc_name"], "edit": p["edit"]}),
        ]))
        rows.append(OrderedDict([
            ("sample_id", p["sample_id"] + "_right"),
            ("source", "FinanceBench"),
            ("task", "consistency_detection"),
            ("language", "en"),
            ("context", p["context"]),
            ("instruction", p["question"]),
            ("claim", p["correct_statement"]),
            ("label", 1),
            ("label_text", "consistent"),
            ("error_category", "none"),
            ("gold_output", p["correct_statement"]),
            ("meta", {"company": p["company"], "doc_name": p["doc_name"]}),
        ]))
    jdump(os.path.join(OUT, "unified_consistency_samples.jsonl"), rows)
    return rows


def main():
    fv_det, _ = build_finverbench()
    fb_fact, fb_pairs = build_financebench()
    finben_summary = build_finben()
    build_unified(fv_det, fb_fact, fb_pairs)

    print("\n==== 汇总 ====")
    print("FinVerBench 一致性判定样本 : {}".format(len(fv_det)))
    print("FinanceBench 事实一致性样本: {}".format(len(fb_fact)))
    print("FinanceBench 数值纠错对    : {}".format(len(fb_pairs)))
    print("FinBen 子任务              : {}".format(len(finben_summary)))
    for ds, cn, n in finben_summary:
        print("   - {:<22s} {:>6d}  ({})".format(ds, n, cn))


if __name__ == "__main__":
    main()
