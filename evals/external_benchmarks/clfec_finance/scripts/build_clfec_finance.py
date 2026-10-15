#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLFEC 金融子集构建脚本（幂等，可重复运行）。

做的事：
1. 读取官方原始文件 raw/CLFEC.official.json（sha256 已冻结）；
2. 只保留 domain == "Finance" 的段落；
3. 对全量数据做质量审计，定位官方数据自身的瑕疵样本并写入 audit/quarantine_candidates.jsonl；
4. 把金融子集按 4 个官方诊断拆分（mix / fec_only / lec_only / no_error）分别导出为
   - data/inputs.<split>.jsonl  仅含 sample_id + input_text（模型输入，绝不含答案）
   - data/gold.<split>.jsonl    含 corrected_text + 编辑操作 cors + 元信息（标准答案）
5. 生成 data/MANIFEST.json 与 data/_checksums.sha256。

绝不静默删除任何样本：被排除的样本一律在 audit/ 中留痕。
"""

import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "CLFEC.official.json")
DATA = os.path.join(ROOT, "data")
AUDIT = os.path.join(ROOT, "audit")

SPLITS = ["mix", "fec_only", "lec_only", "no_error"]
DOMAIN = "Finance"
ID_PREFIX = "clfec-fin"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    """原子写出：先写临时文件再替换，避免半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def apply_cors(text, cors):
    """把编辑操作施加到文本上：左闭右开区间，从后往前替换。"""
    buf = list(text)
    for c in sorted(cors, key=lambda x: x["start"], reverse=True):
        buf[c["start"]:c["end"]] = list(c["candidate_word"])
    return "".join(buf)


def audit_item(item):
    """返回该样本的缺陷标签列表（空表示无缺陷）。"""
    tags = []
    text, gold, cors = item["input_text"], item["corrected_text"], item["cors"]
    for c in cors:
        s, e = c["start"], c["end"]
        if not (0 <= s < e <= len(text)):
            tags.append("span_out_of_range")
        elif text[s:e] != c["error_word"]:
            tags.append("span_mismatch")
        if c["candidate_word"] == c["error_word"]:
            tags.append("no_op_edit")
    iv = sorted((c["start"], c["end"]) for c in cors)
    for a, b in zip(iv, iv[1:]):
        if b[0] < a[1]:
            tags.append("overlapping_edit")
    if apply_cors(text, cors) != gold:
        tags.append("apply_mismatch")
    if item["type"] == "no_error":
        if text != gold or cors:
            tags.append("no_error_violation")
    elif text == gold and cors:
        tags.append("text_unchanged")
    return tags


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    for d in (DATA, AUDIT):
        os.makedirs(d, exist_ok=True)

    raw_bytes = open(RAW, "rb").read()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    items = json.loads(raw_bytes.decode("utf-8"))

    # ---------- 1. 全量审计，定位官方瑕疵 ----------
    quarantine = []
    for it in items:
        tags = audit_item(it)
        if tags:
            quarantine.append(OrderedDict([
                ("source_id", it["id"]),
                ("domain", it["domain"]),
                ("split", it["type"]),
                ("reason", sorted(set(tags))),
            ]))
    with open(os.path.join(AUDIT, "quarantine_candidates.jsonl"), "w", encoding="utf-8") as f:
        for q in quarantine:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    bad_ids = {q["source_id"] for q in quarantine}
    fin_bad = [q for q in quarantine if q["domain"] == DOMAIN]

    # ---------- 2. 筛选金融子集 ----------
    finance = [it for it in items if it["domain"] == DOMAIN]
    finance_clean = [it for it in finance if it["id"] not in bad_ids]

    # 中性、不泄漏拆分/正误含义的全局唯一 id（按官方文件出现顺序分配）
    for i, it in enumerate(finance_clean, start=1):
        it["_sample_id"] = f"{ID_PREFIX}-{i:04d}"

    # ---------- 3. 分流导出 ----------
    per_split = OrderedDict()
    for split in SPLITS:
        sub = [it for it in finance_clean if it["type"] == split]
        inputs, golds = [], []
        for it in sub:
            # 输入文件：只放模型能看到的内容，绝不包含任何答案字段
            inputs.append(OrderedDict([
                ("sample_id", it["_sample_id"]),
                ("input_text", it["input_text"]),
            ]))
            # 答案文件：标准答案与编辑定位
            etc = Counter(c["error_type"] for c in it["cors"])
            golds.append(OrderedDict([
                ("sample_id", it["_sample_id"]),
                ("source_id", it["id"]),
                ("domain", it["domain"]),
                ("split", split),
                ("corrected_text", it["corrected_text"]),
                ("num_edits", len(it["cors"])),
                ("error_type_counts", OrderedDict(sorted(etc.items()))),
                ("cors", [OrderedDict([
                    ("start", c["start"]),
                    ("end", c["end"]),
                    ("error_word", c["error_word"]),
                    ("candidate_word", c["candidate_word"]),
                    ("error_type", c["error_type"]),
                ]) for c in sorted(it["cors"], key=lambda x: x["start"])]),
            ]))
        write_jsonl(os.path.join(DATA, f"inputs.{split}.jsonl"), inputs)
        write_jsonl(os.path.join(DATA, f"gold.{split}.jsonl"), golds)
        per_split[split] = OrderedDict([
            ("paragraphs", len(sub)),
            ("edits", sum(len(it["cors"]) for it in sub)),
            ("error_type_counts", OrderedDict(sorted(
                Counter(c["error_type"] for it in sub for c in it["cors"]).items()))),
        ])

    # 未使用的金融样本（如有被隔离的）也要留痕
    unused = [it for it in finance if it["id"] in bad_ids]

    # ---------- 4. 清单 ----------
    manifest = OrderedDict([
        ("dataset", "CLFEC-Finance"),
        ("task", "中文金融段落：语言错误 + 事实错误 联合纠正"),
        ("language", "zh"),
        ("source", OrderedDict([
            ("name", "CLFEC"),
            ("upstream_repo", "https://github.com/jiu2021/CLFEC-Dataset"),
            ("upstream_commit", "af4edeae56eb56846532cd1af62fe03a6586a38c"),
            ("file", "data/CLFEC.json"),
            ("file_sha256", raw_sha),
            ("license", "MIT (Copyright (c) 2026 The CLFEC Authors)"),
        ])),
        ("filter", OrderedDict([
            ("domain", DOMAIN),
            ("splits", SPLITS),
            ("kept", len(finance_clean)),
            ("excluded", len(unused)),
            ("total_rows_in_full_dataset", len(items)),
            ("defective_rows_in_full_dataset", len(quarantine)),
            ("defective_rows_in_finance", len(fin_bad)),
        ])),
        ("splits", per_split),
        ("totals", OrderedDict([
            ("paragraphs", len(finance_clean)),
            ("edits", sum(len(it["cors"]) for it in finance_clean)),
            ("characters", sum(len(it["input_text"]) for it in finance_clean)),
        ])),
        ("files", OrderedDict([
            ("inputs", [f"inputs.{s}.jsonl" for s in SPLITS]),
            ("gold", [f"gold.{s}.jsonl" for s in SPLITS]),
        ])),
        ("field_policy", "inputs.* 只含 sample_id 与 input_text；corrected_text / cors / split 等答案字段只出现在 gold.* 中。"),
    ])
    with open(os.path.join(DATA, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---------- 5. 校验和 ----------
    sums = []
    for fn in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, fn)
        if os.path.isfile(p) and fn != "_checksums.sha256":
            sums.append(f"{sha256_of(p)}  {fn}")
    with open(os.path.join(DATA, "_checksums.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(sums) + "\n")

    # ---------- 输出摘要 ----------
    print(f"全量 {len(items)} 段, 其中官方瑕疵 {len(quarantine)} 段 (金融 {len(fin_bad)} 段)")
    print(f"金融子集保留 {len(finance_clean)} 段, 隔离 {len(unused)} 段")
    for s in SPLITS:
        d = per_split[s]
        print(f"  {s:9} 段落={d['paragraphs']:3}  编辑={d['edits']:4}")
    print(f"合计 段落={manifest['totals']['paragraphs']} 编辑={manifest['totals']['edits']} 字符={manifest['totals']['characters']}")


if __name__ == "__main__":
    main()
