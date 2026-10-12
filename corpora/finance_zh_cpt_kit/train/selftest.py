# -*- coding: utf-8 -*-
"""开箱即训包 · 自检脚本（纯标准库，无需安装任何第三方依赖）。

作用：在正式训练之前，快速确认「这个包是完整、可用、没有坏数据的」。
它做四类检查：

  A. 结构检查：必需文件是否齐全、大小是否合理；
  B. 数据抽样检查：字段是否齐全、n_chars 是否与正文一致、split 是否与文件名一致；
  C. 工具检查：脚本能否通过语法编译、配置 JSON 是否合法；
  D. 冒烟样本：生成一个极小数据集，用来先跑通训练流程（可选）。

用法：
    python train/selftest.py                  # 抽样自检（秒级）
    python train/selftest.py --sample 2000    # 每个文件抽查 2000 条
    python train/selftest.py --make_smoke     # 额外生成 examples/ 下的冒烟数据
    python train/selftest.py --full           # 全量统计行数（较慢，约 1~2 分钟）

退出码 0 = 全部通过；非 0 = 有检查未通过。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_FILES = [
    "README.md",
    "QUICKSTART.md",
    "TRAINING_GUIDE.md",
    "VERIFICATION.md",
    "LICENSE_NOTES.md",
    "stats.json",
    "MANIFEST.json",
    "data/finance_zh.train.jsonl.gz",
    "data/finance_zh.val.jsonl.gz",
    "samples/sample_1000.jsonl",
    "scripts/download_corpus.py",
    "scripts/build_corpus.py",
    "scripts/verify_corpus.py",
    "train/train_cpt.py",
    "train/tokenize_and_pack.py",
    "train/mix_general_corpus.py",
    "train/selftest.py",
    "train/requirements-train.txt",
    "configs/qwen2.5-1.5b-lora.sh",
    "configs/qwen2.5-7b-lora.sh",
    "configs/qwen2.5-7b-full.sh",
    "examples/llamafactory/dataset_info.json",
    "examples/llamafactory/llamafactory_cpt.yaml",
    "run_cpt.sh",
]

PY_FILES = [
    "scripts/download_corpus.py",
    "scripts/build_corpus.py",
    "scripts/verify_corpus.py",
    "train/train_cpt.py",
    "train/tokenize_and_pack.py",
    "train/mix_general_corpus.py",
    "train/selftest.py",
]

JSON_FILES = [
    "stats.json",
    "MANIFEST.json",
    "examples/llamafactory/dataset_info.json",
]

REQ_FIELDS = ["id", "title", "text", "source", "date", "year", "n_chars", "split"]
MIN_CHARS = 100

fails: List[str] = []
warns: List[str] = []


def fail(msg: str) -> None:
    fails.append(msg)
    print("  [FAIL] %s" % msg)


def warn(msg: str) -> None:
    warns.append(msg)
    print("  [WARN] %s" % msg)


def ok(msg: str) -> None:
    print("  [ ok ] %s" % msg)


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


# ---------------------------------------------------------------------------
# A. 结构检查
# ---------------------------------------------------------------------------
def check_structure() -> None:
    print("[A] 结构检查")
    for rel in REQUIRED_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            fail("缺少文件：%s" % rel)
        else:
            size = os.path.getsize(p)
            if size == 0:
                fail("文件为空：%s" % rel)
            else:
                ok("%-52s %s" % (rel, human(size)))
    # 数据文件尺寸下限（防止下载/解压不完整）
    for rel, floor in [("data/finance_zh.train.jsonl.gz", 500 * 1024 * 1024),
                       ("data/finance_zh.val.jsonl.gz", 20 * 1024 * 1024)]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and os.path.getsize(p) < floor:
            fail("%s 体积异常偏小（%s），疑似未下载完整" % (rel, human(os.path.getsize(p))))


# ---------------------------------------------------------------------------
# B. 数据抽样检查
# ---------------------------------------------------------------------------
def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def check_split(rel: str, split: str, sample: int, full: bool) -> Optional[int]:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        fail("找不到 %s" % rel)
        return None
    n = 0
    checked = 0
    bad = 0
    ids_seen = set()
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if not full and checked >= sample:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                fail("%s 第 %d 行 JSON 非法：%s" % (rel, n, e))
                bad += 1
                break
            miss = [k for k in REQ_FIELDS if k not in r]
            if miss:
                fail("%s 第 %d 行缺字段 %s" % (rel, n, miss))
                bad += 1
            if r.get("split") != split:
                fail("%s 第 %d 行 split=%r，应为 %r" % (rel, n, r.get("split"), split))
                bad += 1
            if r.get("n_chars") != len(r.get("text", "")):
                fail("%s 第 %d 行 n_chars 与正文长度不一致" % (rel, n))
                bad += 1
            if len(r.get("text", "")) < MIN_CHARS:
                fail("%s 第 %d 行正文过短（%d 字）" % (rel, n, len(r.get("text", ""))))
                bad += 1
            _id = r.get("id", "")
            if _id in ids_seen:
                warn("%s 第 %d 行 ID 在抽样范围内重复：%s" % (rel, n, _id))
            ids_seen.add(_id)
            checked += 1
            if bad >= 20:
                fail("错误过多，提前停止 %s 的检查" % rel)
                break
    if bad == 0:
        mode = "全量" if full else "抽样 %d" % checked
        ok("%s：%s 行，校验 %s 行，无异常" % (rel, "{:,}".format(n), mode))
    return n


def check_data(sample: int, full: bool) -> Dict[str, int]:
    print("[B] 数据检查")
    n_train = check_split("data/finance_zh.train.jsonl.gz", "train", sample, full)
    n_val = check_split("data/finance_zh.val.jsonl.gz", "val", sample, full)

    # 与 stats / MANIFEST 对齐
    out = {}
    try:
        with open(os.path.join(ROOT, "stats.json"), "r", encoding="utf-8") as f:
            stats = json.load(f)
        tot = stats["totals"]
        out["train"] = tot["train"]
        out["val"] = tot["val"]
        if n_train is not None and full and n_train != tot["train"]:
            fail("train 实际行数 %d != stats 记录的 %d" % (n_train, tot["train"]))
        if n_val is not None and full and n_val != tot["val"]:
            fail("val 实际行数 %d != stats 记录的 %d" % (n_val, tot["val"]))
        if tot["train"] + tot["val"] != tot["documents_kept"]:
            fail("stats 内部不一致：train + val != documents_kept")
        if not full:
            ok("stats.json：train=%s val=%s 合计=%s（%s 亿字）"
               % ("{:,}".format(tot["train"]), "{:,}".format(tot["val"]),
                  "{:,}".format(tot["documents_kept"]), "%.2f" % (tot["chars"] / 1e8)))
    except Exception as e:
        fail("stats.json 读取失败：%s" % e)

    try:
        with open(os.path.join(ROOT, "MANIFEST.json"), "r", encoding="utf-8") as f:
            mani = json.load(f)
        for rel, meta in mani.get("files", {}).items():
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                fail("MANIFEST 列出的文件不存在：%s" % rel)
            elif "bytes" in meta and os.path.getsize(p) != meta["bytes"]:
                fail("%s 大小与 MANIFEST 不一致（%d != %d）"
                     % (rel, os.path.getsize(p), meta["bytes"]))
        ok("MANIFEST.json：文件大小与记录一致")
    except Exception as e:
        fail("MANIFEST.json 读取失败：%s" % e)
    return out


# ---------------------------------------------------------------------------
# C. 工具检查
# ---------------------------------------------------------------------------
def check_tools() -> None:
    print("[C] 工具检查")
    for rel in PY_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                src = f.read()
            compile(src, p, "exec")   # 只做语法检查，不写字节码文件
        except SyntaxError as e:
            fail("%s 语法错误：第 %s 行 %s" % (rel, e.lineno, e.msg))
    ok("Python 脚本语法编译：全部通过")

    for rel in JSON_FILES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            warn("（跳过，文件不存在）%s" % rel)
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception as e:
            fail("%s JSON 非法：%s" % (rel, e))
    ok("JSON 配置文件：全部合法")

    # 训练脚本关键能力探测（不导入 torch，只做静态检查）
    tp = os.path.join(ROOT, "train/train_cpt.py")
    if os.path.exists(tp):
        src = open(tp, "r", encoding="utf-8").read()
        for needle, desc in [
            ("IterableDataset", "流式数据集"),
            ("DataCollatorForLanguageModeling", "语言建模 collator"),
            ("--dry_run", "数据管道自检开关"),
            ("eos_token_id", "文档边界 EOS"),
        ]:
            if needle in src:
                ok("train_cpt.py 含 %s" % desc)
            else:
                fail("train_cpt.py 缺失 %s（%s）" % (desc, needle))


# ---------------------------------------------------------------------------
# D. 冒烟样本
# ---------------------------------------------------------------------------
def make_smoke(n: int = 200) -> None:
    print("[D] 生成冒烟样本")
    src = os.path.join(ROOT, "data/finance_zh.train.jsonl.gz")
    dst_dir = os.path.join(ROOT, "examples")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "smoke_%d.jsonl" % n)
    wrote = 0
    with open_text(src) as f, open(dst, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w.write(line + "\n")
            wrote += 1
            if wrote >= n:
                break
    ok("已写出 %s（%d 条，可直接用于 10 步内跑通训练流程）" % (os.path.relpath(dst, ROOT), wrote))


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="开箱即训包自检")
    p.add_argument("--sample", type=int, default=500, help="每个数据文件抽样条数")
    p.add_argument("--full", action="store_true", help="全量统计行数（较慢）")
    p.add_argument("--make_smoke", action="store_true", help="生成 examples/ 冒烟数据")
    p.add_argument("--smoke_n", type=int, default=200, help="冒烟数据条数")
    args = p.parse_args(argv)

    print("=" * 68)
    print("金融领域语料库 · 开箱即训包 自检")
    print("包根目录：%s" % ROOT)
    print("=" * 68)

    check_structure()
    check_data(args.sample, args.full)
    check_tools()
    if args.make_smoke:
        make_smoke(args.smoke_n)

    print("=" * 68)
    if fails:
        print("结果：失败 %d 项，警告 %d 项" % (len(fails), len(warns)))
        for m in fails:
            print("  - %s" % m)
        print("=" * 68)
        return 1
    print("结果：全部通过（警告 %d 项）" % len(warns))
    for m in warns:
        print("  - %s" % m)
    print("提示：确认无误后可执行  bash run_cpt.sh  开始训练。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
