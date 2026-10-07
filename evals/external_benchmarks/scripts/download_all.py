#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键重新下载三个基准的原始数据到 sources/ 目录。

说明（重要）：
  * 本环境访问 github.com / huggingface.co 主站受限，脚本默认走可用的镜像与 CDN：
      - GitHub 文件 : cdn.jsdelivr.net  （失败回退 raw.githubusercontent.com）
      - GitHub 目录 : api.github.com 取文件清单
      - HuggingFace : hf-mirror.com
  * FinBen 部分子任务（flare-finqa / flare-fpb / flare-fiqasa / flare-ectsum /
    flare-multifin-en）为受限(gated)数据集，需在 HuggingFace 上申请授权后才能下载，
    脚本会自动跳过并提示。

运行: python3 scripts/download_all.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "sources")

GH_API_TREE = "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
JSDELIVR = "https://cdn.jsdelivr.net/gh/{repo}@{branch}/{path}"
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{path}"
HF = "https://hf-mirror.com/datasets/{repo}/resolve/main/{path}"
HF_TREE = "https://hf-mirror.com/api/datasets/{repo}/tree/main?recursive=true"


def curl(url, dst, timeout=300, tries=4):
    """下载 url 到 dst，返回是否成功（做基本大小校验）。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(tries):
        r = subprocess.run(["curl", "-sL", "-m", str(timeout), "-o", dst, "-w", "%{http_code}", url],
                           capture_output=True, text=True)
        if r.stdout.strip() == "200" and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        time.sleep(2)
    return False


def curl_json(url, timeout=40, tries=4):
    for _ in range(tries):
        r = subprocess.run(["curl", "-sL", "-m", str(timeout), url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------
FINANCEBENCH_FILES = [
    "README.md",
    "evaluation_playground.ipynb",
    "data/financebench_open_source.jsonl",
    "data/financebench_document_information.jsonl",
]


def download_financebench():
    print("\n[FinanceBench] github.com/patronus-ai/financebench")
    for p in FINANCEBENCH_FILES:
        dst = os.path.join(SRC, "FinanceBench", p)
        ok = curl(JSDELIVR.format(repo="patronus-ai/financebench", branch="main",
                                  path=urllib.parse.quote(p)), dst)
        if not ok:
            ok = curl(RAW.format(repo="patronus-ai/financebench", branch="main",
                                 path=urllib.parse.quote(p)), dst)
        print("   {} {}".format("OK  " if ok else "FAIL", p))


# ---------------------------------------------------------------------------
# FinVerBench
# ---------------------------------------------------------------------------
FVB_KEEP_PREFIX = ("data/benchmark", "data/converted", "src/", "paper/")
FVB_KEEP_FILE = ("requirements.txt", "run_pipeline.sh", "test_api.py", ".gitignore")


def download_finverbench():
    repo, branch = "SiluPanda/finverification-bench", "main"
    print("\n[FinVerBench] github.com/{}".format(repo))
    tree = curl_json(GH_API_TREE.format(repo=repo, branch=branch))
    if not tree or "tree" not in tree:
        print("   无法获取文件清单，跳过")
        return
    files = []
    for t in tree["tree"]:
        if t["type"] != "blob":
            continue
        p = t["path"]
        if p.endswith(".pdf") or "figures" in p:
            continue
        if p.startswith(FVB_KEEP_PREFIX) or p in FVB_KEEP_FILE:
            files.append(p)
    for p in files:
        dst = os.path.join(SRC, "FinVerBench", p)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        ok = curl(JSDELIVR.format(repo=repo, branch=branch, path=urllib.parse.quote(p)), dst)
        if not ok:
            curl(RAW.format(repo=repo, branch=branch, path=urllib.parse.quote(p)), dst)
    print("   已处理 {} 个文件".format(len(files)))


# ---------------------------------------------------------------------------
# FinBen
# ---------------------------------------------------------------------------
FINBEN_OPEN = [
    "finben-finer-ord", "finben-fomc", "flare-fomc",
    "flare-finred", "flare-fnxl", "flare-convfinqa", "flare-tatqa",
]
FINBEN_GATED = ["flare-finqa", "flare-fpb", "flare-fiqasa", "flare-ectsum", "flare-multifin-en"]


def download_finben():
    print("\n[FinBen] huggingface.co/TheFinAI")
    for ds in FINBEN_OPEN:
        tree = curl_json(HF_TREE.format(repo="TheFinAI/" + ds))
        if not tree:
            print("   {} 清单获取失败".format(ds))
            continue
        for s in tree:
            if s.get("type") != "file" or not s["path"].endswith(".parquet"):
                continue
            dst = os.path.join(SRC, "FinBen", ds, s["path"])
            if os.path.exists(dst) and (not s.get("size") or os.path.getsize(dst) == s["size"]):
                continue
            ok = curl(HF.format(repo="TheFinAI/" + ds, path=urllib.parse.quote(s["path"])), dst)
            print("   {} {} {}".format("OK  " if ok else "FAIL", ds, s["path"]))
    print("   受限(需授权)子任务，已跳过：{}".format(", ".join(FINBEN_GATED)))


if __name__ == "__main__":
    os.makedirs(SRC, exist_ok=True)
    download_financebench()
    download_finverbench()
    download_finben()
    print("\n完成。随后运行: python3 scripts/build_datasets.py")
