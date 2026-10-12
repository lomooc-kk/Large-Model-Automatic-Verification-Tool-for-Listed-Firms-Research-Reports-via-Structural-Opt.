#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载「金融领域语料库」官方数据包（锁定上游 revision），并校验 SHA-256。

上游：ModelScope peopletech/Financial（人民网科技公司），Apache License 2.0。
本脚本只做「取回 + 校验」，不做任何清洗；清洗与切分由 build_corpus.py 完成。

用法：
    # 1) 下载官方 zip（默认放到与脚本同级的上游缓存目录）
    python3 scripts/download_corpus.py

    # 2) 指定输出目录 / 断点续传（默认开启）
    python3 scripts/download_corpus.py --out-dir /data/cache

    # 3) 只校验已有文件
    python3 scripts/download_corpus.py --verify-only /data/cache/金融领域语料库.zip

    # 4) 优先从 GitHub Release 下载（若上游不可达）
    python3 scripts/download_corpus.py --prefer-release

退出码：0 成功（并通过哈希校验）；非 0 失败。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---- 冻结的上游信息（改动需同步更新 MANIFEST.json） ----
MODELSCOPE_DATASET = "peopletech/Financial"
REVISION = "558891a9acb2b56fa3d85f1820724dc515ad2fd9"
FILE_NAME = "金融领域语料库.zip"
ZIP_BYTES = 1243338095
ZIP_SHA256 = "9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43"

MODELSCOPE_URL = (
    "https://www.modelscope.cn/api/v1/datasets/{ds}/repo"
    "?Revision={rev}&FilePath={fn}"
).format(ds=MODELSCOPE_DATASET, rev=REVISION, fn=urllib.parse.quote(FILE_NAME))

# 可选：GitHub Release 备用通道（由本仓库的 Release 提供，见 MANIFEST.json 的 mirror 字段）
RELEASE_URL = os.environ.get("FMZH_RELEASE_URL", "")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "_cache"))


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f%s" % (n, u)
        n /= 1024.0
    return "%.1fGB" % n


def download(url: str, dest: str, expected_size: int = ZIP_BYTES) -> None:
    """带断点续传的流式下载。"""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    pos = os.path.getsize(dest) if os.path.exists(dest) else 0
    if pos == expected_size:
        print("[=] 已存在完整文件，跳过下载：%s" % dest)
        return
    if pos > expected_size:
        print("[!] 本地文件大于预期，从头重下")
        os.remove(dest)
        pos = 0

    req = urllib.request.Request(url, headers={"User-Agent": "fmzh-downloader/1.0"})
    if pos:
        req.add_header("Range", "bytes=%d-" % pos)
        print("[~] 断点续传，从 %s 继续" % _human(pos))
    else:
        print("[↓] 开始下载：%s" % url)

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        total = expected_size
        mode = "ab" if pos else "wb"
        got = pos
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got % (64 << 20) < (1 << 20):
                    speed = (got - pos) / max(1e-6, time.time() - t0)
                    sys.stdout.write(
                        "\r    %s / %s  (%.1f MB/s)   "
                        % (_human(got), _human(total), speed / 1048576)
                    )
                    sys.stdout.flush()
    print("\n[✓] 下载完成：%s（%.1fs）" % (dest, time.time() - t0))


def verify(path: str, sha: str = ZIP_SHA256) -> bool:
    if not os.path.exists(path):
        print("[✗] 文件不存在：%s" % path)
        return False
    size = os.path.getsize(path)
    print("[*] 校验 %s（%s）" % (path, _human(size)))
    if size != ZIP_BYTES:
        print("[✗] 大小不符：期望 %d，实际 %d" % (ZIP_BYTES, size))
        return False
    actual = _sha256(path)
    if actual != sha:
        print("[✗] SHA-256 不符\n    期望 %s\n    实际 %s" % (sha, actual))
        return False
    print("[✓] SHA-256 一致：%s" % actual)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="下载并校验金融领域语料库官方数据包")
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="输出目录（默认 ../_cache）")
    ap.add_argument("--prefer-release", action="store_true", help="优先使用 GitHub Release 备用通道")
    ap.add_argument("--verify-only", metavar="ZIP", help="只校验指定 zip，不下载")
    args = ap.parse_args()

    if args.verify_only:
        return 0 if verify(args.verify_only) else 1

    dest = os.path.join(args.out_dir, FILE_NAME)
    urls = []
    if args.prefer_release and RELEASE_URL:
        urls.append(RELEASE_URL)
    urls.append(MODELSCOPE_URL)

    last = None
    for u in urls:
        try:
            download(u, dest)
            if verify(dest):
                print("\n下一步：python3 scripts/build_corpus.py --zip %r" % dest)
                return 0
        except urllib.error.HTTPError as e:
            last = "HTTP %s: %s" % (e.code, u)
            print("[!] 失败：%s" % last)
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, e)
            print("[!] 失败：%s" % last)
    print("[✗] 所有通道均失败：%s" % last)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
