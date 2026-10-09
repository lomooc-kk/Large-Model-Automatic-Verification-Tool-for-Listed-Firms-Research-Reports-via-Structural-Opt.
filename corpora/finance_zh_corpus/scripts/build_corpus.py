#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金融领域语料库 → 学习版语料（私有容错构建脚本，唯一命名以避免被并发任务干扰）"""
import argparse, json, os, re, gzip, html, hashlib, io, sys, time, zipfile, collections, datetime

DEFAULT_ZIP = "/home/user/5361227762534420236/fmcorpus/corpus.zip"
ZIP_SHA256 = "9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43"
UPSTREAM_REVISION = "558891a9acb2b56fa3d85f1820724dc515ad2fd9"
MIN_CHARS = 100
VAL_PERMILLE = 50
SAMPLE_N = 1000
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
# 仅移除“真正的 HTML 标签”，保留正文中以尖括号书写的缩写（如 <PPP> <EPC>）
HTML_TAG_RE = re.compile(
    r"</?(?:p|div|br|hr|span|a|img|b|i|u|em|strong|font|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|dl|dt|dd|"
    r"blockquote|h[1-6]|script|style|center|section|article|figure|figcaption|iframe|video|audio|source|"
    r"meta|link|sup|sub|small|pre|code|nobr|o:p|embed|object|param|form|input|button|select|option|textarea)\b[^>]*>",
    re.I,
)
WS_RE = re.compile(r"[ \t\u3000\xa0\u200b\ufeff]+")
REPL = "\ufffd"


def fix_zip_name(name):
    try:
        return name.encode("cp437").decode("utf-8")
    except Exception:
        return name


def norm_date(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        if v <= 0:
            return None
        ts = v / 1000.0
        if ts > 4e9:
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


def clean_text(raw):
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    t = html.unescape(raw)          # 先解码实体，避免 <br> 变成真标签后残留
    t = SCRIPT_RE.sub(" ", t)
    t = HTML_TAG_RE.sub("\n", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ln in t.split("\n"):
        ln = WS_RE.sub(" ", ln).strip()
        if ln:
            out.append(ln)
    return "\n".join(out)


def stable_split(doc_id):
    h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < VAL_PERMILLE else "train"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    root = os.path.abspath(args.out_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    zip_path = os.path.abspath(args.zip)
    t0 = time.time()
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "samples"), exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    members = [i for i in z.infolist()
               if fix_zip_name(i.filename).endswith(".json") and "__MACOSX" not in fix_zip_name(i.filename)]
    members.sort(key=lambda i: fix_zip_name(i.filename))
    print("[*] 待处理文件 %d 个" % len(members), flush=True)

    seen = set()
    src_counter = collections.Counter(); year_counter = collections.Counter(); lens = []
    total_raw = total_kept = dup = malformed = short = empty = badchar = missing_date = empty_title_dropped = 0
    per_file = {}; sample_pool = []

    ftrain = gzip.open(os.path.join(root, "data/finance_zh.train.jsonl.gz"), "wt", encoding="utf-8", compresslevel=6)
    fval = gzip.open(os.path.join(root, "data/finance_zh.val.jsonl.gz"), "wt", encoding="utf-8", compresslevel=6)
    fsample = io.open(os.path.join(root, "samples/sample_1000.jsonl"), "w", encoding="utf-8")

    for info in members:
        fn = fix_zip_name(info.filename).split("/")[-1]
        n_raw = n_kept = n_dup = n_bad = n_short = 0
        with z.open(info) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                n_raw += 1; total_raw += 1
                try:
                    o = json.loads(raw)
                except Exception:
                    n_bad += 1; malformed += 1
                    continue
                if not isinstance(o, dict):
                    n_bad += 1; malformed += 1
                    continue
                text = o.get("contentText")
                if text is None:
                    text = o.get("content")
                title = str(o.get("title") or "").strip()
                src = str(o.get("publishSource") or "").strip()
                date = norm_date(o.get("dataTime"))
                if not title:
                    empty_title_dropped += 1
                    continue
                text = clean_text(text)
                n_chars = len(text); lens.append(n_chars)
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
                (fval if rec["split"] == "val" else ftrain).write(line + "\n")
                if src: src_counter[src] += 1
                if date: year_counter[date[:4]] += 1
                else: missing_date += 1
                total_kept += 1; n_kept += 1
                if len(sample_pool) < SAMPLE_N:
                    sample_pool.append(line)
                else:
                    j = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % total_kept
                    if j < SAMPLE_N:
                        sample_pool[j] = line
                if total_kept % 100000 == 0:
                    print("  ... kept=%d raw=%d (%.1fs)" % (total_kept, total_raw, time.time() - t0), flush=True)
        per_file[fn] = {"rows_raw": n_raw, "rows_kept": n_kept, "dup": n_dup, "malformed": n_bad, "too_short": n_short}
        print("文件 %s 完成: raw=%d kept=%d dup=%d bad=%d short=%d" % (fn, n_raw, n_kept, n_dup, n_bad, n_short), flush=True)

    ftrain.close(); fval.close()
    for ln in sample_pool:
        fsample.write(ln + "\n")
    fsample.close()

    train_lines = sum(1 for _ in gzip.open(os.path.join(root, "data/finance_zh.train.jsonl.gz"), "rt", encoding="utf-8"))
    val_lines = sum(1 for _ in gzip.open(os.path.join(root, "data/finance_zh.val.jsonl.gz"), "rt", encoding="utf-8"))
    lens.sort()
    def pct(p): return lens[min(len(lens) - 1, int(len(lens) * p))] if lens else 0
    total_chars = sum(lens)

    stats = {
        "dataset": "金融领域语料库 (ModelScope peopletech/Financial, 人民网科技公司)",
        "license": "Apache License 2.0",
        "upstream_revision": UPSTREAM_REVISION,
        "zip_sha256": ZIP_SHA256,
        "build_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params": {"min_chars": MIN_CHARS, "val_permille": VAL_PERMILLE, "sample_n": SAMPLE_N},
        "per_file": per_file,
        "totals": {"documents_kept": total_kept, "documents_raw": total_raw, "train": train_lines, "val": val_lines,
                   "chars": total_chars, "est_tokens": int(total_chars * 0.85),
                   "distinct_publish_sources": len(src_counter)},
        "quality": {"malformed_or_invalid_utf8": malformed, "empty_text": empty, "too_short_dropped": short,
                    "empty_title_dropped": empty_title_dropped,
                    "duplicate_dropped": dup, "replacement_char_dropped": badchar, "missing_or_invalid_date": missing_date,
                    "dup_ratio": round(dup / total_raw, 6) if total_raw else 0,
                    "keep_ratio": round(total_kept / total_raw, 6) if total_raw else 0},
        "text_len": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99), "max": lens[-1] if lens else 0,
                     "mean": int(total_chars / total_kept) if total_kept else 0},
        "date_year_hist": dict(sorted(year_counter.items())),
        "top20_sources": src_counter.most_common(20),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with io.open(os.path.join(root, "stats.json"), "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=2)

    manifest = {
        "files": {
            "data/finance_zh.train.jsonl.gz": {"lines": train_lines, "bytes": os.path.getsize(os.path.join(root, "data/finance_zh.train.jsonl.gz")), "sha256": sha256_file(os.path.join(root, "data/finance_zh.train.jsonl.gz"))},
            "data/finance_zh.val.jsonl.gz": {"lines": val_lines, "bytes": os.path.getsize(os.path.join(root, "data/finance_zh.val.jsonl.gz")), "sha256": sha256_file(os.path.join(root, "data/finance_zh.val.jsonl.gz"))},
            "samples/sample_1000.jsonl": {"lines": len(sample_pool)},
        },
        "totals": stats["totals"], "quality": stats["quality"],
    }
    with io.open(os.path.join(root, "MANIFEST.json"), "w", encoding="utf-8") as fo:
        json.dump(manifest, fo, ensure_ascii=False, indent=2)

    print(json.dumps(stats["totals"], ensure_ascii=False))
    print(json.dumps(stats["quality"], ensure_ascii=False))
    print("BUILD OK %.1fs" % stats["elapsed_sec"], flush=True)


if __name__ == "__main__":
    main()
