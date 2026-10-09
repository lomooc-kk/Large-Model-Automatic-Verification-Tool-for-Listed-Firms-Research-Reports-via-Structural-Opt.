# -*- coding: utf-8 -*-
"""金融领域语料库「学习版」成品校验（断言式，退出码 0 = 全部通过）"""
import json, gzip, os, re, sys, hashlib, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FILES = {"train": os.path.join(DATA, "finance_zh.train.jsonl.gz"),
         "val": os.path.join(DATA, "finance_zh.val.jsonl.gz")}
MIN_CHARS = 100
TAG_RE = re.compile(
    r"</?(?:p|div|br|hr|span|a|img|b|i|u|em|strong|font|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|dl|dt|dd|"
    r"blockquote|h[1-6]|script|style|center|section|article|figure|figcaption|iframe|video|audio|source|"
    r"meta|link|sup|sub|small|pre|code|nobr|o:p|embed|object|param|form|input|button|select|option|textarea)\b[^>]*>",
    re.I,
)
REQ = ["id", "title", "text", "source", "date", "year", "n_chars", "split"]
# 仅收录 CP437 乱码的特征字符（α/β/μ/Φ 等正规希腊字母不在此列，避免误报）
MOJIBAKE = set("Θ₧ƒ║░▒▓╣╗╝╔╚╦╩╠╬═╧╨╤╥╙╘╒╓╫╪")

fails, warns = [], []
def fail(m): fails.append(m); print("  [FAIL]", m)
def warn(m): warns.append(m); print("  [WARN]", m)


def main():
    ok = True
    ids = set(); fps = set(); per_split_ids = {}
    by_split = collections.Counter()
    n_total = dup = tag_left = repl = ideo = bad_len = missing_field = bad_split = bad_date = empty_title = moji = 0
    for split, path in FILES.items():
        if not os.path.exists(path):
            fail("缺少文件：%s" % path); ok = False; continue
        print("[*] 校验 %s" % os.path.basename(path))
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception as e:
                    fail("%s 第 %d 行 JSON 非法：%s" % (split, ln, e)); break
                n_total += 1
                miss = [k for k in REQ if k not in r]
                if miss:
                    missing_field += 1
                    fail("%s 第 %d 行缺字段 %s" % (split, ln, miss))
                if r.get("split") != split:
                    bad_split += 1
                _id = r.get("id", "")
                if _id in ids:
                    warn("%s 第 %d 行 ID 重复：%s" % (split, ln, _id))
                ids.add(_id)
                per_split_ids.setdefault(split, set()).add(_id)
                t = r.get("text", "")
                if len(t) < MIN_CHARS or r.get("n_chars") != len(t):
                    bad_len += 1
                if TAG_RE.search(t):
                    tag_left += 1
                if "\ufffd" in t:
                    repl += 1
                if "\u3000" in t:
                    ideo += 1
                if not str(r.get("title") or "").strip():
                    empty_title += 1
                d = r.get("date")
                if d is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(d)):
                    bad_date += 1
                if any(ch in MOJIBAKE for ch in t):
                    moji += 1
                fp = hashlib.sha1((r.get("title", "") + "\x01" + t[:2000]).encode("utf-8", "ignore")).hexdigest()
                if fp in fps:
                    dup += 1
                fps.add(fp)
                by_split[split] += 1
        print("    条数=%d" % by_split[split])

    print("\n=== 断言结果 ===")
    def chk(c, m):
        nonlocal ok
        print(("  [PASS] " if c else "  [FAIL] ") + m)
        if not c: fails.append(m); ok = False

    chk(len(ids) == n_total, "ID 全局唯一：%d 唯一 / %d 条" % (len(ids), n_total))
    inter = per_split_ids.get("train", set()) & per_split_ids.get("val", set())
    chk(len(inter) == 0, "train/val 无重叠：交集 %d" % len(inter))
    chk(dup == 0, "精确去重（标题+正文指纹唯一）：重复 %d" % dup)
    chk(tag_left == 0, "无 HTML 标签残留：%d" % tag_left)
    chk(repl == 0, "无替换符 U+FFFD：%d" % repl)
    chk(ideo == 0, "无全角空白 U+3000 残留：%d" % ideo)
    chk(moji == 0, "无典型乱码字符：%d" % moji)
    chk(bad_len == 0, "长度合规(>=%d 且 n_chars 一致)：违例 %d" % (MIN_CHARS, bad_len))
    chk(missing_field == 0, "必需字段齐全：缺失 %d" % missing_field)
    chk(bad_split == 0, "split 与文件名一致：%d" % bad_split)
    chk(bad_date == 0, "日期格式合规(YYYY-MM-DD)：违例 %d" % bad_date)
    chk(empty_title == 0, "标题非空：空标题 %d" % empty_title)

    st = os.path.join(ROOT, "stats.json")
    if os.path.exists(st):
        s = json.load(open(st, encoding="utf-8"))
        tt = s.get("totals", {})
        chk(tt.get("documents_kept") == n_total, "与 stats.json 计数一致：stats=%s 实际=%d" % (tt.get("documents_kept"), n_total))
        chk(tt.get("train") == by_split["train"] and tt.get("val") == by_split["val"],
            "stats.json train/val 一致：%s/%s vs %d/%d" % (tt.get("train"), tt.get("val"), by_split["train"], by_split["val"]))
    else:
        warn("未找到 stats.json")

    # 文件哈希与 MANIFEST 比对
    mf = os.path.join(ROOT, "MANIFEST.json")
    if os.path.exists(mf):
        m = json.load(open(mf, encoding="utf-8"))
        for rel, info in m.get("files", {}).items():
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p) or "sha256" not in info:
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            chk(h.hexdigest() == info["sha256"], "MANIFEST 哈希一致：%s" % rel)
    else:
        warn("未找到 MANIFEST.json")

    print("\n=== 汇总 ===")
    print("总条数=%d  train=%d  val=%d" % (n_total, by_split["train"], by_split["val"]))
    print("FAIL=%d  WARN=%d" % (len(fails), len(warns)))
    print("VERIFY " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
