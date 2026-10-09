# 金融领域语料库（中文金融语料 · 继续预训练用）

人民网科技公司发布的**中文金融语料库**（Apache License 2.0），面向**中文金融大模型的继续预训练（CPT）**。
本目录提供**可复现的下载 → 清洗 → 划分**流水线、**统计画像**与**小样本**；
**数据本体不入 git 仓库**（遵循本仓 `.gitignore` 的"大文件只放本地"约定），改由 **GitHub Release** 提供下载。

---

## 1. 一句话结论

| 项 | 值 |
|---|---|
| 文档数（清洗后） | **700,761** 篇（原始 733,547 篇，保留率 95.5%） |
| 字符数 | 约 **13.9 亿** 字符 |
| 估计 token | 约 **11.8 亿**（中文按 0.85 token/字保守估算） |
| 来源媒体 | **1,319** 家 |
| 时间跨度 | 1973 – 2024（**99% 集中在 2008–2024**，其中 2024 年 41.3 万篇） |
| 语言 / 领域 | 中文 / 金融（政策、银行、保险、股票、证券、期货、基金、理财、投资） |
| 许可 | **Apache License 2.0**（可商用，需保留声明） |

> 规模量级正好落在"**领域继续预训练**"的可用区间（数亿~十几亿 token）。但请注意：**它是新闻语料，不是研报或财报原文**（见 §6 已知限制）。

---

## 2. 数据从哪来（已冻结并校验）

| 项 | 值 |
|---|---|
| 上游 | ModelScope `peopletech/Financial`（人民网科技公司） |
| 锁定 revision | `558891a9acb2b56fa3d85f1820724dc515ad2fd9` |
| 官方文件 | `金融领域语料库.zip`，**1,243,338,095** 字节 |
| **SHA-256** | `9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43` |
| 校验结果 | 本地 `sha256sum` **与上游元信息声明值完全一致** |

> ⚠️ 上游包内是两个 JSONL 文件，**字段名与日期格式不一致**（`contentText`+字符串日期 vs `content`+毫秒时间戳），
> 且带 macOS 的 `__MACOSX` 冗余条目。构建脚本已做归一，细节见 [`MANIFEST.json`](MANIFEST.json) 与 [`SOURCE.md`](SOURCE.md)。

**镜像下载（推荐队友用这个，最快）**

- Release 页面：`releases/tag/corpus-finance-zh-v1`
- 资产：`finance_zh_corpus_peopletech.zip`（1,243,338,095 字节）

---

## 3. 目录内容

```
corpora/finance_zh_peopletech/
├── README.md                 # 本文件
├── SOURCE.md                 # 来源 / 版本 / 许可 / 复现步骤
├── MANIFEST.json             # 机器可读的冻结信息（revision、哈希、字段规范）
├── stats.json                # 统计画像（本目录由真实全量扫描产出）
├── samples/
│   └── sample_1000.jsonl     # 1000 条随机样本（~5.4 MB，已入库，便于先看数据）
└── scripts/
    ├── download_corpus.py    # 下载官方 zip + SHA-256 校验 + 断点续传
    └── build_corpus.py       # 清洗 + 去重 + 稳定划分 + 画像 + 自检
```

---

## 4. 快速上手

```bash
cd corpora/finance_zh_peopletech

# 1) 取回官方数据包并校验（约 1.16 GB）
python3 scripts/download_corpus.py --out-dir ./_cache

# 2) 构建：清洗 + 去重 + 划分 train/val + 写 stats.json + 小样本
python3 scripts/build_corpus.py --zip ./_cache/金融领域语料库.zip --out-dir .

# 只想先看画像与样本（不产生 1 GB 大文件，约 3 分钟）
python3 scripts/build_corpus.py --zip ./_cache/金融领域语料库.zip --out-dir . --no-full
```

产物：

```
data/finance_zh.train.jsonl.gz   # ~95% ，约 1.1 GB
data/finance_zh.val.jsonl.gz     # ~5%
samples/sample_1000.jsonl
stats.json
```

**构建脚本会自动做终检**：回读全部产物并严格按 UTF-8 校验，任何一行非法即非零退出（`[✓] 自检通过`）。

---

## 5. 字段规范

每行一条 JSON（train/val/样本三者同构）：

```json
{
  "id": "fmzh-0000001",
  "title": "标题",
  "text": "清洗后的正文（HTML 去标签 + 实体解码 + 空白归一，保留段落）",
  "source": "证券时报",
  "date": "2024-03-25",
  "year": 2024,
  "n_chars": 703,
  "split": "train"
}
```

| 字段 | 说明 |
|---|---|
| `id` | 中性顺序 ID，**不含来源/正误标记**，避免用于提示词时泄漏信息 |
| `text` | 仅正文，已去 HTML 标签、解码 HTML 实体、归一全角/零宽空白 |
| `source` | 发布媒体（1,319 家） |
| `date` | `YYYY-MM-DD`；无法解析时为 `null`（全量仅 1 条） |
| `split` | `train` / `val`，按 `id` 的 SHA-1 哈希稳定划分（val 占 5%，**无随机性、可复现**） |

**CPT 接入建议**：直接拼 `title + "\n" + text` 作为一条样本；
建议按 `source` 或 `id` 对同源内容另做分组去重（新闻转载常见），并在通用语料中**混入 30%~50%** 以防灾难性遗忘。

---

## 6. 已知限制（请在使用前确认）

1. **体裁是新闻，不是研报/财报**。本语料适合补金融领域词汇与表达，**不能替代**研报纠错任务所需的"原始材料—表述"证据对（那类请用 `evals/` 下的基准）。
2. **时间分布偏斜**：41.3 万篇（约 59%）为 2024 年，2000 年以前仅 43 篇。若关心历史语境，需自行重采样。
3. **来源集中**：前三家（证券时报、证券日报、中国证券报）合计约 52%，存在**媒体风格偏置**。
4. **重复率约 2.8%**：已按 `title+正文前 2000 字` 精确去重并丢弃 20,238 篇；但**近似重复（改写转载）未处理**。
5. **正文含少量版式残留**：清洗是规则式（HTML 标签 + 实体），个别文章可能残留"本文转自…""记者 xxx"等导语，未做内容级剔除。
6. **许可**：Apache 2.0，可商用，但**须保留版权与许可声明**，并标注本项目对数据做了清洗与格式转换。

---

## 7. 相关

- 外部评测基准（纠错/一致性）：`evals/external_benchmarks/`（FinVerBench / FinanceBench / FinBen）
- 中文"语言+事实"联合纠错集：`evals/external_benchmarks/clfec_finance/`
- 训练资源选型建议：见主仓文档与 `tools/`
