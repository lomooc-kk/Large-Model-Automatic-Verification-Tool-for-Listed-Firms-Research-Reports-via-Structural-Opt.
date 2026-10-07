# 金融长文本一致性 / 纠错数据集（financial_consistency_datasets）

面向"金融长文本自身矛盾检测与纠错"任务，把目前最接近的三个公开基准
（**FinVerBench / FinanceBench / FinBen**）下载并统一整理成可直接使用的数据集。

原始基准大多是"金融长文本 → 问答"，需要二次加工才能用于**一致性检测 / 纠错**。
本仓库把三个来源统一成一套稳定的字段规范，并额外派生出**检测样本**与**纠错样本**，
队友拉下来即可直接训练 / 评测。

> 数据规模：**原始 28 MB**，**整理后 50 MB**，共 **20,000+ 条**样本。

---

## 一、交付内容一览

### 1.1 整理后的成品数据（`data/`）

| 文件 | 条数 | 任务类型 | 说明 |
|---|---|---|---|
| `data/finverbench_consistency_detection.jsonl` | **1985** | 一致性判定 | 财报是否内部一致，附错误类别/位置/幅度等完整标注 |
| `data/finverbench_statement_correction.jsonl` | **1942** | 纠错 | "错误报表 → 正确报表"成对样本（长文本纠错） |
| `data/financebench_fact_consistency.jsonl` | **150** | 事实一致性 | 问题 + 证据原文 + 标准答案，用于证据回溯式事实核验 |
| `data/financebench_correction_pairs.jsonl` | **111** | 数值断言纠错 | "原句—错误改写—正确改写"三元组（规则扰动生成，可校验） |
| `data/finben/finben-finer-ord.jsonl` | **1075** | 信息抽取 | 金融文本命名实体识别（PER/LOC/ORG） |
| `data/finben/finben-fomc.jsonl` | **496** | 文本分析 | 央行(FOMC)立场分类 hawkish/dovish/neutral |
| `data/finben/flare-fomc.jsonl` | **496** | 文本分析 | 同上（与 finben-fomc 同源） |
| `data/finben/flare-finred.jsonl` | **1068** | 信息抽取 | 金融关系抽取 |
| `data/finben/flare-fnxl.jsonl` | **318** | 信息抽取 | 财报数字 / XBRL 标签抽取（与数值一致性高度相关） |
| `data/finben/flare-tatqa.jsonl` | **1668** | 数值推理 | 表格 + 文本数值问答 |
| `data/finben/flare-convfinqa.jsonl.gz` | **12594** | 数值推理 | 财报多轮数值问答（体积较大，已 gzip） |
| `data/unified_consistency_samples.jsonl` | **2207** | 一致性判定（统一视图） | FinVerBench 检测样本 + FinanceBench 数值断言正/负样本 |

### 1.2 原始下载（`sources/`）

保持上游仓库的原始目录结构，未做改动，便于与官方对照、重新加工或提交引用。

```
sources/
├── FinanceBench/      # patronus-ai/financebench  data/*.jsonl + README + notebook
├── FinVerBench/       # SiluPanda/finverification-bench  benchmark + instances + src + paper
└── FinBen/            # TheFinAI/finben-* / flare-*  7 个可公开下载子任务的 parquet
```

### 1.3 脚本与文档（`scripts/`、根目录）

| 文件 | 说明 |
|---|---|
| `scripts/download_all.py` | 一键重新下载三个基准（自动走镜像，含受限数据集跳过逻辑） |
| `scripts/build_datasets.py` | 由 `sources/` 生成 `data/` 下的标准化数据集（可重复运行） |
| `SOURCES.md` | 每个来源的仓库地址、许可、引用方式、下载通道 |
| `SCHEMA.md` | 各数据集的字段级说明 |
| `提交与PR指南.md` | 如何把本目录并入你们现有 GitHub 仓库并提 PR |

---

## 二、三个基准的定位与用法

### FinVerBench —— 最贴合"一致性纠错"的现成样本

构建自 43 家标普 500 公司的 SEC 10-K XBRL 财报，给出了**四类错误注入体系**，
每个实例都是"干净报表 / 注入错误报表"的对照，天然适合做检测与纠错：

| 类别 | 含义 | 具体错误类型 |
|---|---|---|
| **AE** 算术错误 | 合计/小计不等于各项之和 | `AE_ROW_SUM`、`AE_COLUMN_SUM` |
| **CL** 跨表勾稽错误 | 同一指标在不同报表间对不上 | `CL_NET_INCOME_TO_RE`、`CL_NET_INCOME_TO_CFS`、`CL_ENDING_CASH` |
| **YOY** 同比连续性错误 | 期初/期末与上期余额接不上 | `YOY_OPENING_BALANCE`、`YOY_COMPUTED_CHANGE` |
| **MR** 量级扰动 | 数值被整体放大/缩小 | `MR_MINOR(0.5%)`、`MR_MODERATE(2%)`、`MR_SIGNIFICANT(10%)`、`MR_EXTREME(25%)` |
| **multi** | 同时注入多类错误 | 组合类型 |

样本分布：**干净 43 条**，**错误 1942 条**（AE 516 / CL 738 / YOY 516 / MR 172）。
可直接作为**纠错/检测样本模板**：把 `error_type` 当作错误类型标签，
把"注入文本 → 干净文本"当作纠错监督信号。

### FinanceBench —— 真实 SEC 报告上的事实一致性 / 证据回溯

含 150 条开源标注问答（完整版本 10,231 题），每条都带**证据字符串**与**页码**。
可用来构造"原句—错误改写—正确改写"样本，评估模型能否发现并纠正事实不一致。
本仓库已把证据、页码、标准答案、文档元信息（公司/期间/行业）全部对齐到一行。

### FinBen —— 多任务金融 NLP 合集，按需挑选子任务

FinBen 覆盖 23 类任务 / 35 个数据集。本仓库挑选并标准化了其中与
**信息抽取 / 文本分析 / 数值推理**相关的可公开子任务；
受限（需申请授权）的子任务已在 `SOURCES.md` 中列出并给出获取方式。

---

## 三、快速开始

```bash
# 1) 环境（仅 FinBen 部分需要 pandas/pyarrow）
pip install -r requirements.txt

# 2) 重新下载原始数据（可选；sources/ 已包含本次下载结果）
python3 scripts/download_all.py

# 3) 由 sources/ 生成整理后的数据集
python3 scripts/build_datasets.py

# 4) 读取样例
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("data/finverbench_consistency_detection.jsonl", encoding="utf-8")]
print("总数:", len(rows))
print("不一致样本占比: %.1f%%" % (100*sum(r["label"] for r in rows)/len(rows)))
print(rows[0]["sample_id"], rows[0]["label_text"], rows[0]["error_category"])
PY
```

读取 FinBen 的 gzip 文件：

```python
import gzip, json
with gzip.open("data/finben/flare-convfinqa.jsonl.gz", "rt", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f]
```

---

## 四、如何据此构造纠错 / 检测训练数据

1. **一致性二分类（检测）**：直接用 `finverbench_consistency_detection.jsonl`，
   `label` 即标签；`context` 是长文本输入，`instruction` 是指令。
2. **长文本纠错（生成式）**：用 `finverbench_statement_correction.jsonl`，
   `corrupted_text` 作为输入、`corrected_text` 作为目标，可配合 `error_location`
   做局部编辑监督。
3. **事实一致性 / 证据回溯**：用 `financebench_fact_consistency.jsonl`，
   `context` + `question` 为输入，`claim` 为待核验断言，`evidence_text` 为金标证据。
4. **数值断言纠错三元组**：用 `financebench_correction_pairs.jsonl`
   （`correct_statement` / `wrong_statement` / `edit` 三元组，由规则扰动生成、可精确校验）。
5. **结构化字段抽取**：用 `finben/flare-fnxl.jsonl`（数字/XBRL 标签）与
   `finben/flare-finred.jsonl`（关系），可为"财务数字 ↔ 表述"一致性提供中间监督。
6. **混合评测**：用 `unified_consistency_samples.jsonl` 做统一格式的检测评测。

---

## 五、已知限制

- **未包含 PDF 原文**：FinanceBench 官方仓库的 `pdfs/` 约 500 MB，本仓库未纳入；
  但证据原文已保存在 JSONL 的 `context` / `evidence_text` 字段中，多数任务无需 PDF。
  如需 PDF，见 `SOURCES.md` 中的获取命令。
- **FinBen 部分子任务受限**：`flare-finqa`、`flare-fpb`、`flare-fiqasa`、`flare-ectsum`、
  `flare-multifin-en` 需在 HuggingFace 申请授权，本仓库未包含，已给出替代来源与获取方式。
- **`financebench_correction_pairs.jsonl` 为规则生成**：采用固定的 ±8% 数值扰动，
  目的是提供一个可校验的负样本基线；如需更自然的错误改写，可替换为大模型改写。
- **网络通道**：本环境访问 github.com / huggingface.co 主站受限，下载脚本默认走
  jsDelivr / hf-mirror 镜像，国内网络环境下可直接复用。

---

## 六、许可与引用

三个来源均为公开学术基准，使用前请遵守各自的许可与引用要求，
详见 `SOURCES.md` 与 `LICENSE_NOTES.md`。
