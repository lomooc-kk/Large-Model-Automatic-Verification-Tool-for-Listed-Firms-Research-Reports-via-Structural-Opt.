# CLFEC-Finance：中文金融段落「语言 + 事实」联合纠错集

本目录是从 **CLFEC**（Chinese Linguistic & Factual Error Correction）官方数据集中**筛选出的金融领域子集**，
并按项目约定做了 **输入 / 答案分离**、**来源冻结** 与 **质量自检**。

> 上游：`jiu2021/CLFEC-Dataset`，commit `af4edeae56eb56846532cd1af62fe03a6586a38c`，许可 **MIT**。
> 论文：*CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing*（arXiv:2602.23845，AACL-IJCNLP 2026）。
> 详见 [`raw/SOURCE.json`](raw/SOURCE.json) 与 [`LICENSE_NOTES.md`](LICENSE_NOTES.md)。

## 1. 这个数据集测什么

每条样本是一段中文金融专业文本（监管公告、交易所公告、市场评述、罚单等），其中被**注入了错误**，
要求模型把段落改对。错误分四类：

| 错误类型 | 含义 | 金融场景例子 |
|---|---|---|
| `Fact_Error` | 事实错误 | 停牌日期 `12月23日` → 应为 `12月22日`；条例编号、金额、统计口径与事实不符 |
| `Word_Error` | 用词错误 | `型势` → `形势`，`警钟` → `警种` |
| `Grammar_Error` | 语法错误 | 语序、搭配、虚词误用 |
| `Punc_Error` | 标点错误 | `．` → `。`，逗号/分号误用 |

它最贴合「**研报表述与原始材料不符**」这一目标的地方在于：`Fact_Error` 的标注粒度精确到
字符区间（`start/end`）与正确片段（`candidate_word`），可直接当作纠错/检测样本模板。

## 2. 怎么用（输入与答案分离）

每个拆分都拆成两个文件，**模型只准看 `inputs.*`**：

```
data/inputs.<split>.jsonl   ← 给模型：只有 {sample_id, input_text}
data/gold.<split>.jsonl     ← 标准答案：{sample_id, source_id, corrected_text, cors, ...}
```

四个诊断拆分（与官方定义一致）：

| split | 段落数 | 编辑数 | 用途 |
|---|---:|---:|---|
| `mix` | 113 | 340 | 语言错误与事实错误**同时出现**，最难，考联合纠错（**重点**） |
| `fec_only` | 57 | 69 | 只含事实错误，考纯 FEC（**重点**，直接对应数值/事实纠错） |
| `lec_only` | 66 | 125 | 只含语言错误，考纯语言纠错（辅助） |
| `no_error` | 32 | 0 | 干净段落，用来量**过度纠正（误报率）** |

共 **268 段 / 534 处编辑 / 97,329 字符**。

### 读取示例

```python
import json
inputs = [json.loads(l) for l in open("data/inputs.fec_only.jsonl", encoding="utf-8")]
gold   = {json.loads(l)["sample_id"]: json.loads(l)
          for l in open("data/gold.fec_only.jsonl", encoding="utf-8")}

item = inputs[0]
print(item["input_text"])                 # 喂给模型
g = gold[item["sample_id"]]
print(g["corrected_text"], g["cors"])     # 标准答案 + 编辑定位
```

### 评分

```bash
# 预测文件每行: {"sample_id": "...", "corrected_text": "..."}
python3 scripts/score_clfec.py --split fec_only --pred pred.fec_only.jsonl
```

输出**段落级检测** P/R/F1 + **误报率**、**编辑级纠正** P/R/F1、段落 EM。

> ⚠️ **不要只看准确率**：本子集 236/268 段都是「有错」，全答「有错」的检测基线高达 **88.06%**。
> 必须同时报编辑级 P/R/F1 与 `no_error` 上的误报率。

## 3. 复现与校验

```bash
python3 scripts/build_clfec_finance.py     # 幂等：从 raw/ 重建 data/（可重复运行）
python3 scripts/verify_clfec_finance.py    # 全部断言，退出码 0 表示数据无错
```

`verify` 覆盖：来源哈希冻结、字段合规、规模与官方口径一致、**输入不含任何答案字段**、
ID 全局唯一且 inputs/gold 一一对应、**编辑可精确还原标准答案**、no_error 一致性。
当前状态：**全部通过**。

## 4. 已知边界（务必先读）

1. **不包含证据原文。** CLFEC 只发布「注错段落 + 金标准改写」，**没有**发布事实纠错所依据的
   原始证据文档。因此它测的是「模型用自己的知识 / 自行检索去发现并改正事实错误」，
   **不能**直接用来测「给定原始材料后的核验能力」。这点与 FinanceBench（自带 `evidence_text`）
   的定位不同，两者互补而非替代。
2. **金融 `no_error` 仅 32 段。** 误报率可测，但样本量不大，宜与其它干净样本合并统计。
3. **`lec_only` 未重点使用。** 若只关心「数值/事实」纠错，`mix` + `fec_only` 即可（170 段）。
4. **上游全量数据存在 6 处瑕疵**（均在 **Law** 领域，金融子集不受影响）。详见 [`AUDIT.md`](AUDIT.md)。

## 5. 目录结构

```
clfec_finance/
├── README.md                    本文件
├── SCHEMA.md                    字段与文件规范
├── AUDIT.md                     质量审计（含上游瑕疵定位）
├── LICENSE_NOTES.md             许可与引用
├── raw/
│   ├── CLFEC.official.json      官方原始全量文件（925 段，sha256 冻结）
│   └── SOURCE.json              来源、版本、哈希、许可
├── data/
│   ├── inputs.<split>.jsonl     模型输入（无答案）
│   ├── gold.<split>.jsonl       标准答案
│   ├── MANIFEST.json            清单与统计
│   └── _checksums.sha256        数据校验和
├── audit/
│   └── quarantine_candidates.jsonl  被隔离样本及原因（全量 6 条，全在 Law）
└── scripts/
    ├── build_clfec_finance.py   幂等构建
    ├── verify_clfec_finance.py  断言自检
    └── score_clfec.py           评分器
```

## 6. 引用

```bibtex
@article{kai2026clfec,
  title  = {CLFEC: A New Task for Unified Linguistic and Factual Error Correction in paragraph-level Chinese Professional Writing},
  author = {Kai, Jian and Zhang, Zidong and Chen, Jiwen and Wu, Zhengxiang and Sun, Songtao and Li, Fuyang and Cao, Yang and Liu, Qiang},
  journal= {arXiv preprint arXiv:2602.23845},
  year   = {2026}
}
```
