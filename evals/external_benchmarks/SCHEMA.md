# 字段规范（SCHEMA）

所有成品数据均为 **JSONL**（UTF-8，每行一个 JSON 对象）。除特别说明外，
`context` / `corrupted_text` / `corrected_text` 等长文本字段保留原始换行符 `\n`。

---

## 1. `data/finverbench_consistency_detection.jsonl` —— 一致性判定（1985 条）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 唯一标识，格式 `finverbench/<instance_id>` |
| `source` | str | 固定 `FinVerBench` |
| `task` | str | 固定 `financial_statement_consistency_detection` |
| `language` | str | `en` |
| `company` | str | 公司名 |
| `period` | str | 报表期间，如 `FY2025` |
| `instruction` | str | 任务指令（原文问句） |
| `context` | str | **长文本输入**：格式化后的三张报表（利润表/资产负债表/现金流量表） |
| `label` | int | **标签**：`1`=不一致（含注入错误），`0`=一致 |
| `label_text` | str | `inconsistent` / `consistent` |
| `error_category` | str | 错误大类：`AE`/`CL`/`YOY`/`MR`/`multi`；干净样本为 `none` |
| `error_category_cn` | str | 大类中文解释 |
| `error_type` | str | 具体错误类型，见下表 |
| `error_location` | str | 被修改的科目路径，如 `cash_flow_statement.cash_from_investing` |
| `error_magnitude_pct` | float | 改动幅度（百分比），干净样本为 `null` |
| `original_value` | float | 修改前的正确数值 |
| `modified_value` | float | 注入后的错误数值 |
| `error_description` | str | 错误注入的自然语言描述 |
| `difficulty` | str | 难度标签，如 `baseline` / `easy` |

**`error_type` 取值**：
`AE_ROW_SUM`、`AE_COLUMN_SUM`、`CL_NET_INCOME_TO_RE`、`CL_NET_INCOME_TO_CFS`、
`CL_ENDING_CASH`、`YOY_OPENING_BALANCE`、`YOY_COMPUTED_CHANGE`、
`MR_MINOR`(0.5%)、`MR_MODERATE`(2%)、`MR_SIGNIFICANT`(10%)、`MR_EXTREME`(25%)、
以及 `multi_*`（多类错误叠加）。

---

## 2. `data/finverbench_statement_correction.jsonl` —— 报表纠错对（1942 条）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 格式 `finverbench_corr/<instance_id>` |
| `source` | str | `FinVerBench` |
| `task` | str | `financial_statement_correction` |
| `language` | str | `en` |
| `company` / `period` | str | 公司 / 期间（与干净版对齐） |
| `instruction` | str | 纠错指令 |
| `corrupted_text` | str | **输入**：注入错误后的报表长文本 |
| `corrected_text` | str | **目标**：同一公司同期的干净报表长文本 |
| `error_category` / `error_category_cn` | str | 错误大类及其中文解释 |
| `error_type` / `error_location` | str | 具体错误类型 / 出错科目路径 |
| `original_value` / `modified_value` | float | 正确值 / 被改错的值（即需要还原的目标） |
| `error_description` | str | 错误描述 |
| `difficulty` | str | 难度标签 |

> `corrupted_text` 与 `corrected_text` 仅在被改动的那一处数值上不同，
> 可直接用于局部编辑（edit）监督。

---

## 3. `data/financebench_fact_consistency.jsonl` —— 事实一致性 / 证据回溯（150 条）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 格式 `financebench/<financebench_id>` |
| `source` | str | `FinanceBench` |
| `task` | str | `evidence_grounded_fact_consistency` |
| `language` | str | `en` |
| `company` | str | 公司简称 |
| `doc_name` | str | 文档标识，如 `3M_2018_10K` |
| `doc_type` | str | `10k` / `10q` / `8k` / `earnings` |
| `doc_period` | int | 文档期间（年） |
| `gics_sector` | str | GICS 行业 |
| `question_type` | str | 问题类型（原始标注） |
| `question_reasoning` | str | 推理类型，如 `Information extraction` |
| `question` | str | 问题 |
| `claim` | str | **待核验断言**（即标准答案，如 `$1577.00`） |
| `justification` | str | 答案来源说明（来自 10-K 的哪个科目） |
| `evidence_text` | str | 证据原文（拼接） |
| `evidence_page_num` | list[int] | 证据所在页码 |
| `context` | str | **长文本上下文**：证据整页文本（拼接） |

---

## 4. `data/financebench_correction_pairs.jsonl` —— 数值断言纠错三元组（111 条）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 格式 `financebench_corr/<financebench_id>` |
| `source` | str | `FinanceBench` |
| `task` | str | `numeric_claim_correction` |
| `company` / `doc_name` | str | 公司 / 文档标识 |
| `question` | str | 对应问题 |
| `context` | str | 证据长文本 |
| `correct_statement` | str | **正确改写**（原始标准答案） |
| `wrong_statement` | str | **错误改写**（对答案中的数值按规则扰动后得到） |
| `edit` | obj | `{"from": 原数值串, "to": 扰动后数值串}` |
| `perturbation_rule` | str | 扰动规则，当前为 `rule_based_factor=+0.08`（+8%） |
| `gold_answer` | str | 标准答案（与 `correct_statement` 相同） |

> 只对"含金额/百分比/精确实数"的答案生成该三元组（共 111/150 条）。
> 扰动只改一个数值 token，其余文本逐字保持，因此错误定位精确、便于自动校验。

---

## 5. `data/finben/*.jsonl` —— FinBen 子任务标准化（7 个子集）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 格式 `finben/<subset>/<原始id>` |
| `source` | str | `FinBen` |
| `subset` | str | 子集名，如 `finben-finer-ord` |
| `split` | str | `train` / `valid` / `test` |
| `task` | str | 任务英文标识 |
| `task_cn` | str | 任务中文说明 |
| `language` | str | `en` |
| `instruction` | str | 提示词（即原始 `query` 字段） |
| `input` | str | 输入文本（即原始 `text` 字段；无该字段时等同 `instruction`） |
| `output` | str | 参考答案（即原始 `answer` 字段） |
| `extra` | obj | 其余原始字段（如 `choices`、`gold`、`label`、`token`、`turn`、`dialogue_id`） |

> 体积最大的 `flare-convfinqa` 以 `.jsonl.gz` 保存，读取时用 `gzip.open(..., "rt")`。

---

## 6. `data/unified_consistency_samples.jsonl` —— 统一视图（2207 条）

把可用于"一致性判定"的样本归一成同一套字段，便于混合评测。

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | str | 唯一标识 |
| `source` | str | `FinVerBench` / `FinanceBench` |
| `task` | str | 固定 `consistency_detection` |
| `language` | str | `en` |
| `context` | str | 长文本上下文 |
| `instruction` | str | 指令 / 问题 |
| `claim` | str \| null | 待判定断言；FinVerBench 样本为 `null`（一致性体现在 `context` 整体） |
| `label` | int | `1`=一致，`0`=不一致 |
| `label_text` | str | `consistent` / `inconsistent` |
| `error_category` | str | 错误大类；一致样本为 `none` |
| `gold_output` | str \| null | 参考输出（FinanceBench 样本为正确答案） |
| `meta` | obj | 附加信息（公司、期间、文档、编辑操作等） |

**构成**：FinVerBench 1985 条（标签分布见下）+ FinanceBench 数值断言正/负样本 111×2=222 条。

```
FinVerBench 标签分布：label=1（不一致）1942 条，label=0（一致）43 条
```
