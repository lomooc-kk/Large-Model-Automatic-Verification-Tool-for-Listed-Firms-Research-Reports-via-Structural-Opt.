# 质量审计（AUDIT）

审计对象：官方 CLFEC 全量文件 `data/CLFEC.json`（925 段），
sha256 `807337618f89d0e5384c0553734a925f907bd8eb03182adbbb2a4e230efe006b`，
commit `af4edeae56eb56846532cd1af62fe03a6586a38c`。

审计方法：对每段样本逐条校验编辑区间与文本的一致性，并把 `cors` 施加到 `input_text`
后与 `corrected_text` 逐字比对。脚本见 `scripts/build_clfec_finance.py`（`audit_item`）。

## 1. 结论

| 范围 | 段落 | 有瑕疵 | 进入本子集 |
|---|---:|---:|---:|
| 官方全量 | 925 | **6** | — |
| 其中金融 Finance | 268 | **0** | **268（全部）** |
| 其中法律 Law | 220 | 6 | —（不在本子集范围） |

**金融子集 268 段逐项体检全部通过，零瑕疵**，因此本子集未剔除任何金融样本。
官方全量的 6 处瑕疵**全部落在 Law 领域**，已逐条写入 `audit/quarantine_candidates.jsonl` 留痕
（不静默删除）。

## 2. 官方全量 6 处瑕疵明细

| source_id | 领域 | 拆分 | 问题 |
|---|---|---|---|
| `0f967842-…-8f2f33be10ed` | Law | fec_only | `span_mismatch` + `apply_mismatch`：`[168:175]` 声称 `第二百二十五条`，实际切片为 `》第二百二十五` |
| `1e968cae-…-309bb6502ab7` | Law | mix | `apply_mismatch`：编辑施加后与 `corrected_text` 不符 |
| `a39d0a78-…-83bab7fbfe6e` | Law | mix | `apply_mismatch`：编辑施加后与 `corrected_text` 不符 |
| `e83bac35-…-0645cbd09a8b` | Law | **no_error** | `no_error_violation` + `apply_mismatch`：标为无错但 `input_text != corrected_text`，且 `cors` 为空（差异未被编辑覆盖，如「科技」↔「技术」） |
| `f05484e5-…-90f23a87efa0` | Law | lec_only | `span_mismatch` + `apply_mismatch`：`[849:852]` 声称 `西宁市`，实际切片为 `。西宁` |
| `fdde60bb-…-cce21cbd4ae9` | Law | mix | `span_mismatch` + `apply_mismatch`：`[132:134]` 声称 `遵守`，实际切片为 `法规` |

> 影响面：这 6 段若被直接拿来训练或算分，会引入错误的「错→对」监督信号。
> 本子集已全部排除（且它们本就不属于金融领域）。

## 3. 本子集自身的检查项（`verify_clfec_finance.py` 全过）

1. 来源冻结：`raw/CLFEC.official.json` sha256 == `raw/SOURCE.json` 记录值；
2. 字段合规：`inputs.*` 恰为 `{sample_id, input_text}`；`gold.*` 字段集合固定；
3. 规模与官方口径一致：金融 268 段；`mix 113 / fec_only 57 / lec_only 66 / no_error 32`；
   编辑 534；字符 97,329；
4. **无泄漏**：`inputs.*` 不含任何答案字段；`sample_id` 不含拆分/正误语义；
5. **ID 唯一**：268 个 `sample_id` 全局唯一，且 `inputs` 与 `gold` 逐条对应；
6. **正确性**：所有编辑 `input_text[start:end] == error_word`；所有 `cors` 可精确还原
   `corrected_text`；`no_error` 段 `input == gold` 且无编辑；有错段 `input != gold`。

## 4. 残余风险与提示（不构成错误，但影响解读）

1. **不均衡**：金融子集 236/268 段有错，多数类基线 **88.06%**。只报准确率会严重虚高，
   必须报编辑级 P/R/F1 与 `no_error` 误报率。
2. **`no_error` 仅 32 段**：误报率可测但样本偏少。
3. **无证据原文**：CLFEC 未发布事实纠错所依据的原始证据文档，无法在该集上直接测
   「给定原始材料后的核验能力」。
4. **人工复核建议**：本子集通过了机器可校验的全部不变式，但**未**逐条人工复核
   语义层面「金标准是否确实更正确」。若要作为高置信度评测集，建议对
   `mix` + `fec_only`（170 段）做一轮人工抽检。
