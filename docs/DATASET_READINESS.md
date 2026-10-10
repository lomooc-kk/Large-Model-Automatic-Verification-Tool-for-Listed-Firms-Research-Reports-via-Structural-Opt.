# 数据集准入与独立任务评分

当前实验入口见 [新数据与节点迭代](ACTIVE_DATASET_ITERATION.md) 与 `evals/current_dataset.json`。本页保留通用准入模块及历史接口示例；其中 FinED 示例只供旧诊断材料的结构审查，不用于新推理或效果验证。当前声明节点的实际白名单是 `instruction + claim + evidence_text`，由 `prepare_active_suite.prompt_payload()` 提供，不能省略问题语境。

`evals/dataset_readiness.py` 在架构或节点实验开始前只读检查数据与评测契约。它不调用模型、下载数据、执行数据包里的脚本或修改原始集；报告只含字段约定、数量、哈希和问题描述。`ready` 表示所声明用途在结构上可以运行，不表示金标已人工裁定、模型通过验收或数据没有预训练污染。

## 接入接口

```python
from evals.dataset_readiness import require_dataset_ready, build_inference_payload

config = {
    "task": "fined",
    "root": "artifacts/research442-rerun-20261005",
    "role": "regression",
    "scorer": "fined_strict",
    "scope": "all_hints",
}
report = require_dataset_ready(config)  # 阻断时抛 DatasetReadinessError，.report 有完整原因
payload = build_inference_payload("fined", input_record)
```

也可调用 `audit_dataset(config)`，它始终返回 `status=ready|blocked` 的 JSON 可序列化字典。调用方应在请求模型、载入测试示例或写入知识/记忆系统之前通过这道准入。只向推理节点传递 `build_inference_payload()` 返回值；金标、整个审计配置和带 ID 的原始记录不进入提示词或检索记忆。

`scorer` 是调用方与准入共同检查的契约标识，不能仅修改标识而继续使用另一套评分函数。主入口仍须显式路由相应评分实现；这个模块不会替换 `run_v2.py`，也不会自动接通外部分支代码。

| 数据任务 | 必须使用的 scorer | 推理正文 | 评分边界 |
|---|---|---|---|
| `fined` | `fined_strict` | `content` | 原有严格同类型、所有 span 包含、一对一最大匹配 |
| `clfec` | `clfec_edits` | `input_text` | 本模块 `score_clfec()`；中文段落与编辑纠正 |
| `external:finverbench_detection` | 同任务字符串 | `instruction` + `context` | 原生一致性标签评分，排除模糊标签并留数 |
| `external:finverbench_correction` | 同任务字符串 | `instruction` + `corrupted_text` | 原生文本纠错评分 |
| `external:financebench_qa` | 同任务字符串 | `instruction` + `evidence_text` | 给定证据的 QA 评分 |
| `external:financebench_claim_verification` | 同任务字符串 | `claim` + `evidence_text` | 声明与证据核验评分 |
| `external:financebench_correction` | 同任务字符串 | `instruction` + `wrong_statement` | 原生数值断言纠正评分 |

FinBen 的实体、关系、立场和金融 QA 子任务尚未接入此门禁；传入不支持的任务会阻断，不能将其统一索引计作 FinED 错误实例。外部集的 `adapters.to_project_gold()` 没有主线 `errors/spans`，不能直接送入 FinED 评分器。对于 claim 任务，只喂 adapter 的 `content=evidence_text` 会漏掉待核验声明，必须同时传递 `claim`。

## 数据角色和分母

- `regression`：可用于已曝光样本上的工程回归，报告明确它不是新的独立测试。
- `development`：用于提示、协议或节点开发。文件名为 `test` 不自动提升为独立验收集。
- `evaluation`：要求额外提供与正文哈希绑定的源组角色分配及 `exposure=unseen` 声明；这是一份可检查的人工声明，不是自动证明从未曝光。
- 已识别的 `research442` 归档只能用 `regression`，不能被请求为 `evaluation`。442 篇中 1747 条可评分金标与 45 条排除标注分别报告；不会改变这批历史数据的用途。
- CLFEC 的 `mix/fec_only/lec_only/no_error` 是错误组成分组，使用 `groups` 参数；它们不是开发/测试划分。给 `split=fec_only` 会阻断。
- 外部 `dev/test` 角色按数据包的源组索引检查；同一来源组不可同时出现。正式 evaluation 仍需额外的曝光声明。

FinED 必须显式选择 `scope=all_hints|candidate|confirmed`。三者分别对应原 runner 的 `all_review_hints_detection`、`candidate_detection`、`verified_detection`。全部提示应包括被拒绝而未被代表的提示；candidate 两组有不同的拒绝保存方式，不能用它替代总人工队列负担。confirmed 只包含程序确定性证明，模型自述不能升级确认；当前模型直接组没有确认器，其确认数不能当作模型发现能力。

CLFEC 保留 `Fact_Error/Word_Error/Grammar_Error/Punc_Error` 四类，并只映射到内部的 `factual/lexical/grammatical/punctuation` 说明名称，**没有**暗中映射为 FinED 15 类标签。`score_clfec(inputs, golds, predictions)` 接受 `{sample_id, corrected_text}` 预测：

1. 全部输入都在段落 EM、检测和编辑分母内，缺失预测不会被跳过；重复和未知预测 ID 报错。
2. 字符编辑用同一个 `SequenceMatcher(autojunk=False)` 对金标全文和预测全文做规范化，避免上游相邻编辑拆分方式导致“输出完全正确却不满分”。报告另留上游标注编辑数，指标明确叫 `canonical_edit_correction`，不声称等同上游官方分数。
3. FPR 分母只含原文与金标相同的干净段落。若干净段落存在缺失预测，完整 FPR 为 `null`，并独立报告已完成干净段落的观察值及缺失数。
4. 本批 268 段、534 处原生编辑、32 段干净。事实纠错没有对应证据文档，不能据此宣称给定来源材料的核验能力。

全量评分器 sanity 校验得到 577 处规范化编辑，原生标注仍为 534 处，两个分母必须分别注明。把输入原样回显得到段落 EM 32/268、编辑 TP=0/FN=577；把金标全文直接作为 oracle 得到 EM 268/268、规范化编辑 TP=577/FP=0/FN=0。这仅验证计分完整性，不是模型效果。

外部 FinanceBench claim 数据的 A/B ID 编码正误，原数据包规范 `to_prompt` 会剥除 ID。本模块通过白名单再次保证 ID 不入提示；试图在 `prompt_fields` 中加入 `sample_id/doc_id/group_id` 会阻断。整个输入对象中嵌套答案字段也会被拒绝。

## CLI

报告必须输出在数据源目录之外，避免覆盖原始文件。成功退出 0；阻断返回非零。

```powershell
python evals/dataset_readiness.py --task fined --root artifacts/research442-rerun-20261005 --role regression --scorer fined_strict --scope all_hints --out output/readiness/fined442.json

python evals/dataset_readiness.py --task clfec --root <clfec_finance目录> --role regression --scorer clfec_edits --out output/readiness/clfec.json

python evals/dataset_readiness.py --task external:financebench_claim_verification --root <external_benchmarks目录> --role development --split dev --scorer external:financebench_claim_verification --out output/readiness/claim-dev.json

python -m unittest evals.tests.test_dataset_readiness -v
```

evaluation 所需 `--split-manifest` 格式如下；需要覆盖选中数据的每个 ID，且来自同一个 `source_group` 的 assignment 不得跨角色。FinED/CLFEC 的 `input_sha256` 是正文 UTF-8 字节的哈希；external 是默认 `build_inference_payload()` 用 `json.dumps(..., ensure_ascii=False, sort_keys=True)` 序列化后的 UTF-8 哈希。

```json
{
  "assignments": [
    {
      "document_id": "selected-id",
      "role": "evaluation",
      "source_group": "original-document-or-source-family",
      "input_sha256": "sha256-of-input",
      "exposure": "unseen"
    }
  ]
}
```

该 manifest 必须由实际的样本使用记录产生，不能为了通过门禁给已读/已调参的样本填写 `unseen`。主线文档里的旧 `holdout` 文件名也不能覆盖较新的曝光记录。

## 正式节点离线对照

`evals/run_node_ablation.py` 直接调用生产函数 `yjcheck.span_normalization.normalize_candidate_spans`。脚本先在独立子进程中调用 `require_dataset_ready()`；该子进程可以做金标结构检查，但只返回无金标正文的报告。主进程在准入成功后加载历史输入/预测，完成全部 source-only 变体，最后才打开金标做评分。

```powershell
python evals/run_node_ablation.py --package artifacts/research442-rerun-20261005 --out output/node-ablation/new-run
python -m unittest evals.tests.test_dataset_readiness evals.tests.test_node_ablation -v
```

`--out` 必须是数据包之外尚不存在的目录。结果包括 `results.json`（准入报告、协议、代码哈希、完整分母和各组指标）及 `changed-candidates.jsonl`（逐条原始/变体证据）。不会写回历史预测，亦不会调用模型。

首次正式回放使用相同严格评分器、同一历史队列、相同类型与状态，得到：

| 全部提示口径 | 原版 TP/FP/FN | 规范化后 TP/FP/FN | 原版 F1 | 规范化后 F1 |
|---|---|---|---|---|
| model_direct | 1211 / 604 / 536 | 1233 / 582 / 514 | 68.00% | 69.23% |
| hybrid | 1235 / 625 / 512 | 1257 / 603 / 490 | 68.48% | 69.70% |

两组各归一化 55 个候选、64 个相邻边界，22 篇各恢复一个严格命中。442 篇、1747 可评分金标与 45 排除标注未改变；新模型调用为 0。提升表示同一候选证据分段方式造成的评分损失得到修复，不表示模型新学到事实或原有业务争议已解决。该历史语料只能作回归。
