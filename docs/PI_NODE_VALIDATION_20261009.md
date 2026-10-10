# Pi 架构与节点改造：离线验证记录（2026-10-09）

本轮已经验证数据准入边界，并用生产代码完成相邻证据片段归一化的离线对照。**没有实时 Pi 模型推理或 OpenViking 检索的准确率结果。** 下表中的改善来自历史候选的定位表达规范化，不能归因于 Pi、OpenViking、提示词更新或模型知识增长。

本文结果引用已生成的 JSON；所有产物路径均相对仓库根目录。正式回放记录了所用源码 SHA-256，便于区分基线提交与本轮工作区实现。各项实验的新模型调用均为 0。

## 数据准入

准入检查只读原始数据，核对输入/金标配对、定位、答案字段隔离、提示词字段、任务与评分器、来源哈希、分母及用途。`ready` 表示该用途的结构条件满足，不代表人工业务签核或模型性能达标。

| 数据与用途 | 结果 | 关键口径 |
|---|---|---|
| FinED 历史 442 篇，`regression/all_hints` | ready | 1880 个归档文件哈希通过；1747 条可评分金标、45 条排除标注 |
| 同一 442 篇请求 `evaluation` | blocked | 已经推理、曝光和诊断，只能作为回归集，不能称为新 holdout |
| CLFEC 金融子集，`regression` | ready | 268 段、534 处原生编辑、32 段干净；单独的纠错任务与评分 |
| 五类已接入外部任务，`development/dev` | ready | 每个任务使用自己的输入与 scorer；不并入 FinED 总分 |

FinED 的 `all_hints`、`candidate`、`confirmed` 分开报告。下面的节点对照只使用 **全部提示 `all_review_hints_detection`**，没有把自动确认精度与候选发现能力混在一起。

CLFEC 的 `mix/fec_only/lec_only/no_error` 是错误组成分组，不是 train/dev/test。它的事实纠错没有随附来源证据文档，适合中文段落纠错诊断，不能直接证明给定财报材料的核验能力。其四类标签保留独立口径，不自动映射成 FinED 15 类。

外部 FinVerBench 的一致性标签、FinanceBench 的 QA/断言核验/纠错与 FinBen 辅助任务不能混成一个准确率。特别是 claim 核验必须同时输入 `claim` 与 `evidence_text`；只输入证据正文会遗漏待核验断言。原始 A/B 样本 ID 编码标签，准入投影将 ID 留在提示词之外。FinVerBench 检测 dev 共 464 条，排除 35 条模糊标签后剩余 429 条全部有错，不能据此估计干净文本误报率。FinBen 子任务尚未接入这道主线准入。

准入证据：

- [442 回归准入](../output/node-audit-20261009/readiness/fined442-regression.json)
- [442 作为 evaluation 的阻断记录](../output/node-audit-20261009/readiness/fined442-evaluation-blocked.json)
- [CLFEC 准入](../output/node-audit-20261009/readiness/clfec-regression.json)
- [外部 claim 开发集准入](../output/node-audit-20261009/readiness/financebench_claim_verification-development.json)
- 其余四类外部任务：`output/node-audit-20261009/readiness/<task>-development.json`。

## 生产节点对照

正式脚本调用 `yjcheck.span_normalization.normalize_candidate_spans`，没有在实验脚本复制节点实现。它只合并**同一个既有候选**内已经精确对应原文的重叠、紧接或仅隔空白的片段；不跨候选、不填入缺失词、不改变错误类型、状态和候选数量。

准入在独立子进程执行，结构审计只返回元信息。主回放进程先构造完全部 source-only 变体，最后才打开金标评分。两边都使用原来的严格同类型、完整片段包含、一对一最大匹配评分器；基线复算与归档指标一致。

| 组别 / 全部提示口径 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| model_direct，历史原版 | 1211 | 604 | 536 | 66.72% | 69.32% | 68.00% |
| model_direct，片段归一化 | 1233 | 582 | 514 | 67.93% | 70.58% | 69.23% |
| hybrid，历史原版 | 1235 | 625 | 512 | 66.40% | 70.69% | 68.48% |
| hybrid，片段归一化 | 1257 | 603 | 490 | 67.58% | 71.95% | 69.70% |

两组各处理 442 篇；direct 的候选数保持 1815，hybrid 保持 1860。各组归一化 55 个候选、合并 64 个边界，**22 篇各恢复一个严格命中：TP +22、FP −22、FN −22**。1747 条可评分金标及 45 条排除标注保持不变，没有缺失预测文档。

这说明部分损失来自同一个错误的连续证据被拆成多个 span。它不等于模型多发现了 22 个新错误，也不证明剩余候选的业务语义正确，更不构成独立测试集上的收益证明。

正式产物：

- [指标、完整分母、准入报告与源码哈希](../output/node-audit-20261009/formal-node-ablation/results.json)
- [逐候选变更证据](../output/node-audit-20261009/formal-node-ablation/changed-candidates.jsonl)

## CLFEC 评分器自检

本地 CLFEC scorer 对金标全文和预测全文采用相同的确定性字符编辑分段，避免上游相邻标注的切分方式导致正确全文无法得到满分。本批 **534 处原生标注**对应 **577 处规范化编辑**；两个分母分开留痕，不声称规范化分数等于上游官方分数。

| 纯评分对照，无模型推理 | 规范化编辑 TP/FP/FN | 段落完全匹配 |
|---|---|---|
| 将输入原样回显 | 0 / 0 / 577 | 32 / 268（11.94%） |
| 直接以金标全文作为 oracle | 577 / 0 / 0 | 268 / 268（100%） |

这些数值只验证计分完整性。oracle 明确使用答案，不是待测模型输出，不可当成模型准确率。缺失预测保留分母；干净段落为 32 条，FPR 只按干净分母计算，干净预测缺失时不输出完整 FPR。

证据：[CLFEC 评分器 sanity JSON](../output/node-audit-20261009/readiness/clfec-scorer-sanity.json)。

## 复现与待补验证

在仓库根目录执行。回放输出目录必须位于源数据包之外且尚不存在；复跑时换一个新目录。

```powershell
python evals/dataset_readiness.py --task fined --root artifacts/research442-rerun-20261005 --role regression --scorer fined_strict --scope all_hints --out output/validation-replay/fined442-readiness.json

python evals/run_node_ablation.py --package artifacts/research442-rerun-20261005 --out output/validation-replay/span-node

python -m unittest evals.tests.test_dataset_readiness evals.tests.test_node_ablation -v
```

外部集的真实路径、文件哈希、所选用途和数量保留在各准入 JSON 中。其他数据任务的 CLI 与 evaluation 源组/曝光声明格式见 [数据准入说明](DATASET_READINESS.md)。

2026-10-09 最终集成回归 **610 项通过，0 失败、0 跳过**，不把重复运行和 subtest 另计一次：

| 测试命令范围 | 通过数 |
|---|---:|
| `unittest discover -s factcheck/tests` | 300 |
| `unittest discover -s pdfparse/tests`（包含 31 项 OpenViking HTTP/范围测试及 10 项真实 ONNX 测试） | 95 |
| `unittest discover -s evals/tests` | 173 |
| `unittest discover -s frontend/tests` | 25 |
| `npm --prefix agent_runtime test`（实际 Pi SDK） | 15 |
| `unittest discover -s agent_runtime/tests -p test_python_bridge.py` | 2 |

Python 使用仓库 `.venv`，`PYTHONPATH=factcheck/src;pdfparse/src;pdfparse/tests;.`。Node 构建及 `git diff --check` 通过。完整命令见 [实施说明](PI_OPENVIKING_IMPLEMENTATION.md)。本地运行日志位于 `output/node-audit-20261009/`。

业务入口还执行了真实 DOCX 解析、确定性核查和结果落盘：来源补入前保持待复核，读取并校验来源哈希后，真实规则识别研报 200 万元与来源 100 万元的差异；产物清单验证通过。此测试仅模拟调度和检索响应，不能代替真实向量召回质量测试。

回归时发现并修复一个既有 PDF 导出问题：15pt 中文文本框容不下字体，被误判为缺依赖而跳过；调整后长补证说明正常分页，渲染错误不再伪装成依赖缺失。前端 25 项均实际执行。

后续同日已完成本机服务部署、真实 OpenViking 入库检索以及三个真实 Pi 合成案例，16 次语言模型调用保守记账 0.110938 元；独立结果见 [真实服务验证](LOCAL_PI_SERVICE_VALIDATION_20261009.md)。本地 embedding 与范围解析新增测试已计入上表。真实研报准确率与召回质量仍需冻结任务、数据角色、模型、提示、检索条件与评分器后单独测；本报告中的历史节点增益和合成案例都不能替代这些实验。
