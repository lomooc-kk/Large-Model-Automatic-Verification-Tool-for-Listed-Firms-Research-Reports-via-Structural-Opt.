# 外部基准集成说明

本目录补充引入 **FinVerBench / FinanceBench / FinBen** 三份公开金融基准，作为 FinED-Bench 主线之外的"一致性检测 / 数值纠错 / 错误注入模板"补充评测与样本源。

## 与主线（FinED-Bench v2）的关系

- **不改动** `data/v2/` 下已有的 997 篇固定划分（开发 598 / 评测 200 / 保留 199），不影响现有 `evals/run_v2.py` 流程。
- **仅作为外部补充**：
  1. **错误类型模板**：FinVerBench 的 AE（算术）/ CL（跨表勾稽）/ YOY（同比连续性）/ MR（量级扰动）四类注入规则，可借鉴用于研报数值矛盾的合成负例，扩展 `negative_review` 难负例池。
  2. **事实一致性 / 证据回溯**：FinanceBench 提供 SEC 10-K 上的问答 + 证据片段，可用于评估 factcheck 模块在"找证据—核实结论"链路上的召回率。
  3. **混合子任务**：FinBen 下的 `flare-fnxl`（财报数字标签）、`flare-tatqa`（表格数值推理）、`flare-convfinqa`（多轮财务问答）可直接作为单能力回归测试。
- **统一视图**：`data/unified_consistency_samples.jsonl` 已把三份基准归一为 `{source, task, text, error_span, correction, label, error_type, evidence}` 公共字段，便于混合评测。

## 文件清单

```
evals/external_benchmarks/
├── README.md                       # 本文件
├── SCHEMA.md                       # 所有 .jsonl 的字段定义
├── SOURCES.md                      # 原始下载来源、许可证、受限子集说明
├── LICENSE_NOTES.md                # 各基准原许可证摘要（MIT/CC-BY-SA-4.0/CC-BY-NC-SA-4.0）
├── requirements.txt                # 运行 build_datasets.py 的依赖（pandas、tqdm）
├── data/
│   ├── finverbench_consistency_detection.jsonl   # 1985 条一致性判定（含 label）
│   ├── finverbench_statement_correction.jsonl    # 1942 条"错误报表→正确报表"纠错对
│   ├── financebench_fact_consistency.jsonl       # 150 条事实一致性 / 证据回溯（含 evidence_text）
│   ├── financebench_correction_pairs.jsonl       # 111 条数值断言纠错三元组（原句/错误/正确）
│   ├── unified_consistency_samples.jsonl         # 2207 条统一视图混合评测
│   └── finben/                                   # FinBen 7 个子任务（见下）
│       ├── finben-finer-ord.jsonl                # 金融命名实体（粗+细标签）
│       ├── finben-fomc.jsonl / flare-fomc.jsonl  # FOMC 货币政策立场分类
│       ├── flare-finred.jsonl                    # 金融关系抽取
│       ├── flare-fnxl.jsonl                      # 财报数字 XBRL 标签抽取
│       ├── flare-tatqa.jsonl                     # 表格数值问答
│       └── flare-convfinqa.jsonl.gz              # 多轮财务对话问答（gzip）
└── scripts/
    └── build_datasets.py            # 可复现：从 sources 原始文件重建本目录 data/
```

## 快速使用

```bash
pip install -r evals/external_benchmarks/requirements.txt

# 跑一致性判定基线（示例：模型直接检测，随机负例）
python3 -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('evals/external_benchmarks/data/finverbench_consistency_detection.jsonl').open()]
print('FinVerBench 总量:', len(rows))
print('标签分布:')
from collections import Counter
print(Counter(r['label_text'] for r in rows))
print('错误类型分布:')
print(Counter(r.get('error_type','clean') for r in rows).most_common(10))
"

# 统一视图混合评测加载器示例
def load_unified(path='evals/external_benchmarks/data/unified_consistency_samples.jsonl'):
    import json
    with open(path) as f:
        for line in f:
            yield json.loads(line)
```

## 建议接入点

1. **扩展难负例**：从 `finverbench_statement_correction.jsonl` 抽取 AE/CL/YOY 类错误，转成 FinED-Bench 的错误 schema 后追加到 `data/v2/negative_review.jsonl`（仍需人工审核后 `approve-import`，不自动注入）。
2. **数值子模块单测**：`flare-fnxl.jsonl` 可直接用于验证 `factcheck/src/` 中数值/单位/期间抽取的正确性。
3. **跨报告一致性评估**：`finverbench_consistency_detection.jsonl` 天然对应"研报—财报配对核查"场景，可独立于 FinED-Bench 跑一条一致性检测评测线。
4. **证据召回评测**：`financebench_fact_consistency.jsonl` 自带 `evidence_text` + `page_ref`，可评估 RAG 检索段落在事实核验中的召回。

## 受限数据说明（重要）

FinBen 中 **5 个子集因 HuggingFace 许可限制未能匿名下载**：`flare-finqa`、`flare-fpb`、`flare-fiqasa`、`flare-ectsum`、`flare-multifin-en`。SOURCES.md 已列出授权链接与替代源（如 FinQA 原始 GitHub、Financial PhraseBank），需要时由具有 HF 账号与 dataset 授权的队友在本地运行 `scripts/build_datasets.py --with-restricted` 补齐。

## 许可证

本目录引入的三个基准分别遵循：
- FinVerBench：MIT
- FinanceBench：MIT License（Patronus AI）
- FinBen / FLARE：CC-BY-SA-4.0（部分子任务 CC-BY-NC-SA-4.0，见 LICENSE_NOTES.md）

均允许学术研究与再分发；商用/竞赛使用前请再次核对各上游 LICENSE 最新版本。
