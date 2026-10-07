# 许可与合规说明

本目录是对三个**公开学术基准**的下载、格式转换与字段标准化，**未修改**其语义内容。
所有原始数据均保留在 `sources/` 中以便对照。

## 各来源许可状态

| 来源 | 仓库 | 许可状态 | 备注 |
|---|---|---|---|
| FinanceBench | github.com/patronus-ai/financebench | 仓库未附 SPDX 许可文件 | 学术研究用途；请以仓库页声明为准 |
| FinVerBench | github.com/SiluPanda/finverification-bench | 仓库未附 SPDX 许可文件 | 学术研究用途；请以仓库页声明为准 |
| FinBen（`finben-*`） | huggingface.co/datasets/TheFinAI/finben-* | **CC-BY-NC-4.0** | 非商业用途，需署名 |
| FinBen（`flare-*`） | huggingface.co/datasets/TheFinAI/flare-* | 部分标注 **MIT**，部分未标注 | 以各数据集页面为准 |

## 使用建议

1. **署名**：使用任一来源时，请按 `SOURCES.md` 中的 BibTeX 引用对应论文。
2. **非商业**：`finben-*` 系列为 CC-BY-NC-4.0，请勿用于商业用途。
3. **受限数据集**：`flare-finqa` / `flare-fpb` / `flare-fiqasa` / `flare-ectsum` /
   `flare-multifin-en` 未包含在本目录中，需自行向 HuggingFace 申请授权后获取，
   并遵守其各自的许可条款。
4. **本目录的转换脚本**（`scripts/`）与本说明文档可自由使用。

## 本目录的派生数据

`data/` 下的文件是对上述原始数据的格式转换与派生：

- `finverbench_*`、`financebench_fact_consistency`、`finben/*`：**格式转换**，
  语义与上游一致，沿用上游许可。
- `financebench_correction_pairs`、`unified_consistency_samples`：**派生数据**
  （数值扰动生成 / 字段归一），由 `scripts/build_datasets.py` 可复现生成，
  沿用其来源数据的许可约束。

**注意**：派生数据中的错误改写出自固定规则（±8% 数值扰动），仅用于构造负样本基线，
不代表真实业务中的错误形态。
