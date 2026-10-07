# 数据来源、许可与下载通道

本仓库的数据全部来自三个公开基准。本文件记录**来源、下载地址、许可、引用方式**，
以及本环境实际的**下载通道**（因为 github.com / huggingface.co 主站访问受限）。

---

## 1. FinanceBench

| 项 | 内容 |
|---|---|
| 全称 | FinanceBench: A New Benchmark for Financial Question Answering |
| 仓库 | https://github.com/patronus-ai/financebench |
| 数据规模 | 开源样本 **150** 条问答；完整版 10,231 题；配套 361 条文档元信息 |
| 数据文件 | `data/financebench_open_source.jsonl`、`data/financebench_document_information.jsonl` |
| 许可 | 仓库未附 SPDX 许可文件，使用时请以仓库页声明为准（学术研究用途） |
| 引用 | Islam et al. 2023, arXiv:2311.11944 |

**未纳入的部分**：官方仓库的 `pdfs/` 目录约 500 MB（各公司 10-K/10-Q 原文 PDF）。
如需获取：

```bash
# 本环境下 github.com 不可直连，用 gh-proxy 整仓打包（约 500 MB）
curl -L -o financebench.zip \
  "https://gh-proxy.com/https://github.com/patronus-ai/financebench/archive/refs/heads/main.zip"
```

> 注：本仓库的 JSONL 中已包含证据原文（`evidence_text` / `context` 字段）与页码，
> 绝大多数一致性检测 / 事实核验任务无需额外下载 PDF。

---

## 2. FinVerBench

| 项 | 内容 |
|---|---|
| 全称 | FinVerBench: Evaluating Large Language Models on Financial Statement Verification |
| 仓库 | https://github.com/SiluPanda/finverification-bench |
| 数据规模 | **1985** 个实例（43 家公司；干净 43 条 + 注入错误 1942 条） |
| 核心文件 | `data/benchmark/benchmark.json`（全量）、`data/benchmark/instances/*.json`（示例）、`data/converted/*.json` |
| 许可 | 仓库未附 SPDX 许可文件，使用时请以仓库页声明为准 |
| 参考页面 | https://benchmarklist.com/benchmarks/finverbench/ |

**未纳入的部分**：`data/raw/` 与 `data/processed/`（原始 XBRL 财报，约 400 MB）。
本仓库保留 `data/benchmark/`、`data/converted/`、`src/`、`paper/`，已足够复现全部样本。

---

## 3. FinBen

| 项 | 内容 |
|---|---|
| 全称 | FinBen: A Holistic Financial Benchmark for Large Language Models |
| 仓库 | https://github.com/The-FinAI/FinBen ；评估框架 https://github.com/The-FinAI/PIXIU |
| 规模 | 23 类任务 / 35 个数据集（本仓库挑选可公开的信息抽取 / 文本分析 / 数值推理子任务） |
| 数据托管 | HuggingFace 组织 **TheFinAI**（`finben-*` / `flare-*`） |
| 许可 | 各子集不同：`finben-*` 标注为 **cc-by-nc-4.0**；`flare-*` 部分为 **mit**，部分未标注；以各数据集页面为准 |
| 引用 | Xie et al. 2024, arXiv:2402.12659 |

### 3.1 已下载（可公开获取）

| 子集 | 任务 | 语言 | 获取方式 |
|---|---|---|---|
| `finben-finer-ord` | 命名实体识别 | en | `hf-mirror.com/datasets/TheFinAI/finben-finer-ord` |
| `finben-fomc` | 央行立场分类 | en | `hf-mirror.com/datasets/TheFinAI/finben-fomc` |
| `flare-fomc` | 央行立场分类 | en | 与 finben-fomc 同源 |
| `flare-finred` | 关系抽取 | en | `hf-mirror.com/datasets/TheFinAI/flare-finred` |
| `flare-fnxl` | 数字 / XBRL 标签抽取 | en | `hf-mirror.com/datasets/TheFinAI/flare-fnxl` |
| `flare-tatqa` | 表格+文本数值问答 | en | `hf-mirror.com/datasets/TheFinAI/flare-tatqa` |
| `flare-convfinqa` | 多轮数值问答 | en | `hf-mirror.com/datasets/TheFinAI/flare-convfinqa` |

### 3.2 受限（gated，需申请授权后下载）

以下子集在 HuggingFace 上为受限数据集，匿名访问返回
`403: Access to dataset ... is restricted and you are not in the authorized list`：

| 子集 | 任务 | 替代 / 获取方式 |
|---|---|---|
| `flare-finqa` | 财报数值推理问答 | 原始 FinQA 数据可从 https://github.com/czyssrs/FinQA 获取 |
| `flare-fpb` | 金融情感（PhraseBank） | 原始 Financial PhraseBank 为公开数据，可自行检索获取 |
| `flare-fiqasa` | 金融情感 | 原始 FiQA 数据集 |
| `flare-ectsum` | 财报摘要 | 原始 ECTSum 仓库 |
| `flare-multifin-en` | 多语言金融 | 需 HF 授权 |

授权获取流程：登录 HuggingFace → 打开对应数据集页 → 申请 access →
在页面内填写理由 → 获批后用带 token 的地址下载：

```bash
export HF_TOKEN=<你的token>
curl -L -H "Authorization: Bearer $HF_TOKEN" -o data.parquet \
  "https://huggingface.co/datasets/TheFinAI/flare-finqa/resolve/main/data/test-00000-of-00001-5ed0ee6b1f761c33.parquet"
```

---

## 4. 本环境使用的下载通道（重要）

| 目标 | 主站是否可达 | 实际使用通道 |
|---|---|---|
| github.com（网页/克隆） | ❌ 不可达 | `gh-proxy.com/<原始URL>`（打包 zip） |
| api.github.com（取文件清单） | ✅ 可达 | 直接调用 |
| raw.githubusercontent.com | ⚠️ 可达但严重限速 | 回退通道 |
| cdn.jsdelivr.net（GitHub CDN） | ✅ 可达且快 | **首选**，单文件下载 |
| huggingface.co | ❌ 不可达 | `hf-mirror.com`（镜像） |

`scripts/download_all.py` 已按上表封装好回退逻辑，可直接复用。

---

## 5. 统一引用（BibTeX）

```bibtex
@misc{islam2023financebench,
  title={FinanceBench: A New Benchmark for Financial Question Answering},
  author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
  year={2023}, eprint={2311.11944}, archivePrefix={arXiv}, primaryClass={cs.CL}
}

@misc{finverbench2026,
  title={FinVerBench: Evaluating Large Language Models on Financial Statement Verification},
  year={2026}, note={https://github.com/SiluPanda/finverification-bench}
}

@inproceedings{xie2024finben,
  title={FinBen: A Holistic Financial Benchmark for Large Language Models},
  author={Qianqian Xie and Weiguang Han and Zhengyu Chen and others},
  year={2024}, eprint={2402.12659}, archivePrefix={arXiv}, primaryClass={cs.CL}
}
```
