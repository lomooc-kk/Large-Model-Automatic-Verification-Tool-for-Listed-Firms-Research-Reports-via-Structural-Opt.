# 队友上手卡 · 金融领域语料库（开箱即训版）

> 一句话：700,758 篇中文金融新闻、约 11.8 亿 token，**下载即可给模型灌金融知识**，不用自己写训练代码。

---

## 一、这是什么

| 项 | 内容 |
|---|---|
| 来源 | 人民网科技 / ModelScope `peopletech/Financial` |
| 许可 | **Apache 2.0（可商用）** |
| 规模 | 700,758 篇文档，13.88 亿字，约 **11.8 亿 token** |
| 划分 | 训练 665,605 / 验证 35,153（**零交叉**） |
| 用途 | 大模型**领域继续预训练（CPT）**，灌金融知识 |
| 不是 | ❌ 不是评测集（评测请用 `evals/external_benchmarks/`） |

字段：`id` / `source`（媒体）/ `date` / `title` / `text`

---

## 二、三步开训

```bash
# 0) 下载并解压（Release 标签：finance-corpus-cpt-v1）
unzip finance_zh_corpus_cpt_kit.zip && cd 金融领域语料库_开箱即训版

# 1) 秒级自检 —— 不需要装任何第三方库
python3 train/selftest.py

# 2) 一键训练 —— 自动装依赖 + 下模型 + 自检 + 训练
bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --lora --install
```

`run_cpt.sh` 会自动完成：查数据 → 装依赖 → 下模型 → 自检 → 训练，全程无需手写配置。

---

## 三、硬件档位（按显卡选一条）

| 档位 | 显存 | 命令 |
|---|---|---|
| 1.5B · LoRA（先小跑通） | 8–16 GB | `bash configs/qwen2.5-1.5b-lora.sh` |
| **7B · LoRA（推荐）** | ~24 GB | `bash configs/qwen2.5-7b-lora.sh` |
| 7B · 全参（多卡） | 8×A100-80G | `bash configs/qwen2.5-7b-full.sh` |

---

## 四、数据怎么读

```python
import json, gzip
with gzip.open('data/finance_zh.train.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        break
print(d['source'], d['date'], d['title'])   # 媒体 / 日期 / 标题
```

---

## 五、两条铁律（最容易踩）

1. **必须混通用语料 30%~50%**，否则灾难性遗忘（金融变强、通用能力崩）。
   已内置工具：`python3 train/mix_general_corpus.py`（实测比例精确：60% 金融 + 40% 通用 = 通用占比 40%）。
2. **语料是"米"不是"尺子"。** 它用来给模型灌知识；评测另用 `evals/` 下的基准，别混。

---

## 六、可选加速（反复训练时用）

```bash
# 预分词打包，省掉每次训练的重复分词
python3 train/tokenize_and_pack.py --data-dir data --out-dir packed
```

LLaMA-Factory 用户：见 `examples/llamafactory/`（已按官方 `stage: pt` 规范写好）。

---

## 七、已知边界（提前知道，避免踩坑）

- **体裁偏新闻资讯**，不含结构化财报表格；"拿原始财报纠研报错"要用带证据的基准。
- **时间分布不均**：2024 + 2023 约占 77%，按年份挑样本时注意。
- **只保证格式与一致性无错**（16 项断言 + 7 项实测全过），新闻本身观点与真伪未逐条核验。

---

## 八、更多文档

| 文件 | 内容 |
|---|---|
| `QUICKSTART.md` | 5 分钟上手 |
| `TRAINING_GUIDE.md` | 超参、显存、融合、评估 |
| `VERIFICATION.md` | 7 项真实运行验证记录（含修掉的 9 个坑）|
| `LICENSE_NOTES.md` | 各来源许可与合规说明 |
