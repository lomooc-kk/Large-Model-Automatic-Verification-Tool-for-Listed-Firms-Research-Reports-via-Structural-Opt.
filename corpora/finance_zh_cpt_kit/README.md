# 金融领域语料库 · 开箱即训版（finance_zh_corpus）

**目的：让模型学习金融领域知识。** 中文财经新闻语料，已清洗、去重、划分，并附带**开箱即用的继续预训练（CPT）工具链**。

数据来自 **ModelScope `peopletech/Financial`（人民网科技公司，Apache License 2.0）**。

> 一句话：**这是"米"（用来灌知识的语料），不是"尺子"（评测集）。**
> 评测请用同项目的 `evals/external_benchmarks/`。

> 🚀 **只想赶快开始？直接看 [`QUICKSTART.md`](QUICKSTART.md)** —— 一条命令跑通。
> 🎛️ 想调参、算显存、做 LoRA 融合、接评测 → 看 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)。

---

## 一、五分钟跑通

```bash
# 1) 自检（不需要装任何第三方库）
python3 train/selftest.py

# 2) 一条命令：装依赖 + 下模型 + 自检 + 开训
bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --lora --install
```

| 你的机器 | 用哪个 |
|---|---|
| 8~16 GB 显存 | `bash configs/qwen2.5-1.5b-lora.sh` |
| ~24 GB 显存（推荐） | `bash configs/qwen2.5-7b-lora.sh` |
| 多卡集群 | `bash configs/qwen2.5-7b-full.sh` |
| 只有 CPU | 见 `QUICKSTART.md` 第四节 |

---

## 二、成品规模

| 指标 | 数值 |
|---|---|
| 文档总数（可用） | **700,758 篇** |
| 训练集 / 验证集 | **665,605 / 35,153**（95 : 5） |
| 总字数 | **1,387,907,908**（约 13.9 亿字） |
| 估算 token 数 | **约 11.8 亿**（1,179,721,721） |
| 来源媒体数 | **1,319** 家 |
| 单篇长度 | 中位 1,124 字 / 均值 1,980 字 / P90 3,524 字 / 最长 94,126 字 |
| 时间跨度 | 1973 – 2024（峰值 2024 年 412,616 篇） |
| 原始条数 → 保留率 | 733,547 → 700,758（**95.5%**） |

**主要来源**：证券时报 189,814、证券日报 105,619、中国证券报 72,335、上海证券报 60,891、
中国新闻网 10,281、中国经济网 7,349、中国能源报 7,300、经济日报 6,261、新华网 5,807、
界面新闻 5,648 ……（完整 1319 家见 `stats.json`）

> 约 11.8 亿 token，正好落在"领域继续预训练"的可用区间（一般 5~10 亿 token 起）。

---

## 三、文件清单

```
【训练工具链】← 本次新增
  QUICKSTART.md                5 分钟上手（先看这个）
  TRAINING_GUIDE.md            训练手册：超参 / 显存 / 防遗忘 / 融合 / 评估
  VERIFICATION.md              实测验证记录（跑通结果 + 已修问题）
  run_cpt.sh                   一键入口：装依赖 + 下模型 + 自检 + 训练
  train/
    train_cpt.py               主训练脚本（流式 + packing，支持 LoRA / 全参 / dry-run）
    tokenize_and_pack.py       可选·预分词打包，训练时直接加载，省重复分词
    mix_general_corpus.py      混入通用语料，防灾难性遗忘（推荐混 30%~50%）
    selftest.py                免依赖自检（结构 / 数据 / 脚本 / 冒烟样本）
    requirements-train.txt     训练依赖清单
  configs/
    qwen2.5-1.5b-lora.sh       小试（8~16 GB）
    qwen2.5-7b-lora.sh         正式推荐（~24 GB）
    qwen2.5-7b-full.sh         多卡全参
  examples/
    llamafactory/              LLaMA-Factory 接入（dataset_info + yaml + 说明）

【数据】
  data/
    finance_zh.train.jsonl.gz  训练集（665,605 条，1.13 GB）
    finance_zh.val.jsonl.gz    验证集（ 35,153 条， 58 MB）
  samples/
    sample_1000.jsonl          1000 条抽样，便于快速预览
  scripts/
    download_corpus.py         下载并校验官方 zip（锁定 revision + SHA-256）
    build_corpus.py            构建成品（清洗/去重/划分，可复现）
    verify_corpus.py           成品断言校验（16 项，退出码 0 = 全过）
  stats.json                   全量统计画像
  MANIFEST.json                产物哈希与条数清单
  LICENSE_NOTES.md             许可与合规说明
  README.md                    本文件
```

---

## 四、数据格式

每行一条 JSON（JSONL），字段如下：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 全局唯一编号，形如 `fmzh-0000001`（**中性编号，不含任何标签信息**） |
| `title` | string | 标题 |
| `text` | string | 正文（已去 HTML、已归一空白，段落用 `\n` 保留） |
| `source` | string | 发布媒体，如 `证券时报` |
| `date` | string \| null | `YYYY-MM-DD`，缺失为 `null` |
| `year` | int \| null | 年份（便于按时间筛选/切分） |
| `n_chars` | int | `text` 的字符数（与 `len(text)` 严格一致） |
| `split` | string | `train` 或 `val`（与所在文件一致） |

**读取示例**

```python
import json, gzip

def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

for i, rec in enumerate(read("data/finance_zh.train.jsonl.gz")):
    if i >= 3:
        break
    print(rec["date"], rec["source"], rec["title"])
```

---

## 五、三种用法

**① 继续预训练（CPT）——把金融知识灌进权重（本包主用途）**

```bash
bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --lora --install
```

**② 指令微调（SFT）——教"怎么答"**

本语料是纯文本，不直接是"指令—回答"对。可派生任务：摘要、标题生成、关键数字抽取、
事件要素抽取、观点立场判断等：

```json
{"instruction": "请根据以下财经报道，提炼核心事实。", "input": "<正文>", "output": "<摘要>"}
```

**③ RAG——把知识放进索引（推荐用于"事实可溯源"）**

```python
for para in rec["text"].split("\n"):
    if len(para) >= 50:
        index.add({"text": para, "title": rec["title"],
                   "source": rec["source"], "date": rec["date"]})
```

> 金融场景强合规，RAG 能给出处（媒体、日期），比把知识塞进权重更可控、更新更快。

---

## 六、质量保证（"别有错"）

### 6.1 清洗规则

1. **文件级**：自动修正官方 macOS 打包导致的 CP437 文件名乱码（`ΘçæΦ₞ì…` → `金融领域语料库`）。
2. **编码级**：非法/异常字节容错处理；含替换符 `U+FFFD` 的文档一律剔除。
3. **结构级**：去除 `<script>/<style>` 与全部 HTML 标签、解码 HTML 实体、按标签断段。
4. **空白级**：全角空格 `U+3000`、零宽字符、多余空白全部归一，逐行 trim，丢弃空行。
5. **完整级**：正文 < 100 字视为噪声剔除；标题为空不纳入。
6. **重复级**：按 `(title + 正文前 2000 字)` 的 SHA-1 精确去重，**跨两个源文件统一去重**。
7. **划分级**：`train/val` 按文档 ID 哈希稳定切分（95:5），**去重后切分，无跨集泄漏**。

### 6.2 剔除明细（全部留痕，未静默删除）

| 类别 | 条数 |
|---|---:|
| 解析失败 / 非法编码 | **0** |
| 正文为空 | 0 |
| 正文过短（<100 字） | 12,542 |
| 空标题（"上接A1版"类接续文） | 5 |
| 完全重复 | 20,239 |
| 含替换符乱码 | 3 |
| 日期缺失/非法 | 1 |
| （合计保留） | **700,758** |

### 6.3 机器校验

```bash
python3 scripts/verify_corpus.py     # 数据成品：16 项断言，退出码 0 = 全过
python3 train/selftest.py            # 训练包：结构 + 数据抽样 + 脚本/配置 + 冒烟
```

覆盖：ID 全局唯一 · train/val 无重叠 · 精确去重 · 无 HTML 残留 · 无 `U+FFFD` ·
无全角空白 · 无典型乱码字符 · 长度合规 · 字段齐全 · split 与文件名一致 ·
日期格式合规 · 标题非空 · 与 `stats.json`/`MANIFEST.json` 计数与哈希一致。

---

## 七、复现方法

```bash
# 1) 下载官方数据包（锁定 revision，自动校验 SHA-256）
python3 scripts/download_corpus.py

# 2) 构建语料成品（清洗 + 去重 + 划分 + 统计）
python3 scripts/build_corpus.py --zip /path/金融领域语料库.zip --out-dir .

# 3) 校验成品
python3 scripts/verify_corpus.py
```

冻结信息：

- 上游：`ModelScope peopletech/Financial`，revision `558891a9acb2b56fa3d85f1820724dc515ad2fd9`
- 官方 zip SHA-256：`9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43`
- 成品哈希：见 `MANIFEST.json`

---

## 八、许可与引用

- 上游数据：**Apache License 2.0**（人民网科技公司 / ModelScope `peopletech/Financial`），
  允许商用与再分发，需保留许可与声明。
- 训练脚本 `train/`、`configs/`、`run_cpt.sh` 为**本包原创**，随包以 Apache 2.0 提供。
- 再分发本成品时请一并保留 `LICENSE_NOTES.md`。
- 内容为公开财经新闻报道，版权归各原始媒体；如用于对外发布/商用，请自行核验各媒体转载与使用条款。

---

## 九、已知边界

1. **体裁偏新闻资讯**，非结构化财报表格；财报数值/表格类任务建议搭配 XBRL、结构化财报数据。
2. **时间分布不均**：2024 与 2023 两年占约 77%，历史年份样本少，做"时序泛化"需注意采样权重。
3. **单篇长度差异大**（最长 9.4 万字），训练前建议按 `max_len` 切块或做长度分桶。
4. **不含问答/指令标注**，SFT 用途需自行构造指令数据。
5. 本包只做**格式与一致性**层面的"无错"保证；**内容观点、事实真伪未逐条核验**（新闻语料本身以报道为准）。
6. **未做近似去重**：改写转载类重复未处理，高质量训练前建议补一轮 MinHash/SimHash。
