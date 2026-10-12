# 训练手册 · 把金融语料灌进模型

面向要真正跑出效果的人。内容包括：方案选择、超参、显存估算、防遗忘、监控、导出与评估。

---

## 一、先决策：你到底该用哪条路

| | 继续预训练 CPT | 指令微调 SFT | RAG |
|---|---|---|---|
| 解决什么 | **"知道什么"**（知识内化） | **"怎么答"**（格式/风格/任务） | **"事实从哪来"**（可溯源） |
| 数据需求 | 5~10 亿 token 起 | 几千~几十万条指令对 | 不需训练 |
| 成本 | 高（多卡） | 中（LoRA 单卡可跑） | 低 |
| 更新知识 | 慢，要重训 | 慢，要重训 | **快，改索引即可** |

**本语料库（约 11.8 亿 token）是为 CPT 设计的。** 如果你的目标只是"懂金融 + 会做金融问答"，
更省钱的顺序是：**好提示词 → RAG → 小规模 CPT + SFT 组合**。

---

## 二、超参数建议（本任务专用）

CPT 与 SFT 的超参差异很大，以下是针对本语料的建议值：

| 参数 | 建议 | 说明 |
|---|---|---|
| 学习率 | **5e-6 ~ 2e-5** | 全参取小值（5e-6）；LoRA 可取大（1e-5~2e-5） |
| 序列长度 | 2048（小模型）/ 4096（大卡） | 本语料单篇中位 1124 字 ≈ 1500 token，2048 足够 |
| Batch（全局） | 128 ~ 512 条序列 | 单卡 bs=1~2 × 梯度累积 8~16，看显存 |
| Warmup | 2% ~ 3% | 防早期震荡 |
| 调度器 | cosine | 长训稳定 |
| Weight decay | 0.1 | 标准做法 |
| 训练量 | 1000万~1亿 token（LoRA）<br>5亿~全部（全参） | 不是越多越好，要盯验证集 loss |
| 精度 | bf16（Ampere+）/ fp16（较老卡） | 纯 CPU 用 fp32 |
| Gradient checkpointing | 显存紧就开 | 省 30%~40% 激活显存，慢约 20% |

**看 loss 判断何时停**：验证集 loss 连续几个 eval 点不再下降，就该停。过训会让模型只会"续写新闻体"。

---

## 三、显存与硬件估算

| 模型 | 方式 | 显存 | 参考卡型 |
|---|---|---|---|
| 1.5B | LoRA | 8~16 GB | RTX 3090 / 4090 |
| 1.5B | 全参 | 24~40 GB | A100-40G |
| 7B | LoRA | ~24 GB | 4090 24G / A100-40G |
| 7B | 全参 | 8 × 80 GB | 8×A100/H100 |
| 13B | LoRA | 40~48 GB | A100-80G |
| 70B | LoRA | 4 × 80 GB | 4×H100（+ ZeRO / FSDP） |

**估算口诀**：全参训练显存 ≈ 权重（fp16 约 2 字节/参数）× 4（权重+梯度+优化器状态） + 激活。
LoRA 只需存小矩阵，显存主要由权重 + 激活决定。

速度参考：单张 A100，7B LoRA、seq_len 2048、全局 batch 128，约 **1.5~3 秒/步**；
单张 4090 约为其 40%~60%。

---

## 四、数据配比与防遗忘（最容易翻车的地方）

1. **通用语料必须混**：纯金融语料训练 = 通用能力退化。建议通用占比 **30%~50%**。
   ```bash
   python3 train/mix_general_corpus.py \
     --finance data/finance_zh.train.jsonl.gz \
     --general /你的通用语料.jsonl.gz \
     --general_ratio 0.4 --out data/mixed.train.jsonl.gz
   ```
2. **顺序也有影响**：先通用后金融（或持续混合）比"先金融后通用"更稳。
3. **本语料的时间偏斜**：2024+2023 占约 77%。若你要模型懂历史脉络，
   可按年份做**均衡采样**（用 `year` 字段加权）。
4. **近似重复**：本包做了精确去重（2.8%），但"改写转载"未处理。
   对质量要求高时，建议再跑一轮 MinHash/SimHash 近似去重。

---

## 五、训练与监控

### 启动

```bash
# 推荐：单卡 LoRA
bash configs/qwen2.5-7b-lora.sh

# 或自定义
python3 train/train_cpt.py \
  --model_name_or_path ./models/Qwen2.5-7B \
  --data_dir data --output_dir out/qwen7b-lora \
  --use_lora --max_seq_len 2048 --max_steps 30000 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 8e-6 --bf16 --gradient_checkpointing
```

### 先做三件自检（强烈建议）

```bash
python3 train/selftest.py            # 包是否完整
python3 train/train_cpt.py --dry_run --data_dir data --tokenizer_name_or_path ./models/Qwen2.5-1.5B
                                     # 数据管道是否正常
python3 train/train_cpt.py ... --max_steps 20 --max_docs 2000   # 20 步跑通
```

### 日志怎么读

- `loss`：平稳下降正常；突增 → 学习率过大或数据异常；
- `grad_norm`：长期 > 10 说明不稳，考虑降 lr 或加 warmup；
- `eval_loss`：**真正要看的指标**，转升即过拟合/退化，应停止。

想要可视化：`--report_to tensorboard`，然后 `tensorboard --logdir out/`。

---

## 六、LoRA 权重合并与导出

LoRA 训练产出的是适配器（体积小），两种用法：

```bash
# 方式一：训练时就在合并（推荐，省事）
python3 train/train_cpt.py ... --use_lora     # 保存的是适配器，可直接被 peft 加载

# 方式二：事后合并成一个完整模型（便于直接部署）
python3 - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-7B"
adapter = "out/qwen7b-lora"
out = "out/qwen7b-lora-merged"

tok = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out)
tok.save_pretrained(out)
print("已合并到", out)
PY
```

---

## 七、训练完怎么评估（别只看 loss）

1. **语言建模指标**：验证集 perplexity（越低越好，但要注意语料领域差异）。
2. **金融知识基准**：**CFLUE**（金融资格题，注意是 CC BY-NC-SA 非商用）、
   **FinanceIQ**、**FinBen**。
3. **事实一致性/纠错**：本项目的 `evals/external_benchmarks/`（FinVerBench、FinanceBench、CLFEC）。
4. **通用能力回归**：跑一遍通用评测（如 C-Eval 子集），确认**没有灾难性遗忘**。
5. **人工抽检**：让业务同学看 50~100 条生成结果，判断"像不像懂金融的人写的"。

> 提醒：类别不均衡时不要只看准确率。例如"全答有错"的基线可能就有 97% 准确率，
> 必须分任务报 **召回 / 精确率 / 误报率**。

---

## 八、常见陷阱清单

1. ❌ 一上来就微调 —— 先提示词、再 RAG、最后微调。
2. ❌ 把 SFT 当知识注入 —— SFT 改行为，灌知识要 CPT/RAG。
3. ❌ 纯金融语料训 CPT —— 必混 30%~50% 通用语料。
4. ❌ 数据没去近似重复 —— 会过拟合、通用能力掉。
5. ❌ 表格当纯文本喂 —— 财报知识大量在表格，建议转"表头—行内容"的自然语言。
6. ❌ 踩测试集 —— 评测集必须冻结，训练/评测严格分开。
7. ❌ 只看总体准确率 —— 不均衡时要看分类指标。
8. ❌ 忽视许可 —— 商用前逐一核实语料与基准的许可条款。

---

## 九、成本与时间粗估

以 **7B LoRA、单张 A100-40G、训练 30000 步（约 4 亿 token）** 为例：

| 项 | 估算 |
|---|---|
| 训练时长 | 约 20 ~ 30 小时 |
| 云算力成本 | 约 ¥15 ~ 40 / 小时 → 合计约 ¥300 ~ 1200 |
| 存储 | 模型（fp16 约 15 GB）+ 适配器（约 1 GB） |

按 1 亿 token 计算：**7B LoRA 约 8~12 小时**（单卡 A100）。

---

## 十、附录：token 与磁盘换算

- 中文 1 个汉字 ≈ **0.85 个 token**（Qwen 系 BPE，实测本语料 13.88 亿字 ≈ 11.8 亿 token）。
- 本包原始：`train` 1.13 GB（gz） / 解压约 3.6 GB。
- 训练时每 1 亿 token ≈ 100 MB 显存可忽略的中间文件；但**预打包**（`tokenize_and_pack.py`）
  会把 token 落成 int32，1 亿 token ≈ 400 MB，注意磁盘。
