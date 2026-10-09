# 验证记录（Verification）

本文件记录本训练工具包的**实际跑通结果**，以及验证过程中发现并修复的问题。
目的是让队友拿到包时能确信：这些脚本不是"看起来能跑"，而是**真的跑过**。

---

## 一、验证环境

| 项 | 值 |
|---|---|
| Python | 3.13.15 |
| torch | 2.14.1+cpu（CPU 环境，无 GPU） |
| transformers | 5.19.0 |
| datasets | 5.1.0 |
| peft | 0.21.2 |
| 分词器 | Qwen2.5-1.5B 官方 tokenizer（真实文件，词表 151,936） |
| 验证模型 | 随机初始化的极小 Qwen2 结构模型（9.81 M 参数，**词表与真实模型完全一致**） |

> 说明：验证用"极小模型 + 真实分词器"的组合，是为了在 CPU 与受限磁盘下，
> 用**与真实模型完全相同的配置结构与词表**跑通全链路。真实 7B/1.5B 权重未下载实测。

---

## 二、跑通结果矩阵

| # | 验证项 | 命令 | 结果 |
|---|---|---|---|
| 1 | 包完整性自检 | `python3 train/selftest.py --sample 300 --make_smoke` | ✅ **全部通过（0 失败 0 警告）**，并生成冒烟数据 |
| 2 | 数据管道自检 | `train_cpt.py --dry_run --max_seq_len 256`（**真实 `data/` 目录**） | ✅ 读取 50 篇 / 产出合法序列 / token id 越界检查通过 |
| 3 | **真实训练（全参）** | `train_cpt.py --max_steps 6`（CPU） | ✅ 6 步 + 2 次 eval + 保存模型成功（`eval_loss=11.94`） |
| 4 | **真实训练（LoRA）** | `train_cpt.py --use_lora --max_steps 2` | ✅ 自动识别 7 个目标层，产出 `adapter_model.safetensors` |
| 5 | 预分词打包 | `tokenize_and_pack.py --max_seqs 30` | ✅ 产出 30 条定长序列（HF Dataset） |
| 6 | 读取预打包训练 | `train_cpt.py --packed_dir ...` | ✅ 直接加载，1 步训练通过 |
| 7 | 防遗忘混料 | `mix_general_corpus.py --general_ratio 0.4` | ✅ 60 金融 + 40 通用 = 100 条，比例精确 |
| 8 | 一键入口 | `bash run_cpt.sh --model <本地> --dry-run` | ✅ 全流程 6 步走通 |

对话训练日志摘要：

```
[train] max_steps=6  seq_len=128  bs=1 x ga=1  lr=1e-05  lora=False
{'eval_loss': '11.94', ...}
{'train_loss': '11.94', 'train_runtime': '18.12', ...}
[done] 模型已保存到 /tmp/smoke_out
```

---

## 三、验证中发现并修复的问题

这些问题都是**实际运行才暴露**的，已全部修复：

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| 1 | 自检清单里文件名写错（`qwen3-8b-full.sh` 实际不存在） | 自检误报缺文件 | 更正为 `qwen2.5-7b-full.sh` |
| 2 | 语法检查用 `cfile=/dev/null` 抛 `FileExistsError` | 自检崩溃 | 改用内置 `compile()` 做纯语法检查 |
| 3 | **transformers 5.x 移除了 `warmup_ratio` / `overwrite_output_dir`** | 训练直接 `TypeError` 起不来 | 自动按版本换算 `warmup_steps`，并过滤不支持参数（打印提示） |
| 4 | **transformers 5.x 的 `Trainer` 不再接受 `tokenizer=`** | 训练起不来 | 自动改用 `processing_class=` |
| 5 | transformers 5.x 把 `torch_dtype` 改名 `dtype` | 弃用告警 | 双向兼容 |
| 6 | **流式洗牌缓冲区尾部文档被静默丢弃** | 最多静默少喂 2000 篇 | 数据读完后 flush 缓冲区 |
| 7 | `--dry_run` 预览按 token 截断，显示成 `�` | 让人误以为数据脏 | 改为解码完整序列再截字符 |
| 8 | **断言用 `tokenizer.vocab_size` 当上界，但 EOS 等特殊 token 的 id 恰好等于甚至大于该值** | 自检误报"token 越界"（真实训练不受影响） | 改为按「分词器词表 / len(tokenizer) / eos+1 / 模型词表」取合法上界；并新增**分词器与模型词表匹配**的前置检查 |
| 9 | `--dry_run` 未限制数据量（默认开洗牌时会先扫两千余篇） | 自检慢、且掩盖小样本问题 | dry_run 内关闭洗牌并把读取量收敛到 `--dry_run_docs` |

> 第 3、4 条尤其关键：**如果没实测，队友用新版 transformers 会直接跑不起来。**

---

## 四、尚未覆盖的部分（如实说明）

1. **未下载真实 Qwen2.5-1.5B/7B 权重实测**（磁盘与带宽限制）；
   但分词器、模型结构、词表均与真实模型一致，且训练全链路已跑通。
2. **未在 GPU / 多卡环境实测**（当前环境无 GPU）；
   多卡分支按 `accelerate launch` 标准写法提供，参数为标准配置。
3. **LLaMA-Factory 路径未实测**（需其独立框架环境）；
   其 `dataset_info.json` 按官方 `stage: pt` + `columns: {prompt: text}` 规范编写。
4. **数据内容层面**：只做格式与一致性保证，**新闻观点与事实真伪未逐条核验**。

---

## 五、如何复验

```bash
# 1) 秒级自检（无需第三方库）
python3 train/selftest.py --sample 300

# 2) 数据管道（需要 transformers + 一个分词器）
python3 train/train_cpt.py --dry_run --data_dir data \
        --tokenizer_name_or_path <你的模型或分词器目录> --max_seq_len 256

# 3) 10 步跑通（建议先小规模）
python3 train/train_cpt.py --model_name_or_path <模型> --data_dir data \
        --output_dir /tmp/cpt_smoke --max_steps 10 --max_seq_len 256 \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --eval_docs 20
```

复验日期：2026-10-09
