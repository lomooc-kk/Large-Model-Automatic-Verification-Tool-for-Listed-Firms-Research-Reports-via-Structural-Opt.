# 快速开始 · 5 分钟跑通，一条命令开训

> 目标读者：拿到这个压缩包、想尽快让模型「学金融」的队友。
> 全程只需 3 步。**遇到任何问题先看第六节「常见报错」。**

---

## 一、先搞懂：这个包是干什么的

| 你想做的事 | 用这个包的方式 |
|---|---|
| 让模型**知道**金融知识（术语、政策、市场表达） | ✅ **继续预训练（CPT）** ← 本包主用途 |
| 让模型**会做**某个金融任务（抽取、纠错、问答） | 需另配指令数据做 SFT，本包只提供语料原料 |
| 让模型**查得到**最新/私有知识 | 建索引做 RAG，本包可提供切块文本 |

一句话：**这是"米"，用来给模型灌金融知识；不是"尺子"（评测集）。**

---

## 二、三步开跑

### 第 0 步：确认包是完整的（5 秒）

```bash
cd 金融领域语料库_开箱即训版
python3 train/selftest.py
```

看到 `结果：全部通过` 即可继续。这一步**不需要装任何第三方库**。

### 第 1 步：装环境 + 下模型 + 自检 + 开训（一条命令）

```bash
bash run_cpt.sh \
  --model Qwen/Qwen2.5-1.5B \
  --lora \
  --install \
  --seq-len 2048 \
  --steps 20000
```

这条命令会自动依次完成：

1. 检查数据是否齐全；
2. 安装依赖（torch + transformers + datasets + peft）；
3. 从 **ModelScope** 自动下载基座模型到 `models/`；
4. 跑一遍**数据管道自检**（确认分词、打包无误）；
5. 开始训练，权重存到 `out/cpt`。

> 没有 GPU？把 `--model` 换成更小的模型也能跑通（训练会很慢，仅用于验证流程）。

### 第 2 步：看结果

```bash
ls out/cpt                # 训练好的权重
tail -n 20 out/cpt/*.json # 训练日志（loss 是否在下降）
```

试玩一下：

```bash
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("out/cpt", device_map="auto")
t = AutoTokenizer.from_pretrained("out/cpt")
print(t.decode(m.generate(**t("2024年A股市场", return_tensors="pt"), max_new_tokens=64)[0]))
PY
```

---

## 三、按你的机器选方案（三档，任选其一）

| 方案 | 脚本 | 显存 | 说明 |
|---|---|---|---|
| ① 小试 | `bash configs/qwen2.5-1.5b-lora.sh` | 8~16 GB | 验证流程、快速迭代 |
| ② **推荐** | `bash configs/qwen2.5-7b-lora.sh` | ~24 GB | 主力方案，效果与成本平衡最好 |
| ③ 大规模 | `bash configs/qwen2.5-7b-full.sh` | 8×A100-80G | 多卡全参，追求最强领域能力 |

三档脚本都支持用环境变量覆盖参数，例如：

```bash
MODEL=./models/Qwen2.5-7B OUT=out/my-run bash configs/qwen2.5-7b-lora.sh
```

---

## 四、只有 CPU / 想先跑通流程

```bash
# 1) 装依赖（CPU 版 torch）
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r train/requirements-train.txt

# 2) 生成 200 条冒烟数据 + 只跑自检
python3 train/selftest.py --make_smoke

# 3) 用冒烟数据 10 步跑通（把 smoke 数据当成训练集）
mkdir -p /tmp/smoke/data
cp examples/smoke_200.jsonl /tmp/smoke/data/finance_zh.train.jsonl
python3 train/train_cpt.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B \
  --data_dir /tmp/smoke/data --output_dir /tmp/smoke/out \
  --max_steps 10 --max_seq_len 256 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --eval_docs 20 --eval_steps 10
```

---

## 五、想让模型「别忘了通用能力」（重要）

只用金融语料训练，模型会**金融变强、通用能力退化**。请把通用中文语料按 30%~50% 混入：

```bash
python3 train/mix_general_corpus.py \
  --finance data/finance_zh.train.jsonl.gz \
  --general /你的通用语料.jsonl.gz \
  --general_ratio 0.4 \
  --out data/mixed.train.jsonl.gz
```

然后把 `--data_dir` 指向混合文件所在目录（或把 `mixed.train.jsonl.gz` 重命名为 `finance_zh.train.jsonl.gz`）。
通用语料可以是：维基/百科、通用网页语料、你自己的业务文本等，格式支持 `.jsonl[.gz]` 与 `.txt[.gz]`。

---

## 六、常见报错速查

| 报错 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | 没装依赖 | 加 `--install`，或按第四节手动装 |
| `FileNotFoundError: ... finance_zh.train.jsonl.gz` | 数据缺失/路径不对 | 确认在包根目录执行；确认 `data/` 下有文件 |
| `找不到模型 / 下载失败` | 网络问题 | 手动下载后 `--model <本地路径>`；或配 `HF_ENDPOINT` |
| `CUDA out of memory` | 显存不足 | 降 `--max_seq_len`、`--per_device_train_batch_size 1`、加 `--gradient_checkpointing`、改用 `--use_lora` |
| `分词器缺少 eos_token` | 用错了分词器 | 用基座模型自带分词器，勿混用 |
| loss 不下降或为 NaN | 学习率过大 / 数据异常 | 学习率降到 `5e-6~1e-5`；先跑 `python3 train/train_cpt.py --dry_run` 查数据 |
| 训练极慢 | 在 CPU 上跑 | 换 GPU；或先用冒烟数据验证流程 |

---

## 七、下一步看哪里

- 想调参、算显存、做 LoRA 融合、接评测 → 看 **`TRAINING_GUIDE.md`**
- 想用 LLaMA-Factory 而不是自带脚本 → 看 **`examples/llamafactory/README.md`**
- 想知道数据是怎么清洗出来的、质量如何 → 看 **`README.md`** 与 `stats.json`
- 想复现整条数据生产线 → 看 **`scripts/`** 下的三个脚本

---

## 八、一句话总结

```bash
python3 train/selftest.py && bash run_cpt.sh --model Qwen/Qwen2.5-1.5B --lora --install
```

跑完这条，模型就开始学金融了。
