# 字段与文件规范（SCHEMA）

## 1. 目录内文件

| 文件 | 说明 |
|---|---|
| `data/inputs.<split>.jsonl` | 模型输入，**只含** `sample_id` 与 `input_text` |
| `data/gold.<split>.jsonl` | 标准答案与编辑定位 |
| `data/MANIFEST.json` | 来源、筛选口径、各拆分统计 |
| `data/_checksums.sha256` | 数据文件校验和 |
| `raw/CLFEC.official.json` | 官方原始全量文件（未改动） |
| `raw/SOURCE.json` | 来源、commit、文件哈希、许可 |
| `audit/quarantine_candidates.jsonl` | 被隔离样本及原因 |

`<split>` ∈ `mix` | `fec_only` | `lec_only` | `no_error`。

## 2. `inputs.<split>.jsonl`

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | string | 中性唯一 ID，形如 `clfec-fin-0009`。**不**编码拆分、正误或答案语义 |
| `input_text` | string | 含错误的段落原文，喂给模型 |

> 约束：`inputs.*` 中**不得**出现 `corrected_text`、`cors`、`candidate_word`、`error_word`、
> `error_type`、拆分名等任何答案字段。`verify_clfec_finance.py` 会断言这一点。

## 3. `gold.<split>.jsonl`

| 字段 | 类型 | 说明 |
|---|---|---|
| `sample_id` | string | 与 inputs 一一对应 |
| `source_id` | string | 官方原始 UUID，仅用于溯源 |
| `domain` | string | 恒为 `Finance` |
| `split` | string | `mix` / `fec_only` / `lec_only` / `no_error` |
| `corrected_text` | string | 改正后的金标准段落 |
| `num_edits` | int | 编辑条数（`no_error` 为 0） |
| `error_type_counts` | object | 该段各错误类型计数 |
| `cors` | array | 编辑列表，见下 |

### `cors[]` 元素

| 字段 | 类型 | 说明 |
|---|---|---|
| `start` | int | 起始字符偏移，**左闭右开**，相对 `input_text` |
| `end` | int | 结束偏移（不含） |
| `error_word` | string | 错片段，满足 `input_text[start:end] == error_word` |
| `candidate_word` | string | 正确片段 |
| `error_type` | string | `Fact_Error` / `Word_Error` / `Grammar_Error` / `Punc_Error` |

### 关键不变式

对任一条样本：

1. 对每个编辑，`input_text[start:end] == error_word`；
2. 把所有 `(start, end) → candidate_word` 从后往前施加到 `input_text`，结果**逐字等于**
   `corrected_text`（即 `len(cors) == 0` 时 `input_text == corrected_text`）；
3. `num_edits == len(cors)`；
4. `no_error` 拆分满足 `input_text == corrected_text` 且 `cors == []`。

以上 4 条由 `scripts/verify_clfec_finance.py` 全量断言。

## 4. `sample_id` 命名

- 形如 `clfec-fin-NNNN`，四位零填充，按官方文件中金融段落的**出现顺序**分配，全局唯一；
- 有意**不**包含拆分名（`mix`/`fec_only`）、错误类型或正误标记，避免 ID 泄漏答案；
- 原始 UUID 保留在 `gold.source_id`，便于回溯。

## 5. 编码与格式

- 全部文件 **UTF-8**，JSONL 为「每行一个 JSON 对象」；
- 段落内换行以 `\n` 保留在 `input_text` / `corrected_text` 中；
- 字符偏移按 **Unicode 码点**计数（与官方一致），非字节。
