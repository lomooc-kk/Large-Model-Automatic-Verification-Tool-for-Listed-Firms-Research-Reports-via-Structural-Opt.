# 许可与来源说明（LICENSE_NOTES）

## 1. 上游来源

| 项 | 值 |
|---|---|
| 数据集 | CLFEC (Chinese Linguistic & Factual Error Correction) |
| 仓库 | https://github.com/jiu2021/CLFEC-Dataset |
| commit | `af4edeae56eb56846532cd1af62fe03a6586a38c`（2026-09-22） |
| 原始文件 | `data/CLFEC.json` |
| 文件 sha256 | `807337618f89d0e5384c0553734a925f907bd8eb03182adbbb2a4e230efe006b` |
| 许可 | **MIT License**，Copyright (c) 2026 The CLFEC Authors |
| 论文 | arXiv:2602.23845；AACL-IJCNLP 2026 Main Conference |

## 2. 本目录做了什么

- **只做筛选与重组**：按 `domain == "Finance"` 过滤出 268 段；按官方 4 个诊断拆分分文件；
  把「模型输入」与「标准答案」拆成 `inputs.*` / `gold.*`。
- **未修改任何标注内容**：`input_text`、`corrected_text`、`cors`（含 `start/end`、
  `error_word`、`candidate_word`、`error_type`）均逐字取自官方文件。
- 额外产物（`MANIFEST.json`、`audit/`、`scripts/`）为本仓库原创，同样以 MIT 兼容方式提供。

## 3. 原始文件随附

`raw/CLFEC.official.json` 为官方全量文件的**原样拷贝**（3,262,381 字节，sha256 见上），
随本目录一并保留，便于离线复现与哈希核对。它同样受上游 MIT 许可约束。

## 4. 使用要求

1. 学术或商业使用请遵守 **MIT** 条款，并保留上游版权声明；
2. 发表成果时请**引用上游论文**（BibTeX 见 `README.md` 第 6 节）；
3. 上游数据可能随版本更新，若需复现请固定到上文 commit。

## 5. 免责

本子集仅对公开数据做领域筛选与格式重组，不对上游标注的语义正确性作额外担保；
机器可校验的不变式已由 `scripts/verify_clfec_finance.py` 全量断言并通过，
残余风险见 `AUDIT.md` 第 4 节。
