# 外部金融语料库接入

本项目将外部语料分为八类：公司披露、财务/XBRL、监管规则、宏观与货币、市场行情、
新闻舆情、研究文本、评测数据。机器可读来源目录位于
`config/financial_corpus_catalog.json`，记录地区、语言、角色、官方入口、更新频率、许可边界、
接入状态与优先级。

## 安全边界

- `production` 才能进入默认 `financial-evidence` 命名空间。
- `evaluation` / `development` 必须显式传入 `--allow-evaluation`，并使用 `eval-*` 命名空间。
- 文件名为 `gold.*` 的答案文件永远拒绝入库，防止检索泄漏答案。
- 新闻和券商研报只有在取得授权后才能由 `quarantine` 改为 `production`。
- 每条记录保留来源、外部 ID、角色、分类、语言与 SHA256；OpenViking 仅召回文档，证据仍回链本地索引。

## 查看目录

```powershell
$env:PYTHONPATH = "pdfparse/src"
python -m yjparse corpus-catalog
```

## 导入生产 JSONL

输入每行一个 JSON 对象。正文按顺序识别 `content`、`context`、`text`、`evidence`、
`evidence_text`、`context_full_page`、`corrupted_text`；标识识别 `document_id`、`doc_id`、
`wrong_statement`、`claim`、`input`；标识识别 `document_id`、`doc_id`、`sample_id`、`id`、
`accession_number`。即使 FinBen 的答案与输入同处一个源记录，导出器也只提取输入字段，
不会把 `gold_output` 写入索引或 Markdown。

```powershell
python -m yjparse import-corpus `
  --input data/external/sec/records.jsonl `
  --kb data/corpus-kb/sec `
  --source-id sec-edgar-filings `
  --category company_disclosure `
  --role production `
  --project financial-evidence `
  --language en
```

生成物与 PDF 知识库使用同一契约：`markdown/*.md`、`kb_index.jsonl`、
`ingest_manifest.json`。OpenViking 服务启动后执行既有入库入口，Pi 无需改造即可通过
`search_evidence` / `read_evidence` 使用。

## 隔离导入现有外部评测输入

以下示例只用于验证检索链，不得与生产证据库混合：

```powershell
python -m yjparse import-corpus `
  --input evals/external_benchmarks/data/financebench_qa/inputs.dev.jsonl `
  --kb data/corpus-kb/eval-financebench `
  --source-id financebench `
  --category evaluation `
  --role evaluation `
  --project eval-financebench `
  --language en `
  --allow-evaluation
```

## 来源落地顺序

1. 中国上市公司公告：巨潮、上交所、深交所、北交所、港交所；先采用人工/受控下载的 PDF 目录入口。
2. SEC EDGAR：申报全文、Submissions、Company Facts/XBRL；使用官方 API 或夜间 bulk ZIP。
3. 监管与宏观：证监会、人民银行、国家统计局、FRED、World Bank、IMF。
4. 行情数据：仅在明确许可范围内接入，默认不再分发。
5. 新闻和研报：必须有供应商合同或用户授权，未授权内容保持隔离。

“接入完成”必须分别验证：来源下载、哈希/许可审计、本地索引、OpenViking 入库、检索回读、
Pi 工具调用和人工证据抽查；仅生成清单不等于生产验收。
