# PR #11 金融语料的用途与接入边界

核验日期：2026-10-09。本轮通过 GitHub 公开 API 核对 PR 与 Release 元信息，并检查已有本地快照中的代码、清单和 1000 条样本。没有下载约 1.2 GB 全量数据，没有执行训练、安装依赖或启动服务。

## 版本与变化

- [PR #11](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./pull/11) 在本轮读取时为 open，head 为 `d822563b94ccc92682a223bcc988b596452babeb`，base 为 `15312b3a7dd73be0179fec69daf7fac5b449a346`。
- PR 共新增 24 个文件，均位于 `corpora/finance_zh_cpt_kit/`。
- 已有 `remote-audit` 的 CPT 快照为 `660f3f2b74da65a1da4e52c3878f65e3fcb21e9a`。与当前 head 的[固定版本比较](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./compare/660f3f2b74da65a1da4e52c3878f65e3fcb21e9a...d822563b94ccc92682a223bcc988b596452babeb)只有 1 个新提交、1 个新增文件 [FOR_TEAMMATES.md](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/FOR_TEAMMATES.md)。训练代码、数据构建代码、统计和样本没有变化。

这个 PR 的主交付是 **CPT 继续预训练工具链及金融新闻语料**。新增上手卡不代表增加了检索服务或新的模型质量结果。

## 声明规模与实际核验范围

| 项目 | PR / 清单声明 | 本轮实际确认 |
|---|---|---|
| 文档量 | 700,758 篇；train 665,605、val 35,153 | 清单中的计数一致；未重新遍历全量文件 |
| 字符量 | 1,387,907,908 字 | 已读取声明，未复算全量 |
| token 量 | 约 1,179,721,721 | 构建脚本实际用 `int(total_chars * 0.85)` 估算；不是全量 tokenizer 实测 |
| 来源媒体 | 1,319 家 | 已读取统计声明；不是逐媒体事实认证 |
| 时间 | 全量统计 1973–2024，主要集中于 2023–2024 | 样本年份为 2008–2024；没有证明其覆盖 2025–2026 的事实 |
| 全量资产 | Release 中的 `finance_zh_corpus_cpt_kit.zip` | API 确认存在，大小 1,200,466,158 字节；未下载、未重算 ZIP 哈希 |
| 样本 | `samples/sample_1000.jsonl` | 已逐行读取 1000 条；Git blob SHA 与当前 head 文件一致 |

全量数量来源：[MANIFEST.json](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/MANIFEST.json)、[stats.json](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/stats.json)。token 估算公式见 [build_corpus.py 第 187 行](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/scripts/build_corpus.py#L187)。

[Release：finance-corpus-cpt-v1](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./releases/tag/finance-corpus-cpt-v1) 资产 API 提供的 digest 为：

```text
sha256:d5060f73869ff84e5c0f6b87b80f760b41d718c6c5ba135eb0237e1677100513
```

这属于远端元信息，不写成本轮下载后验证成功。

## 样本结构与来源链

已核验的 [1000 条样本](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/samples/sample_1000.jsonl)全部只有 8 个字段：

```text
date, id, n_chars, source, split, text, title, year
```

- 1000 条均无 `gold/errors/answer/corrected_text/cors/candidate_word/label/has_error/error_type` 等答案字段。
- 1000 条都有日期；1000 条均无 `url/source_url/original_url`。
- `source` 是发布媒体名，不是可直接打开的原始文章、交易所公告或财报 URL。
- 没有逐条原始文档页码、采集时间、原始记录位置或事实签核状态。构建代码确实仅投影正文、标题、媒体、日期和派生字段，见 [build_corpus.py 第 125–149 行](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/scripts/build_corpus.py#L125)。

样本文件内容校验值：

```text
Git blob SHA-1: 1f6ed96ba07848fa166404a9b04c561d6a4665bd
File SHA-256:  35f17c9adaf49e90a0e92511ab4bf607978b78796130fe0cadf20f53fd52db09
```

包级来源可以追溯：上游声明为 ModelScope `peopletech/Financial`，revision 为 `558891a9acb2b56fa3d85f1820724dc515ad2fd9`，官方原始 ZIP SHA-256 为 `9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43`。这些冻结信息有助于重建同一语料版本，但不等于每条新闻的事实已经核实，也没有补齐原文链接。

## 三种用途要分开

| 用途 | 当前可以做什么 | 不能据此声称什么 |
|---|---|---|
| CPT 领域训练 | 使用训练/打包/混料工具链，以财经新闻训练语言建模能力 | 没有真实大模型在项目纠错任务上的质量收益结果 |
| 金融背景检索 | 按白名单读取纯语料，小批分块并保留媒体、日期、record ID 与哈希，用于术语、背景及核查线索 | 命中新闻不等于事实被证实；训练/验证划分也不是项目评测 holdout |
| 财报事实证据 | 当前语料可提供寻找原始资料的线索 | 缺少公告/财报原文及其定位，不能替代原始来源或单独升级为 `confirmed_error` |

PR 的 [VERIFICATION.md](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/VERIFICATION.md)记录的是极小随机初始化 Qwen2 模型、真实 tokenizer、CPU 上的训练链路冒烟，并明确未实测真实 1.5B/7B 权重及 GPU/多卡。不能把“训练脚本跑通”当作“金融纠错能力已经提升”。

可以在**不读取评测金标**的情况下导入这份纯语料：只允许 corpus 文件路径与上述字段，排除 `evals` 金标、预测和诊断产物。每条入库记录应保留 corpus revision、record ID、正文哈希、媒体、发布时间、导入时间与分块偏移；来源等级明确为“新闻背景，未逐条验证”。抽样未发现答案字段，不等于已完成全量与评测文本的污染检查。

缺少原文 URL 与原始证据定位时，检索结果只能提供候选依据和补证方向。程序应继续执行原有证据与确定性验证要求，不能仅因为 OpenViking 返回了匹配文本就自动确认错误。

## OpenViking 接入状态及 B 的最小交接

PR #11 **没有提供 OpenViking 运行服务、endpoint、embedding 配置、导入脚本或可检索索引**。README 的 [RAG 示例](https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./blob/d822563b94ccc92682a223bcc988b596452babeb/corpora/finance_zh_cpt_kit/README.md#L144)仅演示 `index.add(...)`，不能据此认定服务已启动或语料已入库。

要继续做真实小批导入与检索验证，B 最少提供：

1. **服务 URL**：本机可访问的 OpenViking endpoint、当前运行位置和健康检查方式；如有认证，给出既有凭据配置位置。
2. **embedding 配置位置**：实际生效的配置文件路径，以及 provider、model、向量维度；据此确认客户端与服务端使用同一配置。
3. **已入库范围**：资源 URI/集合或 namespace、语料版本、文档/分块数量和最近入库时间；尚未导入则明确写“未入库”，以这 1000 条样本开始受控验证。

本文件只说明当前材料的用途与缺口，不启动训练、下载全量语料或修改服务配置。
