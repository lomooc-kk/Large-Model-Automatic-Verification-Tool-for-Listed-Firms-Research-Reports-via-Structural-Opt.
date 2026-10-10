# 本机 Pi + OpenViking 真实服务验证

2026-10-09，Windows 原生部署和合成业务集成完成。OpenViking、中文向量服务、真实 Pi SDK、已有 DeepSeek 接口、B 解析与 C 核查全部参与实际执行。本报告的 3 个案例是集成样例，不是独立真实研报测试集，也不报告「准确率 100%」。

## 当前服务

| 组件 | 实际配置 |
|---|---|
| OpenViking | 0.4.23，`http://127.0.0.1:1933`，本机开发模式 |
| 中文 embedding | `http://127.0.0.1:1934/v1`，`bge-small-zh-v1.5-onnx` |
| 推理方式 | 真实 CPU ONNX，512 维，CLS + L2，最多 4 线程、单并发 |
| 模型文件 | 固定 revision `75c43b069aac4d136ba6bc1122f995fedcfd2781`，共 24,560,715 bytes |
| 独立环境 | `.venv-openviking`；ORT 1.30.0、tokenizers 0.23.1、NumPy 2.5.3 |
| 持久化 | `data/local-openviking/workspace`，本地模型与配置均在项目 `data/` 中 |
| Pi | 官方 `pi-agent-core` / `pi-ai` 1.1.0，JSONL 桥接 Python |
| 调度模型 | 现有配置 `deepseek-flash`，思考 `low`，调度输出上限 4096 |

两个服务只监听 loopback，不是公网或团队共享部署。OpenViking 以 `vectors_only` 入库、`find` 检索，没有配置 VLM；embedding 只访问本机模型。部署不依赖 Docker、WSL 或 Ollama 常驻进程。

启停命令见 [本机服务说明](../pdfparse/deploy/LOCAL_OPENVIKING.md)。实际停止和重启已执行；停止脚本修复了 PowerShell 7 将 ISO JSON 时间自动转成 DateTime 导致的字符串比对问题，现在比较 UTC 时间刻度，并继续核对 PID、命令行及可执行文件。停止仅针对本次记录的服务进程。

## 真实入库、检索与重启

三份合成财报 PDF 通过实际 B 解析器导出，保留原始 PDF、SHA256、页码、块和坐标。三个待核查研报在 KB 导出后单独生成，没有加入来源库；期望结果文件在所有 Pi 预测写完之后才读取。

1. 三份 Markdown 实际分词分别为 408、456、406 token，没有截断。
2. 全部通过真实 multipart 上传和 OpenViking 入库任务，任务完成后才检索。
3. 三个查询均命中相应来源，读取 L2 原文并回链本地块，`provider=openviking`，没有 BM25 降级。
4. 入库与检索期间 embedding 完成请求从 27 增至 36；输入和实际处理 token 均从 2229 增至 3833，截断计数保持 0。
5. 停止再启动服务，**不重新入库**，三个查询和原文回链仍全部通过。

这是三份短来源的功能验证，不是排序质量或大规模召回评测。OpenViking 做文档召回，B 适配器再用本地块选择和原文索引恢复位置。512 token 的本地轻量模型不能直接代表长研报的完整向量覆盖；长文须增加保持原文定位的分块，并另测召回。

证据：[真实入库与检索结果](../data/local-openviking/smoke/retrieval-result.json)、[重启后持久化查询](../data/local-openviking/smoke/restart-check.json)。

## 真实 Pi 业务链

三个案例初检时都没有直接上传来源，由 Pi 决定后续检索。错误和正确案例执行了实际 search → read → recheck；缺证案例实际执行检索后仍保留待复核。

| 合成案例 | 最终结果 | 模型调用 | 墙钟时间 |
|---|---|---:|---:|
| 研报归母净利润 1.50 亿元；来源 1.55 亿元 | `confirmed_error`，由 C 规则确认 | 5 | 11.204 s |
| 研报营业收入 12.34 亿元；来源一致 | `no_issue`，仅针对该声明 | 5 | 8.203 s |
| 研保存货 9.69 亿元；来源没有该项 | `needs_review`，未编造依据或正确值 | 6 | 14.844 s |

三例均通过结果文件与 manifest 完整性校验。缺证案例进行了多轮查询，说明后续仍可优化无收益检索的停止条件；本次不能据此声称人工复核已经消除。

新语言模型调用共 **16 次**，按本次唯一 call_id 汇总，保守记账 **0.110938 元**。费用采用当日核验的高峰、缓存未命中费率：输入 2 元、输出 8 元/百万 token；这不是供应商实扣账单。每例 Pi 调度上限 0.5 元，累计账本与历史费用保留，没有增加既有授权额度。[官方费率](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

当前本地费率已更新为上述当日保守值，有效至北京时间 2026-10-09 23:59:59，旧字段另存无密钥审计。后续真实调用仍需核验新的价格有效期。模型、密钥、累计预算和历史账本未更换。

证据：[三例真实 Pi 结果与逐轮记账](../data/local-openviking/smoke/pi-result.json)。结果目录、工具事件和模型调用记录均在该 JSON 中；没有将模型自述当成自动确认。

## 复现与下一步

运行脚本 `tools/run_local_pi_smoke.py` 有 `prepare`、`retrieve`、`pi` 三个阶段。`pi` 明确要求当前价格参数，继续使用原共享账本。前置检索结果绑定数据文件哈希、OpenViking endpoint/target、配置文件和 embedding revision；数据或配置变化需重跑检索门禁。

```powershell
# 已有样例无需重建。重建需指定新的 --root。
.venv\Scripts\python.exe tools/run_local_pi_smoke.py retrieve

# 会调用既有模型 API；先核验当日价格，再填参数。
.venv\Scripts\python.exe tools/run_local_pi_smoke.py pi --input-price <核验值> --output-price <核验值> --price-source <官方页面> --price-valid-until <有效期UTC>
```

最终完整 B 回归 95 项通过（包含 31 项 OpenViking HTTP/范围解析和 10 项真实 ONNX 测试）；结合 C 300、评测 173、前端 25、Node 15、跨语言 2，合计 **610 项工程测试，0 失败、0 跳过**。本地 embedding 也在最终部署隔离环境另行通过同样 10 项测试，重复运行不累加计数。真实服务三例另报，不计入这 610 项，也不与 442 篇历史节点回归混分。

CLI 和界面的默认客户端现在读取证据库清单的合法 `project` 来选择检索范围，显式配置与环境设置保持优先。已通过真实 `yjcheck kb search --directory data/local-openviking/smoke/kb` 验证，无须手填内部 URI。

下一阶段应冻结一个未用于当前调试的业务先导集：10 份报告中的 30 条声明，包含错误、正确、缺证，附可回溯的公告日期与页码，按公司/报告隔离；比较原流程、节点优化、Pi 与 Pi+OpenViking，记录证据召回、严格 TP/FP/FN、缺证处理、时延和费用。PR #11 的新闻 CPT 语料可另作背景材料，其定位见 [语料用途核查](PR11_CORPUS_ROLE.md)，不替代这里的原始财报证据。
