# 本机 OpenViking 与 CPU embedding

这里的 PowerShell 脚本管理本仓库的一对普通后台进程：OpenViking 0.4.23 和本地 ONNX embedding。只绑定 `127.0.0.1`，不创建 Windows 系统服务，不写全局环境变量，也不读取现有对话模型密钥。

前置文件：

- 仓库 `.venv-openviking/Scripts/python.exe`，已安装固定版本 OpenViking 及 `local_embedding_server.py` 所需依赖。
- `pdfparse/deploy/local_embedding_server.py`。
- `data/local-openviking/models/bge-small-zh-v1.5` 中已经下载并通过脚本校验的模型文件。启动脚本本身不安装包或下载模型。

从仓库根目录运行：

```powershell
& .\pdfparse\deploy\start-local-openviking.ps1
& .\pdfparse\deploy\stop-local-openviking.ps1
```

可选启动参数为 `-StartupTimeoutSeconds 120` 和 `-EmbeddingThreads 4`；CPU 线程只能设为 1–4。启动时端口 1933 或 1934 已被占用会直接停止，不会接管或结束其他程序。重复启动前先使用配套停止脚本。

| 内容 | 地址或路径 |
| --- | --- |
| OpenViking HTTP | `http://127.0.0.1:1933` |
| embedding OpenAI 兼容接口 | `http://127.0.0.1:1934/v1` |
| embedding 模型标识 | `bge-small-zh-v1.5-onnx`，512 维、JSON float |
| 两个服务的健康检查 | 各自的 `/health` |
| OpenViking 配置 | `data/local-openviking/ov.conf` |
| 索引工作区 | `data/local-openviking/workspace` |
| PID 与启动身份 | `data/local-openviking/processes.json` |
| 每次启动的独立日志 | `data/local-openviking/*-日期时间.stdout.log` / `*.stderr.log` |

首次运行会生成仅本地配置。已有 `ov.conf` 若与脚本配置不同，会保留原文件并报错，需人工检查后再决定如何迁移，不会覆盖其中的密钥。停止时会逐项核对已记录 PID 的进程创建时间、完整命令行、可执行文件路径以及本仓库的 embedding 脚本或 OV 配置路径。无法核对的进程保持运行并报告；正常停止保留模型、索引、配置和日志。

本配置省略 VLM，B 应继续以 `processing_mode="vectors_only"`、`parse_mode="no_split"` 导入其导出的 Markdown，再用 `/api/v1/search/find` 检索及原文读取。它不提供 VLM 摘要、图片理解或带意图分析的 `search`。原始文件仍保留；embedding 输入上限配置为 512 个估计 token，长文向量表示的覆盖与检索效果要单独验证。

启动成功只证明两项健康检查与进程身份通过。完整验收仍需实际导入、等待任务完成、语义查询和按块读取原文。`openviking-server doctor` 会发出 embedding 探测请求，而且其通用检查会把缺少 VLM 报为失败；这种有意选择的向量模式不以 doctor 全项通过为条件。

OpenViking 以本机开发模式运行，没有 HTTP 身份认证。两端服务只能从本机访问。若未来需要其他机器连接，应另行配置认证及访问边界。

参考：官方 [0.4.23 Windows wheel](https://pypi.org/project/openviking/0.4.23/)、[配置说明](https://docs.openviking.ai/en/guides/01-configuration)、[vectors_only 实现](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking/utils/resource_processor.py)、[VLM 可选配置校验](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking_cli/utils/config/vlm_config.py)。
