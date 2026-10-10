# B 模块的 OpenViking HTTP 接入

B 先把解析结果导出为带页码、块编号的 Markdown 和 `kb_index.jsonl`，再由
`yjparse.openviking` 调用 OpenViking。OpenViking 负责召回，最终证据正文仍从本地
索引读取，带 `doc_id / block_id / page / bbox / source_path / run_id` 等原始元数据。

## 接口版本与依赖

本适配器对照官方 **OpenViking v0.4.23** 的 HTTP 路由实现，核对日期为
2026-10-09。HTTP 客户端只用 Python 标准库；不需要在解析环境安装整个 SDK。
可选服务器依赖单独固定在 `deploy/requirements-openviking.txt`。

官方依据：

- [v0.4.23 发布页](https://github.com/volcengine/OpenViking/releases/tag/v0.4.23)
- [资源上传与入库路由](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking/server/routers/resources.py)
- [检索路由](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking/server/routers/search.py)
- [L2 原文读取路由](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking/server/routers/content.py)
- [任务状态路由](https://github.com/volcengine/OpenViking/blob/v0.4.23/openviking/server/routers/tasks.py)

| 操作 | HTTP 接口 | 本地行为 |
|---|---|---|
| 健康检查 | `GET /health` | 报告进程是否健康，不声称模型可用 |
| 上传 Markdown | `POST /api/v1/resources/temp_upload` | multipart 上传真实字节，保留页块注释 |
| 入库 | `POST /api/v1/resources` | `temp_file_id`、`vectors_only`、`parse_mode=no_split` |
| 等待入库 | `GET /api/v1/tasks/{task_id}` | 只有 `completed` 才记为可检索；失败、超时明示 |
| 搜索 | `POST /api/v1/search/find` | 限定 resource、L2 与项目 URI |
| 原文读取 | `GET /api/v1/content/read?uri=...&raw=true` | 从原文锚点回查本地索引 |

上传本地文件必须使用 multipart 临时上传；不能把 Windows 本地路径作为远端
HTTP 服务器的 `path` 参数。`vectors_only` 跳过 VLM 摘要生成，但入库和语义查询
仍可能调用服务器配置的 embedding 模型。仅健康检查不调用模型。

## Python 调用

先执行项目已有 `export-kb`，确保目录中有 `ingest_manifest.json`、
`kb_index.jsonl` 和 `markdown/*.md`。然后在已配置 `PYTHONPATH=pdfparse/src`
的 Python 进程中调用：

```python
from pathlib import Path
from yjparse.openviking import (
    OpenVikingConfig, health, ingest_kb, search_kb, read_evidence,
)

config = OpenVikingConfig.from_env()
kb = Path("pdfparse/data/kb")
print(health(config))

# 仅在已配置服务器/模型并决定入库时调用；可能消耗 embedding 额度。
ingest = ingest_kb(kb, config=config, processing_timeout=60)
print(ingest)

result = search_kb(kb, "营业收入", top_k=5, config=config)
for hit in result["hits"]:
    print(hit["doc_id"], hit["page"], hit["block_id"], hit["text"])
    print(read_evidence(kb, hit["doc_id"], hit["block_id"],
                        config=config, uri=hit["uri"]))
```

客户端环境变量见 `deploy/openviking-client.env.example`。默认连接
`http://127.0.0.1:1933`；未传显式配置及 `OPENVIKING_TARGET_URI` 时，`config_for_kb(kb)` 与入库/检索/读取默认从本地 `ingest_manifest.project` 解析合法单段项目范围，缺失才用 `viking://resources/research-reports`，非法项目名直接失败；manifest 不控制服务器地址。
仅 `localhost`、`127.0.0.1` 和 `[::1]` 允许 HTTP；其他服务器地址必须使用 HTTPS，
即使未配置密钥也不允许远端明文传输。服务器要求认证时使用租户 user/admin key。密钥只通过 `X-API-Key` 请求头传输，
不会进入返回结果、配置 repr 或异常正文；客户端拒绝 HTTP 重定向以避免转发密钥。

`health / ingest_kb / search_kb / read_evidence` 还接受可选 `cancel_check` 回调。
每个 HTTP 请求发出前和成功响应返回后都会调用它；宿主取消时回调应抛出异常。
该异常继续向上传播，不转换成 BM25 降级，也不会继续读取下一个远端候选。
宿主可同时将 `config.timeout` 限制为剩余执行秒数。B 模块不依赖 C 的取消实现。

## 回链与状态

成功入库后，`openviking_bindings.json` 保存服务器、资源根 URI、文档 ID、
Markdown 哈希及索引哈希。索引更新后旧绑定失效，需重新导出并入库。
`wait=False` 只返回 `submitted`，不会被当作完成或可用的检索结果；正常验收请用默认
`wait=True`，让适配器观察到任务 `completed`。

检索响应中的 `abstract`、`overview`、`content` 字段都不作为证据。适配器另行读取
L2 内容，解析 `<!-- block:... page:... -->`，再用 `(doc_id, block_id)` 回查本地索引。
未知块、错误页码、失败页、未绑定资源、入库未完成、旧索引和跨服务器 URI 均不能产生
已确认的证据。缺失锚点时返回 `unconfirmed`，不会补造页码或块 ID。

OpenViking 对文档排序后，块内使用本地 BM25 选择片段，结果标记
`block_selection=local_bm25`；没有词项命中时按该原文块顺序返回并标记
`source_order`。正文使用完整索引值，表格保留 `cells`，不会把摘要或 300 字截断片段
伪装成完整证据。

默认服务失败返回 `status=unavailable`，无证据结果。只有调用者显式设置
`allow_fallback=True`，才启用离线兜底，结果明确为 `provider=bm25`、
`status=fallback`、`openviking_used=false`，并保留 OpenViking 失败原因。
服务有响应但证据无法回链时仍返回 `unconfirmed`，不隐藏证据契约问题。

`read_evidence` 不传 `uri` 时仅读取本地原文，标记 `provider=kb_index`、
`openviking_used=false`；传入 URI 时还要求实时远端内容包含相应有效锚点。
远端读取在传输层失败时，只有显式 `allow_fallback=True` 才会退回本地索引，
此时返回 `status=ok`、`provider=kb_index`、`fallback_from=openviking` 并保留失败原因。

搜索与读取返回相同的稳定 `evidence_id`。`evidence.verified=true` 表示块编号和页码
通过上述回链检查；`verification_scope` 区分本地索引及远端锚点。`source_sha256`
（兼容字段 `sha256`）来自导出清单，`source_file_verified=false` 明确表示适配器没有
重新计算当前磁盘原文件。后续 C 核查节点必须先将实际 `source_path` 文件哈希与清单
比对，再重新读取原文件，不能将传输成功或回链成功视为业务结论正确。

## 验证范围

```powershell
py -3 -m unittest discover -s pdfparse/tests -p test_openviking.py -v
```

合同测试启动本机 HTTP 服务器，实际发送 multipart、JSON、任务轮询与内容读取请求；
覆盖超时、失败、无效 JSON、假摘要、未知块、失败页、旧绑定和显式降级。
测试服务只模拟官方接口，不运行真实向量索引，因此通过合同测试不等于已完成真实语义
检索验收。

2026-10-09 本次环境检查：默认 `127.0.0.1:1933/health` 和本地 Ollama 的
`127.0.0.1:11434/api/tags` 均不可达，默认 Python 未安装 OpenViking，未发现
OpenViking/API 模型密钥环境变量。旧 `kb_integration.md` 的 0.4.21 成功记录是历史
开发机记录，不能充当本机当前服务器可用性的证据。此次没有启动付费推理或下载大语料。
