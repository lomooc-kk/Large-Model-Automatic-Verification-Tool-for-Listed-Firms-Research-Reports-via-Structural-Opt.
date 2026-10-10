# 本机中文向量服务（CPU、离线推理）

`local_embedding_server.py` 为本地 OpenViking 提供真实的 OpenAI-compatible
`POST /v1/embeddings`，使用中文 BGE-small-v1.5 的 ONNX 量化权重。运行时只读本地数据文件，
不下载模型、不执行远程代码、不调用外部 API。服务固定监听 `127.0.0.1`。

模型来源为 [BAAI 原模型](https://huggingface.co/BAAI/bge-small-zh-v1.5) 和
[Xenova ONNX 转换](https://huggingface.co/Xenova/bge-small-zh-v1.5/tree/75c43b069aac4d136ba6bc1122f995fedcfd2781)。
原模型许可证为 MIT；量化模型用于验证集成，检索质量仍须在目标语料上测量。

## 固定版本与下载大小

- Repository: `Xenova/bge-small-zh-v1.5`
- Revision: `75c43b069aac4d136ba6bc1122f995fedcfd2781`
- `onnx/model_quantized.onnx`: 24,010,842 bytes
- `tokenizer.json`: 439,125 bytes
- 加上四个配置/词表文件：总计 **24,560,715 bytes（约 24.56 MB）**。
- 输出：**512 维**，最后一层 `[CLS]` 向量做 L2 归一化。
- 最大输入：**512 tokens，包含特殊 token**；超出部分右侧截断并在响应中明确报告。

默认缓存目录为仓库根目录下 `data/local-openviking/models/bge-small-zh-v1.5`，已被 `data/` 忽略规则覆盖。
启动时会核验权重和分词器 SHA256，缺失或不匹配均直接失败，不会自动联网补齐。

如需在另一台机器复现，先在仓库根目录运行以下 PowerShell。它只下载固定版本的数据：

```powershell
$revision = '75c43b069aac4d136ba6bc1122f995fedcfd2781'
$modelDir = Join-Path (Get-Location) 'data/local-openviking/models/bge-small-zh-v1.5'
$files = @{
  'onnx/model_quantized.onnx' = '15b717c382bcb518ba457b93ea6850ede7f4f1cd8937454aa06972366cd19bcc'
  'tokenizer.json' = '48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26'
  'config.json' = 'd4193ead3a810fd694fa8a31d7fc72fbaebc0668b603e398734bf2f6538ff42f'
  'special_tokens_map.json' = 'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3'
  'tokenizer_config.json' = 'e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a'
  'vocab.txt' = '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c'
}
foreach ($relative in $files.Keys) {
  $destination = Join-Path $modelDir $relative
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
  Invoke-WebRequest -Uri "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/$revision/$relative" -OutFile $destination
  if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $files[$relative]) {
    throw "模型数据校验失败：$relative"
  }
}
```

## 依赖和前台启动

依赖仅为 `onnxruntime==1.30.0`、`tokenizers==0.23.1`、`numpy>=2.3.5,<3`。
无需安装 PyTorch、Transformers、Sentence Transformers 或 FastAPI。
下面命令由部署操作者选择的隔离环境执行；不会修改全局 Python：

```powershell
.\.venv-openviking\Scripts\python.exe -m pip install -r pdfparse/deploy/requirements-local-embedding.txt
.\.venv-openviking\Scripts\python.exe pdfparse/deploy/local_embedding_server.py --host 127.0.0.1 --port 1934 --threads 4
```

也可以用 `--model-dir` 指定已校验的离线缓存目录。`--threads` 只允许 1..4，ONNX inter-op 为 1，
默认只允许一个推理请求同时执行；并发请求返回明确的 HTTP 503 `server_busy`。
`Ctrl+C` 停止前台服务。服务自身不创建后台进程，也不管理 OpenViking 进程。

## API 与 OpenViking 配置

`GET http://127.0.0.1:1934/health` 返回 `status: ok`、模型 ID、revision、CPU provider、维度、线程数和请求限制。
只有模型加载并通过哈希校验后才开始监听。

```powershell
$body = @{model='bge-small-zh-v1.5-onnx'; input=@('营业收入同比增长。','公司收入增加。'); encoding_format='float'} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:1934/v1/embeddings -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

支持单个字符串或 1..16 个字符串，`encoding_format` 可为 `float` 或 `base64`（little-endian float32）。
每条输入最多 16,384 字符，HTTP body 最多 262,144 bytes；不接受 token-ID 输入、空文本或非 512 维请求。
响应保留标准 `data[].embedding`、`index`、`model` 和 `usage`，并增加：

```json
{
  "data": [{"input_tokens": 1052, "processed_tokens": 512, "truncated": true, "truncated_tokens": 540}],
  "usage": {"prompt_tokens": 512, "total_tokens": 512},
  "truncation": {"max_input_tokens": 512, "input_tokens": 1052, "processed_tokens": 512, "truncated_inputs": 1, "policy": "right_truncate_including_special_tokens"}
}
```

以上展示的是附加字段；真实响应还包含向量。计数包含特殊 token，排除 batch padding；
`usage` 表示模型实际处理量，截断前总量单独保留，不能据此声称整篇长研报已向量化。

在 OpenViking 配置中使用以下 embedding 节点，完整存储和服务配置由 OpenViking 部署脚本管理：

```json
{
  "embedding": {
    "max_concurrent": 1,
    "dense": {
      "provider": "openai",
      "api_base": "http://127.0.0.1:1934/v1",
      "api_key": "local-only",
      "model": "bge-small-zh-v1.5-onnx",
      "dimension": 512
    }
  }
}
```

`local-only` 是本机兼容接口的占位值，不是付费服务密钥。本服务不读取它、不转发它。
配置接口依据 [OpenViking 官方配置文档](https://docs.openviking.ai/en/guides/01-configuration)。
当前 B 适配器采用 `vectors_only` 入库和 `/search/find`；小规模接入验证应先使用 512 token 以内的短 Markdown。
长文档需显式分块并保持 page/block 索引；仅保留原文不等于全部原文已参与向量检索。

## 验证

```powershell
py -3 -m unittest discover -s pdfparse/tests -p test_local_embedding_server.py -v
```

测试覆盖真实中文向量的维度/归一化/相关文本排序、float/base64 一致性、截断及 padding 计数、
真实 loopback HTTP health/embedding、错误模型、坏 JSON、请求大小、单并发和离线文件校验。
模型或运行库缺失时真实模型测试明确 skip，输入边界测试仍执行；不能将 skip 报成接入成功。
本机真实验证也已通过现有系统环境 ORT 1.26.0 / tokenizers 0.23.1 / NumPy 2.4.3；
部署锁定推荐 ORT 1.30.0，需在最终部署环境再验证一次。
