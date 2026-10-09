# 来源、版本与许可

## 一、上游来源（冻结）

| 项 | 值 |
|---|---|
| 数据集 | 金融领域语料库 |
| 提供方 | **人民网科技公司**（ModelScope 账号 `peopletech`） |
| 平台 | ModelScope 魔搭社区 |
| 地址 | https://www.modelscope.cn/datasets/peopletech/Financial |
| **锁定 revision** | `558891a9acb2b56fa3d85f1820724dc515ad2fd9` |
| 上游卡片更新日期 | 2025-03-17 |
| 下载方式 | `git clone`（LFS）或 ModelScope SDK；本目录用 API 直链见 `MANIFEST.json` |

> 上游卡片为默认模板，未提供更细的数据说明。字段与规模系本项目实测得到，见 `stats.json`。

## 二、数据包完整性（已验证）

```
文件    金融领域语料库.zip
字节    1,243,338,095
SHA-256 9222db15829a3f58193b1fef7f98bb129de2707a3b2e3c54f463c71660957b43
```

该 SHA-256 由本地 `sha256sum` 计算，**与 ModelScope 仓库文件元信息中声明的值完全一致**。
`scripts/download_corpus.py` 内置该校验，下载后不匹配会直接失败。

包内实际为两个 JSONL 文件（macOS 打包，含 `__MACOSX` 冗余条目，已忽略）：

| 文件 | 字节 | 行数 | 关键差异 |
|---|---:|---:|---|
| `金融领域语料库1.json` | 154,946,532 | 28,190 | 正文字段为 `contentText`；日期为字符串 |
| `金融领域语料库2.json` | 4,128,186,655 | 705,357 | 正文字段为 `content`；日期为**毫秒时间戳** |

> ⚠️ 两个文件字段名与日期格式**不一致**，`scripts/build_corpus.py` 已做归一（见 `MANIFEST.json` 的 `normalized_schema`）。

## 三、许可与使用限制

- **许可证：Apache License 2.0**（上游仓库 `README.md` 声明）。
- 允许商用、允许再分发，**但须保留版权与许可声明，并标注对本文件的修改**。
- 本仓库对原始数据做了**清洗与格式转换**（去 HTML、归一字段、按哈希划分 train/val），属于"修改"，已在本文档与 `MANIFEST.json` 中注明。
- 原始版权归 **人民网科技公司** 所有；本目录的脚本与文档由本项目编写。

## 四、镜像说明

为便于团队直接取用，本项目在 GitHub Release 上放置了同一数据包的镜像：

- 页面：https://github.com/lomooc-kk/Large-Model-Automatic-Verification-Tool-for-Listed-Firms-Research-Reports-via-Structural-Opt./releases/tag/corpus-finance-zh-v1
- 资产：`finance_zh_corpus_peopletech.zip`（1,243,338,095 字节，sha256 与上游一致）
- 说明：GitHub 发布资产的名称不支持中文字符，故使用 ASCII 名；内容与官方 zip **逐字节一致**。

## 五、复现步骤

```bash
# 1) 取回官方数据包并校验
python3 scripts/download_corpus.py --out-dir ./_cache

# 2) 清洗 + 划分（全量）
python3 scripts/build_corpus.py --zip ./_cache/金融领域语料库.zip --out-dir .

# 只想要画像与样本（不产生大文件）
python3 scripts/build_corpus.py --zip ./_cache/金融领域语料库.zip --out-dir . --no-full
```

产物：
- `data/finance_zh.train.jsonl.gz`、`data/finance_zh.val.jsonl.gz`
- `samples/sample_1000.jsonl`
- `stats.json`
