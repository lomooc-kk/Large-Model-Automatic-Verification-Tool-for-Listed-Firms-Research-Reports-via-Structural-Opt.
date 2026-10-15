# 12例研报数值错误分析：阅读与复核

主报告为 `研报数值错误_12例深度Case分析报告_20261015.pdf`，共30页。另有同内容的可编辑Markdown。

- 第2-3页：任务理解、执行范围和12例索引。
- 第4-27页：四类各3例，每例两页。先核对原文及历史现象，再看代码边界、假设和最小实验。
- 第28-30页：讨论与实验优先级、已执行检查、来源及未决事项。
- 优先讨论M02（第12页）、I01（第16页）、C02（第24页），再看I03成功对照。

## 样本与批次

12个主案例来自所提供的第201-250篇测评资料，涉及10份不同文档；均为已有测评记录，不是新编的示例句。测评材料可能已经包含人为扰动，不据此声称这是券商原始发表稿中的真实错误。

历史命中、误报、漏检只根据群文件PDF记录。该批次未附完整原始模型JSON、模型参数、提示及处理轨迹，因此报告把可观察事实与根因假设分开。另一个research442包只用于核对正文、金标定位及可计分状态，没有替换群文件的预测结果。

GitHub已经检查，本地`main`已通过`git pull --ff-only origin main`成功更新到`a3a37752c8e7c34ec2f38cab04d6afbb25a27a2e`，该提交日期为2026-10-12。制作完成前又核对了远端main，仍是同一提交。

## 证据目录

|文件|内容|
|---|---|
|任务TXT原件|仅本地保留；公开包不附原件，来源哈希见provenance.json|
|66页群文件PDF原件|仅本地保留；公开包保留12例所需摘录、截图与来源哈希|
|`evidence/cases.json`|12例金标、偏移、上下文、PDF历史状态和校验结果|
|`evidence/selected_source_records.json`|10份正文的精确快照，含上游来源行号与哈希|
|`evidence/pdf_selected_records.json`|群文件中对应文档的原文及两路检测表提取文本|
|`evidence/pdf_locations.json`、`evidence/images/`|原表物理页码、行坐标和12张表头/金标行摘图|
|`evidence/provenance.json`|Git提交、来源标识、完整SHA-256及数值复算|
|`evidence/current_offline_results.json`|最新版代码在10份完整原文上的离线规则输出及覆盖范围|
|`evidence/parser_probes.json`|4组8条解析输入及实际输出，改写不计入12个真实案例|
|`evidence/code_references.json`|逐例核对过的源码位置与行号|
|`evidence/verification.json`|原文、定位和样本真实性检查|
|`evidence/reproduction_check.json`|随附复核脚本的实际复跑结果|
|`evidence/layout_measurements.json`|30页布局测量结果|
|`qa_report.json`|最终PDF页数、文字、位置和排版检查|
|`manifest.json`|本交付包各文件的大小与SHA-256|

原表截图只保留源表头与选中金标行，省略中间其他行；原表文字没有改写。源文件既有的小方框图标以旁边“命中/漏检”文字为准。

## 已执行与未执行

已执行原始正文/金标逐字核对、SHA-256核验、对应PDF原句查找、Decimal算术复算、最新版离线规则检查和4组成对解析探针。12个主案例全部可计分，全部偏移精确。报告已渲染成30页图像检查版式。

离线调用使用`detector="hybrid", chat=None, examples=()`，没有模型运行。12例目标类型且完整包含金标span的离线匹配为0，不代表最新版完整系统召回，也不能与群文件的历史9/12、7/12直接比较。探针只是解析边界说明，没有作为新增真实样本或修复收益。

未实施拟议策略修改或模型重跑；实验设计均在报告中注明。U02报价单位/数值和M03语义缺值需补原始来源后裁定。负责人、复核人和最终签核由团队实际填写。

## 复核命令

仅校验随包正文快照、偏移和计算，不需要第三方Python包或网络：

```powershell
python tools/reproduce_checks.py
```

在本仓库的artifacts/case-analysis-20261015目录下，可加参数重跑当前离线规则和4组探针；同样不调用模型、不需要密钥：

```powershell
python tools/reproduce_checks.py --repo ../..
```

在其他目录运行时需把`--repo`换成实际仓库路径；源码不同版本时允许出现不同结果，先核对提交，不应盲目要求所有版本复现本包输出。

## GitHub阅读版

入口见[README](README.md)，12个案例另存为`cases/`中的独立页面。公开副本将本机绝对路径改为来源标识或相对路径；报告PDF字节不变；原始群文件和任务原件未包含在公开包中，原始文件哈希保留。PDF中关于“绝对来源路径”的说明对应本地原始交付，公开版请按provenance.json的path_kind解释。
