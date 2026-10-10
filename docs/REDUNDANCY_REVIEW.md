# 独立冗余复核节点

`detect_text(..., redundancy_review=False)` 默认关闭独立复核。显式开启后，在基础检测和引用规范化完成后，调用 `yjcheck.redundancy_review.review_redundancy`。它只给已有冗余提示提出复核意见，不重新检测其他类别、不改写原文，也不把模型意见升级为确认错误。

该开关与 `redundancy_context_experiment` 独立：前者增加一个有独立审计的后续阶段，后者改变基础检测提示。仅开启复核，不会同时开启编号与段落角色提示实验。`legacy_rules` 不支持请求模型复核。

**显式 API 用法**

```python
from yjcheck.text_review import detect_text

# budgeted_chat 是应用配置好的 BudgetedChatClient 或保留相同凭证的缓存包装器。
# 沿用现有预算账本、模型配置、费用校验及传输超时。
report = detect_text(
    source_text,
    document_id=source_id,
    detector="hybrid",
    chat=budgeted_chat,
    redundancy_review=True,
    max_input_tokens=16000,
    normalize_spans=True,
)
```

底层接口为 `review_redundancy(content, report, chat, *, max_input_tokens=16000, normalize_spans=True, policy="actions_v1", include_reason=True)`，返回深拷贝报告。已有复核审计的报告不能再次传入覆盖；需要比较时，应保留各版本完整报告。

默认策略仍为 `actions_v1`，对应 `redundancy-review/1.0`。默认请求保持原有消息内容，便于精确缓存复用。新增策略 `context_v2` 需显式选择，当前解析契约对应 `redundancy-review/2.1`；策略和是否传入原理由写入阶段审计。原 R12 的 2.0 结果和失败状态保持原样，2.1 属于后续 r12b 版本。

```python
context_report = detect_text(
    source_text,
    document_id=source_id,
    detector="hybrid",
    chat=budgeted_chat,
    redundancy_review=True,
    redundancy_review_policy="context_v2",
    redundancy_review_include_reason=False,
    max_input_tokens=16000,
    normalize_spans=True,
)
```

`redundancy_review_include_reason` 在底层名为 `include_reason`，必须是布尔值，默认 `True`。设为 `False` 只省略送给复核模型的候选 `reason`，不省略全文、定位、候选 ID 或审计中的原始理由；同一策略两组系统提示保持相同。它是用于比较原理由是否影响判断的独立实验因素，不代表已经证明存在或消除了锚定偏差。选择非默认策略或隐藏理由时，必须同时开启 `redundancy_review`。

**参与范围与一次事务**

仅选择 `errors` 中 `error_type="冗余语句"`、`status="needs_review"`，且每个引用都有有效字符位置、逐字匹配原文的候选。筛选不依赖文档编号、金标或候选来自模型还是规则。非冗余、已确认错误、无效定位候选不参与；`raw_candidates` 与 `rejected_candidates` 保留原内容。

有目标时，完整原文和目标候选按稳定 ID 一次送入 `chat`，`purpose="text_review.redundancy_review"`。没有目标则为 `not_applicable`，不调用模型。全文与候选超出输入预算时不裁剪、不分段猜测，明确返回未完成。节点不自行重试，传输超时由注入客户端负责。

响应每个目标 ID 必须恰好决策一次。全部决策验证通过后才整体应用；任何未知 ID、重复 ID、缺项、无效字段、理由或证据都回退整批。以下三个动作由 `actions_v1` 直接输出；`context_v2` 的动作由程序按语义结论映射。

|决策|含义与要求|对当前提示的影响|
|---|---|---|
|`retain`|保留疑点，记录复核理由；证据列表可为空|原提示不变|
|`withdraw`|给出正常语境解释，至少两段不重叠的精确原文证据|从当前 `errors` 撤回，完整原提示转入独立审计；不记为定位拒绝|
|`revise`|给出新理由、恰好两个不重叠的成员及至少两段支持证据|调整引用和理由，仍为 `needs_review`，保留已有 ID 与原始候选来源|

撤回的 `normal_context` 限于 `topic_expansion`、`summary_or_conclusion`、`different_subject_or_period`、`different_source_or_condition`、`distinct_item_information`。这些分类是模型对主题展开、摘要结论、主体期间、来源条件或条目新增信息的判断，**不是确定性业务证明**。

证据可用 `{text}`：模块仅在全文唯一精确出现时定位。重复引文须提供有效的 `{start,end,text}`，或引用包含编号等信息的完整唯一原文。显式错误坐标不会被猜测修正；所有位置采用 Unicode 字符左闭右开索引。精确匹配只能证明引用存在，不能证明撤回判断正确。

**`context_v2` 的语境声明与动作映射**

每个响应项先给 `context_units`、`new_information`，再给 `verdict` 和 `reason`，不允许模型直接输出 `action`：

- `context_units`：完整相关句或条目的列表，每项为 `{role, quote}`。`quote` 使用上述精确引用协议；各单元互不重叠，且须关联原候选的每段引用。`role` 仅可取 `heading`、`topic`、`explanation`、`evidence`、`conclusion`、`summary`、`parallel_item`、`quotation`、`other`。
- `new_information`：新增命题或篇章作用列表，每项为 `{summary, unit_indices}`。摘要非空，索引回指当前 `context_units`，为非空、去重、有效整数列表。无新增信息时允许空列表，不要求编造信息；也不重复复制原文证据。
- `verdict`：仅可取 `normal`、`redundant`、`uncertain`。`reason` 解释该语义判断。

2.1 仅增加明确的载体兼容，不修改 2.0 的系统提示或实际消息构造：`context_units.quote` 和 `members` 中的裸字符串等同 `{text: 原字符串}`，必须在全文唯一精确出现，不裁剪空白、不修标点、不猜位置；显式错误坐标仍拒绝。`normal_context` 可为原单个合法枚举字符串，或非空、无重复、全部合法的枚举数组。数组保持原顺序和原类型，单元素数组也不转成字符串，不择取其中一项。`actions_v1` 不接受这两种新增载体。

|语义结论|额外要求|程序派生动作|
|---|---|---|
|`normal`|至少两个完整单元，另给受限枚举 `normal_context`；两个单元不要求互相重复，正用于说明正常展开或不同作用|`withdraw`：撤回错误提示，保留正常原文|
|`redundant`|至少一个完整单元，另给恰好两个非重叠 `members`；各成员须被某个单元完整包含并关联原候选|`revise`：依据双成员调整待裁定提示|
|`uncertain`|至少一个相关完整单元；不另给 `members` 或 `normal_context`|`retain`：保持原提示，保存未决理由|

真机械复写可能只涉及句内重复词，因此不一律禁止短成员；但模型应先阅读包含它们的完整单元，不能只抽正常“主题→原因”展开中的共词维持冗余指控。**单元是否完整、角色是否合适、是否存在新增命题，仍是模型声明。** 程序检查的是字段、索引、精确定位、覆盖与关联，不把这些检查说成语义或业务证明，也不会据 `normal` 自动证明原文正确、据 `redundant` 升级为 `confirmed`。

**引用规范化与审计**

`normalize_spans=True` 时，只有 `revise` 的最终展示引用再次使用现有 `normalize_candidate_spans`：相接或仅空白间隔可合并，有非空正文间隔则不合并。设为 `False` 保留两个成员。原决策成员和独立证据仍保留在审计中，`audit.after` 与最终错误一致；不修改评分器或扩大合并规则。

改写不沿用已经失效的旧 `verification_spans`、`source_anchor_proof`、`paired_source_spans` 等证明标记，完整旧内容保存在 `before`。原始 `represented_model_candidates` 不伪造、不改写。撤回时，`model_candidate_representation` 记录 disposition、原 error ID 和审计关联，不留下指向已撤回活动错误的悬空映射。

主要审计位于 `report.redundancy_review`：

- `decisions`：每项 `before`、`after`、决策、精确证据和定位方法；`withdrawn` 保留撤回原提示。
- `raw_response`、`response_sha256`、`runtime_trace`、`request_sha256`：原始响应与调用凭证。
- `base_coverage`、`base_execution_complete`、`status`、`complete`、`skipped`：基础阶段状态、复核范围与完成情况。
- `policy`、`include_reason`、`schema_version`：本次策略身份。v2 的 `decisions[].decision` 还保留规范化的 `verdict`、`context_units`、`new_information` 及派生动作；重复成员规范化为 `spans`，原始 `members` 仍完整保留在 `raw_response`。外层记录 `action_source="programmed_verdict_mapping"` 和 `context_unit_completeness="model_declared_not_deterministically_verified"`。

前端报告中的“**冗余复核：保留、调整与撤回依据**”展开区域可查看这些决定及撤回依据，供人工继续裁定。

**预算凭证与失败覆盖**

成功响应必须有 `status="ok"`、非空 `call_id`、匹配的 `purpose`、有限且非负的 `cost_cny`，以及与当前消息一致的 `request_sha256`。请求哈希沿用客户端算法：`sha256(json.dumps(messages, ensure_ascii=False).encode())`。缺凭证、空 trace、`reserved` 或未知状态不能提交决策。

复核调用追加独立 trace，`job_index` 为原最大值加一，保存完整原 runtime trace。请求阶段计入 `planned_model_calls`，有效完成才增加 `finished_model_calls`；失败调用的费用凭证也保留供账本审计。

调用失败、超时、截断、无效 JSON、超预算或协议验证失败时，保留原预测和原候选映射，设置整体 `coverage.execution_complete/complete/model_complete=False`，追加 `redundancy_review_incomplete`。基础 `processed_ranges` 等范围不变。复核成功也不会覆盖基础阶段原有的不完整状态。阶段完成不等于错误召回完整，更不等于业务判断已确认。

**已实现的评测计划开关**

`evals/run_known_error_repair.py` 的 `plan` 子命令支持 `--redundancy-review`，默认不启用。开关写入冻结计划和候选版本身份，模型请求缓存仍按实际请求严格匹配；开启实验须创建相应新计划，不能修改已有计划冒充同版本结果。

`plan` 也支持 `--redundancy-review-policy actions_v1|context_v2`（默认 `actions_v1`）和 `--redundancy-review-hide-reason`（默认不隐藏）。非默认参数必须配合 `--redundancy-review`，写入冻结计划并绑定候选版本；不能用相同文档输入来代替完整请求匹配。历史计划/1.0 审计缺少策略字段时按 `actions_v1`、`include_reason=True` 解释；`context_v2` 的 2.0/2.1 审计均须显式匹配策略和理由开关。跨解析契约复用还需评测 runner 对源和目标 schema 明确兼容并校验独立收据，不能仅因 API 成功就复制缓存。

`--cached-detection-only` 要求计划同时开启 `--redundancy-review`、`--live`，并提供 `--reuse-model-responses-from` 来源。其含义是基础 detect/global 请求必须命中经过收据验证的精确历史响应，只有独立复核目的允许产生新请求；基础缓存未命中会拒绝联网。若同时启用 `--replay-only`，所有目的的缓存未命中均禁止联网，包括复核。`plan --live` 仅冻结配置，执行还须走已有 `run --live` 和共享预算校验。

复核缓存保存原始 `decisions` 响应与完整 trace，不转换成检测阶段的 `errors`。复用时核对计划、完成收据及响应完整性。历史响应的原费用凭证保留，不能把缓存命中当成新的付费调用。

只有解析契约改变而请求字节未改变时，才可能在新版本下重放原始响应。API 成功但旧协议失败的响应，需要从原 state 锁定的预测收据认证原文、完整响应与完整 trace，再核对新请求 SHA；不能改旧 raw 来冒充成功，也不能把内存诊断当正式回放。新版本应写新目录、保留原失败与历史费用。若改了提示词，旧响应不能代表新提示词结果。

本节点不读取金标、不按样本编号分支。真实效果应以固定完整批次的机械对照和独立业务复核为准；提示消失不自动等于业务修复。局部单元测试或单批对照不能证明达到 95% 目标，也不能代表新测试集泛化。本开发说明不预写尚未完成的真实批次结果。
