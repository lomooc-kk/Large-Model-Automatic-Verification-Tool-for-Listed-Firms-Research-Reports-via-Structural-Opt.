# 显式检测判定合同

`text_review.detect_text(..., decision_contract=True)` 开启一个默认关闭的输出合同实验。原默认 `errors` 接口和消息保持不变。该实验源于新研报中“理由说明无冲突，内容却仍在errors”的问题；新增字段本身不保证模型业务判断正确，必须通过独立源审和正例保留对照决定是否采用。

## 输出与分流

模型返回唯一顶层 `decisions`，每项恰有 `verdict`、`error_type`、`spans`、`reason`。`error_type` 是十五类中所检查疑点的类型，不代表疑点已经成立。

| verdict | 处置 |
|---|---|
| `error_supported` | 进入既有错误候选锚定、规则证明绑定及融合流程；模型判定本身不升级为confirmed |
| `no_error` | 保留原决策和源文定位，不进入错误列表；不撤销规则或其他模型候选 |
| `insufficient_evidence` | 保留证据不足的具体理由和源文定位，不伪装成正常或已确认错误 |

只要求列出独立疑点，不逐行枚举正常正文。没有疑点时可返回 `decisions: []`；这不是全篇无错证明。

所有决策均存入 `raw_decisions`，保留原序号。例如第三项才是positive时，候选仍为 `model:0:2`。`raw_candidates` 仅包含positive的完整原决策，含verdict；它们继续通过 `model_candidate_representation` 或 `rejected_candidates` 追踪。`model_decision_dispositions` 为全部判定提供唯一的初始去向，非positive也必须锚定原文，定位失败会使候选质量不完整。

`model_decision_dispositions` 描述检测阶段初始分流。若另外开启冗余复核并撤回positive，应继续沿 `candidate_ref` 查询 `model_candidate_representation` 及撤回审计，而不是把初始 `error_candidate` 当作最终错误。首轮真实对照关闭冗余复核。

## 失败与证据边界

- 严格校验JSON、封闭字段、枚举、非空理由及最多400个Unicode字符的理由长度。不会截短文本或补写理由。
- 重复JSON键、损坏JSON、非标准NaN/Infinity、完全重复的决策对象均使整批失败。不会用有效前缀掩盖错误尾项。
- 相同理由但不同位置、类别、判定的对象不会按理由去重；不会按“无冲突”等关键词改写verdict。
- 非positive引文也要经过同源锚定。不存在或有歧义的引文保留拒绝原因，并将候选质量标为不完整。
- `finish_reason=length` 或不完整响应始终失败，即使可见片段恰好能解析成JSON。
- 解析失败的原始最终响应及SHA记录在 `decision_protocol_failures`。运行失败与协议失败分别记录；缓存/运行器还保存原API收据。不会读取或保存推理正文。

协议合法、引文有效、业务正确是三个不同层面。`error_supported` 与reason仍可能语义矛盾；当前代码不会用关键词假装解决该问题，独立业务审查必须覆盖所有三种verdict。

## 开发示例与默认行为

既有开发示例先经过原完整标注、来源分组及目标隔离检查，再将其已有error标签转换为 `error_supported`。不生成新标签、不修改示例原文；缺少理由或不满足新合同的示例会在调用前失败。首轮对照使用空示例。

五处请求构造均传递开关，包括全文预算估计、分窗、跨段比较和真实调用。新合同开销参与原输入预算，不会为满足预算悄悄退回旧合同。

实现位于 `factcheck/src/yjcheck/candidate_decisions.py` 和 `text_review.py`；纯模块不读文件、不看答案、不调用模型。默认关闭时，三篇实际历史请求的消息及SHA逐字匹配。1645项软件测试、797个子测试通过；这些结果不是业务准确率。

首轮固定开发对照协议见 `output/new-competition-pilot-20261010/decision-contract-20261010/protocol.md`。P09/P12/P13均为已见来源，不称盲测；六个新请求在调用前共同冻结，原共享100元账本、无重试，不复用历史回答。

## 首次真实对照结果

六次均完整返回。原errors合同三篇均执行及引文完整，8条原始候选全部进入待复核列表；新合同三篇中两篇执行及引文完整，11条positive中10条进入待复核列表、1条因重复引文无法唯一定位被拒绝。新合同实际没有输出no_error或insufficient_evidence，因此不能宣称已经验证正常解释分流的收益。

P12的资产关系两臂均保留，EPS(X)单位问题两臂均漏。新合同仍出现来源不可读被判为缺数、季度与全年强比，并在P13把两个2027E数字的紧随百分号截在引文外后声称缺单位。原文精确子串只证明引用存在，不证明引文边界和推理正确。

因此合同继续默认关闭。两臂单次采样不足以证明稳定差异；不根据候选数减少或合法JSON给出业务准确率。六次费用1.125070元，完整证据见[开发对照结果](../output/new-competition-pilot-20261010/decision-contract-20261010/progress.md)。下一轮优先在新来源中验证期间与指标口径绑定、引文数值及单位完整性。
