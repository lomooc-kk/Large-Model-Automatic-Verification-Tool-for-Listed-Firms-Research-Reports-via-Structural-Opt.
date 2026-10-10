# 新数据与节点迭代

2026 年 10 月 9 日起，当前迭代采用“旧案例发现问题，新开发集改节点，新验证集比较效果”。旧 FinED、442 篇及历史配对样例保留为诊断材料，不进入新集训练示例、推理队列或效果分母。后来用户另行授权的旧错例回测由独立工作流运行，旧1137事件和成绩始终与新集分开。既有模型调用账本继续使用，历史费用不清零。

2026年10月10日，按用户要求进入新来源测试，不等待旧95%达标。来源库24份中20份为前瞻迭代诊断、4份为开发曝光（P03/P07/P08/P16）；目前P09–P15七篇、50页实际首次尝试，6篇完整执行，5篇同时候选定位完整。P10截断保留失败，P15一条原始候选因重复表格行拒绝。17份未推理来源中只有13份是前瞻角色，不能混算成17篇独立新测试。

本轮完成显式图表/表格边界、同文EPS单位对照、数值引文边界观察三项改动。固定首次六篇及P13 B独立案例零网络回放，消息与历史请求7/7一致；P12/P14各新增EPS待复核候选，P14目录错绑规则提示消失，P13 B两个遗漏百分号有精确warning。没有撤换原始模型输出或把warnings视为业务确认。软件1737项测试、819子测试通过，206文件前后哈希一致。

随后当前代码首次测试P15，1次真实调用、3raw→2needs_review+1rejected，0confirmed，费用0.133422元；估值时点/盈利口径、金额显示舍入及编辑性冗余适用仍暴露问题。预测前0暂定正例、8pending、15局部正常不构成completegold，不报整体F1。保留所有失败，不以同一篇连续调参代替新源验证。

新研报阶段累计21次付费调用、4.004662元，其中14次是历史配置开发对照；共享账本1193次、42.688694/100元。零网络回放不计新样本或费用。显式判定合同仍默认关闭；中文主路径本轮未调用Pi/OpenViking，没有更新模型权重。下一篇前瞻来源P18已完成来源准备，独立交叉审未完成、尚未推理；P16只保留原开发曝光准备角色。当前节点冻结，按新样本发现的共性机制安排后续比较前提、精度传播、表格联合定位工作。

随 PR 发布的结果见[当前交付进展](PR_PROGRESS_20261010.md)与[新来源汇总](validation/pi-openviking-20261010/new-source-summary.json)。详细本地证据保留于 `output/new-competition-pilot-20261010/progress.md`、`mechanism-fixes-20261010/summary.md` 和 `second-pilot-20261010/cross-review/p15-outcome.md`，这些实验目录不随 PR 发布。旧95%目标未完成，不阻塞新来源诊断；没有完整新业务金标，不能宣称最终盲测或完整十五类准确率。

已完成的首轮对照见 [新数据节点迭代结果](NEW_DATA_VALIDATION_20261009.md)。中文纠错候选进入后续复验；声明标签的精度、容差仍需审定，当前一致率不作为业务验收真值。前端原有旧开发示例自动加载也已移除。

## 当前数据

| 任务 | 开发 | 验证 | 封存候选 | 主要用途 |
|---|---:|---:|---:|---|
| CLFEC 中文段落纠错 | 186，含 22 条正常文本 | 82，含 10 条正常文本 | 0 | 必要局部纠正与过度修改 |
| FinanceBench 有证据声明核查 | 26，含 13 条正确声明 | 14，含 7 条正确声明 | 164 | 证据内数值一致性、弃权与检索补证 |

数据存放在 `data/active_suite_v1`，当前入口登记于 [`evals/current_dataset.json`](../evals/current_dataset.json)。输入与金标分文件，中性样本 ID 不携带正误标签，传给模型的白名单不含 ID、来源分组、划分或金标。FinanceBench 的 `instruction` 是问题语境，必须和 `claim` 一起提供，因为不少声明仅是对问题的一个数值回答。

FinanceBench 的开发、验证、封存分别包含 9、5、56 个原文档组；同一来源文档不跨组。CLFEC 按可见段落内容和正确原文聚合重复组件，267 个组件不跨开发与验证；上游没有完整原始研报关联，不能据此保证报告级独立。

与旧集排重只读取 8 个旧输入文件，共 997 个唯一旧文本，以 NFKC 和去空白后的全文哈希比较；没有读取旧金标或用旧集重新评分。新输入及相应正确原文与旧文本的归一化全文重合为 0，隔离记录当前为空。这不排除近似改写或来源关联缺失。

数据结构、哈希、分组、白名单及用途检查由 [`prepare_active_suite.py`](../evals/prepare_active_suite.py) 执行。可复现审计见 [`dataset_audit.ipynb`](../output/active-suite-20261009/dataset_audit.ipynb)，其中不展示任何样本正文或答案。

## 适合比赛的部分与仍缺的部分

这两条任务分别对应比赛系统的中文纠错与证据核验节点，保留上游任务口径。CLFEC 的四类段落纠错不强行改为 FinED 十五类；FinanceBench 的声明标签也不算中文研报的十五类检测成绩。

CLFEC 缺少随附事实来源，不能要求系统凭记忆把不确定金融事实自动确认为错误。FinanceBench 的错误声明是对公开答案数值作统一向上 8% 扰动，范围比实际研报错误窄。当前检索实验把该划分的公开证据节选组成候选库，按真实 token 分块，使用 OpenViking 查询后回读原文；没有把节选伪造为原始 PDF、页码或坐标，也不能据此宣称通过了财报全文检索验收。

比赛完整业务验收还需新增中文研报与原始公告或财报的配对材料，保留公司、报告期、指标、数值、单位、合并/母公司口径、页码和人工裁定。它应包含真实错误、正确但容易误报的表达以及证据不足三类情况。当前两轨可立即用于节点迭代，不能代替这个配对验收集。

## 旧问题如何驱动新节点

| 旧诊断发现 | 新节点或程序检查 | 用新数据观察什么 |
|---|---|---|
| 找到问题但引用改写、位置不完整 | 原文逐字引用及定位校验；保留前轮 span 规范化 | 有效原文引用、完整预测与纠正位置 |
| 不同主体、期间、单位或条款条件被比较 | 证据判读节点对齐主体、期间、指标、数值、单位、口径 | 声明精确率、召回、正确声明误报及弃权 |
| 局部字段缺失被当成整项金融业务缺失 | 先检查上下文；无来源不补造必要值 | 无依据补写、漏改与过改 |
| 标点或风格偏好产生误报 | 中文节点按事实、词语、语法、标点分别检查，只作必要局部修改 | 规范化编辑 F1、原段落完全正确率、正常文本误改率 |
| 缺证时误判或不断重复检索 | Pi 限定搜索、读取、重核顺序；OpenViking 只提供绑定来源 | 检索命中、实际读取、弃权、轮数和费用 |

具体候选代码为 [`claim_verification.py`](../factcheck/src/yjcheck/claim_verification.py) 与 [`correction_review.py`](../factcheck/src/yjcheck/correction_review.py)。两者的模型输出均是待复核建议，不自动升级为业务确认错误。现有中文 PDF 主流程保持既有接口，候选节点先独立比较，避免未验证的结构复杂度进入默认流程。

## 开发和验证顺序

1. 冻结数据版本，检查新旧排重、来源分组和输入/答案隔离。
2. 只在新开发集上比较基础节点与候选节点，检查运行失败和典型错例；开发答案可以用于监督示例。
3. 冻结节点源码、提示、模型、运行预算和检索条件，再对新验证划分完成整批推理。
4. 推理结束后，独立评分命令才读取金标；缺失、失败和弃权保留在分母。
5. 根据验证结果决定是否采用候选。若反复查看验证结果并调参，它就是迭代比较数据，不再称独立盲测。

`final_candidate_reserved` 不在运行器 CLI 可选角色中，函数入口也拒绝运行或入库。其历史曝光无法完全证明，因此只称封存候选，不能自动声明“从未见过”。FinanceBench QA/纠错任务若以后进入训练，必须继承相同来源文档的角色，防止从旁路拿到封存答案。

本轮实际改动属于节点、提示与运行配置优化。监督示例/SFT 文件的生成只表示训练材料准备好；只有真正更新模型权重且保存训练配置、checkpoint 和独立评估，才报告为完成模型训练。

## 运行入口

以下命令从仓库根目录执行。数据与历史运行产物不随源码 checkout 提供：`data/active_suite_v1`、外部预处理源、共享费用账本和历史运行目录都需要另行准备。`evals/current_dataset.json` 的 manifest 路径及 SHA 是本机实验快照指针，不表示克隆仓库后已经有可运行的评测集；本页指向 `data/`、`output/` 的明细是本地审计位置，不能视为随 PR 发布的结果附件。

准备输入必须满足已有转换合同，不能用任意原始下载目录代替：FinanceBench 外部根目录包含带 `upstream_pins` 的 `MANIFEST.json`、`splits/groups.jsonl` 及 `data/financebench_claim_verification/inputs.{dev,test}.jsonl`、对应 `gold`；CLFEC 根目录包含 `raw/SOURCE.json`、`raw/CLFEC.official.json` 及 `data/inputs.{mix,fec_only,lec_only,no_error}.jsonl`、对应 `gold`。JSONL 也接受同名 `.gz`。这些预处理材料的获取和上游版本应先确认；当前 builder 不负责下载或从任意上游格式转换。

已有 manifest 会绑定所有来源文件及旧输入排重文件的绝对路径和 SHA。迁移目录或机器后，不能只复制 manifest，也不要手改路径绕过核验：在新位置用同版本来源重新构建独立 suite，并检查行数、角色、分组和排重范围。旧输入仍只用于排重，不进入新样本或金标分母；若本机缺少原排重输入，不能沿用本页历史“997 个旧文本”的覆盖声明。下面显式传 `--suite`，不覆盖本机历史指针或冻结 manifest。

先将两个占位路径替换为已准备好的实际目录。builder 的无参数默认路径指向原工作区父目录中的审计材料，不适合新 checkout。`--output` 应为新的数据目录。

```powershell
$externalRoot = '<已准备的 external_benchmarks 目录>'
$clfecRoot = '<已准备的 clfec_finance 目录>'
$suiteRoot = 'data/active-suite-local'
.venv\Scripts\python.exe evals/prepare_active_suite.py --external-root $externalRoot --clfec-root $clfecRoot --output $suiteRoot
.venv\Scripts\python.exe evals/prepare_active_suite.py --validate-only --output $suiteRoot
```

准入通过后才安排推理。`run` 会调用模型 API，必须先配置模型、核验有效费率并恢复经授权的既有共享账本；本命令不授权重置历史费用或另建账本绕过累计预算。固定输出上限且不自动重试；输出目录必须是全新目录。

以下开发输出路径同时是当前 fewshot 验证器约定的排除计划位置。仅在该目录不存在的新 checkout 使用；若已有历史运行，保留原件，基础两臂可以另选新 `--out`，但目前 fewshot 运行器不能传入自定义排除计划，不能将另一个计划生成的示例银行直接拿来运行。

```powershell
$devOut = 'data/active-experiments-20261009/clfec-dev-r1'
.venv\Scripts\python.exe evals/run_active_suite.py run --suite $suiteRoot --track clfec --role development --limit 16 --out $devOut
.venv\Scripts\python.exe evals/run_active_suite.py score --suite $suiteRoot --out $devOut
```

FinanceBench 四组分别是基础证据判读、结构化证据判读、Pi 加给定证据、Pi 加 OpenViking 检索证据。检索组先运行 `prepare-retrieval`，并用 `--store` 绑定同一划分的节选库；基础组直接获得已给定的证据，因此检索组不必然更准。

中文候选增加 `structured_fewshot`，仅使用新开发集里词语、语法、标点、正常文本各一条示例。示例选择排除了开发试跑的目标、同组来源及归一化内容重合；示例银行先在独立进程中核对开发来源，推理进程再校验原始字节哈希。验证或封存答案不参与构建示例，选择到训练示例本身的评测会被拒绝。

```powershell
$examplesBank = 'data/active-runs/development-examples-local.json'
.venv\Scripts\python.exe -m evals.prepare_development_examples --suite $suiteRoot --exclude-plan "$devOut/plan.json" --out $examplesBank --sft-out data/active-runs/clfec-development-sft-local.jsonl
.venv\Scripts\python.exe evals/run_active_suite.py run --suite $suiteRoot --track clfec --role validation --workers 4 --arms direct,structured,structured_fewshot --examples-bank $examplesBank --out data/active-runs/clfec-validation-local
.venv\Scripts\python.exe evals/run_active_suite.py score --suite $suiteRoot --out data/active-runs/clfec-validation-local
```

这轮开发暴露了输出截断、引文空白差异、派生比率无法直接引用和重复检索。对应调整包括提高节点输出上限、仅对唯一空白差异定位回原文、将精确槽引文补入来源记录，以及用固定算术和 `Decimal` 复算有出处的数字。算术验证不证明模型选择的指标、期间和口径必然正确；求和/求差不会豁免单位未知，比率/百分比也必须有对应单位语境。Pi 最多搜索两次，同来源片段合并展示，相同来源组合不重复调用核验模型。

新运行会冻结相关源码、评分器、运行计划和输入哈希；完成后发现变化则拒绝按原实验口径评分。声明任务的主比较是全分母正确率与决策覆盖率，并列弃权及正常声明覆盖；不能只看 F1 或零误报。中文任务主看规范化编辑 F1 与完整段落匹配，正常文本缺输出时不将它算作正确无误报。

FinanceBench 全部 204 条的数值改写位置已通过机械审计，记录见 `output/active-suite-20261009/financebench_span_audit.json`。这证明本次改写位置没有触发转换器的潜在重复数字替换问题，不等于逐条验证了业务容差、上下界或近似表述；标签仍限定为合成数值偏差任务。

首轮开发结果位于 `data/active-experiments-20261009/`，不会与旧 442 篇分数合并。公开来源核验也在该目录，发送字段来自固定公开 GitHub 版本，未发送私有研报、配置文件或金标文件。
