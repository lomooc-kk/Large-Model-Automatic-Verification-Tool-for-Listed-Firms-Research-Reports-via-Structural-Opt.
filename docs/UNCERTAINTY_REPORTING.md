# 给模型判断加置信区间：做法、成本与边界

更新日期：2026-10-17。适用于 `factcheck/` 的模型候选判断与 `evals/` 的指标汇报。

## 1. 为什么单次判断没有置信区间

一次模型调用只产生一个结论，没有可估计的抽样变异。要对"这条判断有多稳"给出区间，
必须让同一份文档被独立判断 K 次，用"K 次里出现了几次"作为二项比例来估计。

这不是把模型的语气包装成置信度，也不是让模型自己报一个 0.9。模型自述的置信度未经校准，
不能进入自动确认路径。

## 2. 区间怎么算

对 K 次独立运行、v 次命中，取 Wilson 得分区间（z = 1.96）：

```
p̂ = v / K
中心 = (p̂ + z²/2K) / (1 + z²/K)
半宽 = z·√(p̂(1−p̂)/K + z²/4K²) / (1 + z²/K)
区间 = [中心 − 半宽, 中心 + 半宽]，截断到 [0, 1]
```

选 Wilson 而不是正态近似，是因为小 K 下正态近似会给出宽度为 0 或负数下界。

全票通过时的下界（K = K）：

| K | 3 | 5 | 10 | 15 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|
| 下界 | 0.439 | 0.566 | 0.723 | 0.796 | 0.806 | 0.839 | 0.887 |

这张表就是成本约束的核心：**K=3 全票通过，区间下界只有 0.44**，远不足以支撑自动确认。

## 3. 怎么用于判断

`yjcheck.uncertainty.decide()` 按区间给出三档：

| 条件 | 结论 | 含义 |
|---|---|---|
| 下界 ≥ 门槛 | `auto_confirm` | 即使这批运行偏乐观，也仍然达标 |
| 上界 < 0.5 | `drop` | 即使偏乐观，也大概率不是真问题 |
| 其余，或 K < 3 | `needs_review` | 交给人工 |

门槛默认 0.80，也可以复用 `auto_confirm.py` 的成本模型
（`uncertainty.cost_floor(error_type)`），让"重要性高"的错误类型保持更高门槛。

按 0.80 门槛，全票通过也需要 **K = 16**。也就是说：

- 想让区间真正参与自动确认，单篇要调用 16 次，成本是单次的 16 倍；
- 只做 K=3 或 K=5 时，区间的作用是**排序和展示**，不是放行；
- 更现实的分工是：确定性规则照旧负责自动确认，区间只用来给人工复核排队、给报告标注不确定性。

## 4. 怎么接进现有流程

新增两个模块，均不读金标、不改写候选、不自行升级状态：

| 模块 | 作用 |
|---|---|
| `factcheck/src/yjcheck/uncertainty.py` | Wilson 区间、`decide()`、`cost_floor()`、`minimum_trials_for()` |
| `factcheck/src/yjcheck/consensus.py` | 合并 K 次检测结果，按"类型 + 精确 span"计票并写入 `confidence` |

调用示例（伪代码）：

```python
from yjcheck.text_review import detect_text
from yjcheck.consensus import merge_reports
from yjcheck.uncertainty import decide

runs = [detect_text(text, document_id=doc_id, detector="hybrid", chat=chat) for _ in range(K)]
report = merge_reports(runs)

for candidate in report["errors"]:
    band = decide(Interval(**{k: candidate["confidence"][k] for k in ("votes","trials","p","lo","hi")}))
    # band 只用于排序与展示；是否确认仍由确定性规则决定
```

计票规则是保守的：类型相同且 span 完全一致才算同一条候选，同一轮里重复出现的候选只计一票。
这样票数只会被低估，不会被高估。

## 5. 成本

| K | 相对单次调用 | 用途 |
|---|---|---|
| 1 | 1× | 现状，无区间 |
| 3 | 3× | 只能看出"是否稳定出现"，不能放行 |
| 5 | 5× | 适合作为人工复核的排序信号 |
| 16 | 16× | 全票时下界才到 0.80，可支撑自动确认 |

按 0.034 元/篇估算，单篇 K=16 约 0.54 元；全量 442 篇约 240 元，超出当前 100 元授权，
因此必须在明确授权并更新账本后才能跑。

## 6. 边界

- 区间只描述**重复运行之间的一致性**，不证明业务正确，也不修正提示词偏差；
- 采样之间必须相互独立（同一提示、同一模型、不同采样），复用同一份回复不算新票；
- 每次运行的覆盖范围必须一致，否则票数不可比；
- 已曝光的 442 篇只能做回归，不能因为区间很窄就当成泛化能力；
- 报告里的比例一律写成"分子/分母 + 点估计 + 95% 区间"，分母包含失败与弃权。

## 7. 复现

```powershell
$env:PYTHONPATH='factcheck/src'
.venv\Scripts\python.exe -m unittest discover -s factcheck\tests -p "test_uncertainty.py" -v
.venv\Scripts\python.exe -m unittest discover -s factcheck\tests -p "test_consensus.py" -v
```

两组共 21 项测试，不调用模型、不读取金标。
