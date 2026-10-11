"""Cost-optimized auto-confirm decision for the human-review triage.

Current behavior only auto-confirms candidates that a deterministic rule proves
exactly; everything else becomes ``needs_review``. That is maximally safe but
very expensive: almost every candidate pays the human-review cost.

This module relaxes the bar with an explicit cost model. For each model candidate
we compare two routes:

* human route:    cost = ``human_review_cost`` (fixed, high)
* AI auto-confirm: cost = ``ai_decision_cost`` + false-positive risk
                  = ``ai_decision_cost`` + ``(1 - confidence) * importance * fp_cost_per_importance``

``confidence`` is the probability the candidate is a true error. ``importance``
is how consequential a wrong confirmation (a false positive) would be. When the
AI route is strictly cheaper we auto-confirm, which lowers expected total cost
and encourages the model to decide on its own instead of deferring to a human.

Consequence: low-importance errors (redundancy, ambiguous wording, formatting)
and higher-confidence candidates are auto-confirmed; high-importance errors
(calculation, numeric inconsistency, contradiction) keep a higher confidence bar.
"""
from __future__ import annotations

from dataclasses import dataclass

# Importance of each error type: 1 = least consequential, 5 = most consequential.
# A false positive on a high-importance type is expensive, so those types are held
# to a higher confidence threshold before auto-confirm.
ERROR_IMPORTANCE: dict[str, int] = {
    "计算错误": 5,
    "数值不一致错误": 5,
    "时间矛盾": 4,
    "数值单位错误": 4,
    "语义逻辑矛盾": 4,
    "法规引用错误": 4,
    "金融要素缺失": 3,
    "数值缺失": 3,
    "属性值缺失错误": 3,
    "术语误用": 3,
    "不一致条款": 3,
    "时间信息非法": 2,
    "格式错误": 2,
    "冗余语句": 1,
    "模糊语言": 1,
}

_DEFAULT_IMPORTANCE = 3


@dataclass(frozen=True)
class AutoConfirmConfig:
    """Cost parameters for the auto-confirm decision (abstract units, not money).

    ``human_review_cost`` is high and ``ai_decision_cost`` is low by design, which
    biases the decision toward AI autonomy. ``fp_cost_per_importance`` scales the
    false-positive penalty by error importance.
    """
    human_review_cost: float = 5.5
    ai_decision_cost: float = 0.5
    fp_cost_per_importance: float = 10.0
    model_confidence: float = 0.65


DEFAULT_CONFIG = AutoConfirmConfig()


def confidence_threshold(importance: int, config: AutoConfirmConfig = DEFAULT_CONFIG) -> float:
    """Minimum confidence required to auto-confirm a candidate of ``importance``.

    Derived by solving ``ai_decision_cost + (1 - c) * importance * fp_cost
    < human_review_cost`` for ``c``. Returns 0 when human review is never cheaper
    (i.e. every candidate is auto-confirmed).
    """
    margin = config.human_review_cost - config.ai_decision_cost
    if margin <= 0:
        return 0.0
    return max(0.0, 1.0 - margin / (importance * config.fp_cost_per_importance))


def should_auto_confirm(error_type: str, *, confidence: float | None = None,
                        config: AutoConfirmConfig = DEFAULT_CONFIG) -> bool:
    """Decide whether a model candidate of ``error_type`` should be auto-confirmed."""
    importance = ERROR_IMPORTANCE.get(error_type, _DEFAULT_IMPORTANCE)
    conf = config.model_confidence if confidence is None else confidence
    conf = max(0.0, min(1.0, float(conf)))
    return conf > confidence_threshold(importance, config)
