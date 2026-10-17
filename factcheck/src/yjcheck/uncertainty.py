"""Wilson score intervals for repeated model judgments.

A single model call carries no estimable uncertainty. When the same document is
judged K times independently, the share of runs that produced a candidate is a
binomial proportion, and a Wilson score interval gives a defensible range for
how stable that candidate is. The interval describes sampling variation across
repeated runs only; it says nothing about whether the judgment is business
correct, and it does not correct a biased prompt or a biased sample.

Design constraints kept deliberately:

* No gold answers, no document IDs, no branch on sample identity.
* The interval never promotes a candidate by itself. ``decide()`` only reports
  what the interval supports; the caller keeps its existing deterministic
  confirmation path and its human-review queue.
* Trials below ``MIN_TRIALS`` are reported as ``needs_review``: with K=1 or K=2
  the interval is too wide to distinguish agreement from luck.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

Z_95 = 1.959963984540054
MIN_TRIALS = 3


@dataclass(frozen=True)
class Interval:
    """One estimated proportion with its two-sided Wilson interval."""

    votes: int
    trials: int
    p: float
    lo: float
    hi: float
    method: str = "wilson"

    def as_dict(self) -> dict:
        return {
            "votes": self.votes,
            "trials": self.trials,
            "p": round(self.p, 4),
            "lo": round(self.lo, 4),
            "hi": round(self.hi, 4),
            "method": self.method,
        }


def wilson_interval(votes: int, trials: int, z: float = Z_95) -> Interval:
    """Two-sided Wilson score interval for ``votes`` out of ``trials``.

    Unlike the normal approximation, this stays inside [0, 1] and does not
    collapse to zero width when every observation agrees or no observation
    does. That matters here: a candidate seen in 3 of 3 runs is *not* a
    certainty, and the interval should say so.
    """
    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if type(votes) is not int or not 0 <= votes <= trials:
        raise ValueError("votes must be an integer between 0 and trials")
    if not 0 < z < 10:
        raise ValueError("z must be a small positive number")

    p = votes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # Clamp the algorithmic noise at the boundaries so callers can compare
    # ``lo <= p`` without hitting 5.5e-17 style artefacts.
    if votes == 0:
        lo = 0.0
    if votes == trials:
        hi = 1.0
    return Interval(
        votes=votes,
        trials=trials,
        p=p,
        lo=lo,
        hi=hi,
    )


def unanimous_interval(trials: int, z: float = Z_95) -> Interval:
    """Interval for the best case, ``trials`` votes out of ``trials``."""
    return wilson_interval(trials, trials, z)


def minimum_trials_for(floor: float, z: float = Z_95, limit: int = 200) -> int:
    """Smallest K whose unanimous lower bound reaches ``floor``.

    This is the honest answer to "how many samples do I need before the
    interval supports auto-confirmation": with z=1.96 and floor=0.80 it is 16.
    """
    if not 0 < floor < 1:
        raise ValueError("floor must be between 0 and 1")
    for trials in range(MIN_TRIALS, limit + 1):
        if unanimous_interval(trials, z).lo >= floor:
            return trials
    raise ValueError(f"no K below {limit} reaches floor={floor}")


def decide(interval: Interval, *, floor: float = 0.80,
           drop_below: float = 0.50) -> str:
    """Map an interval to ``auto_confirm`` / ``needs_review`` / ``drop``.

    ``auto_confirm`` requires the *lower* bound to clear ``floor``: the rule
    stays true even if this sample of runs happened to be lucky. ``drop``
    requires the *upper* bound to sit below ``drop_below``: even the optimistic
    reading says the candidate is unlikely to be real.

    Anything else stays for human review. The caller remains responsible for
    its own deterministic checks; this function never confirms on its own.
    """
    if not 0 <= drop_below < floor <= 1:
        raise ValueError("require 0 <= drop_below < floor <= 1")
    if interval.trials < MIN_TRIALS:
        return "needs_review"
    if interval.lo >= floor:
        return "auto_confirm"
    if interval.hi < drop_below:
        return "drop"
    return "needs_review"


def cost_floor(error_type: str) -> float:
    """Threshold implied by the existing cost model in ``auto_confirm``.

    Reuses ``confidence_threshold`` so the interval policy and the human-review
    triage stay on the same footing instead of introducing a second scale.
    """
    from .auto_confirm import ERROR_IMPORTANCE, confidence_threshold

    importance = ERROR_IMPORTANCE.get(error_type, 3)
    return confidence_threshold(importance)
