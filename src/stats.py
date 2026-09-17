"""Inferential statistics used by the analysis, kept separate from the pipeline.

Three things live here, in increasing order of how much they matter:

* :func:`wilson_interval` — a confidence interval for a proportion that behaves
  sensibly at small n. Used for per-track skip rates, where the naive rate is
  actively misleading: a track played twice and never skipped has a 0% skip
  rate, and sorting the crate on that number puts the least-evidenced tracks on
  top. The Wilson lower bound does the shrinking for you.

* :func:`two_proportion_z` — the shuffle-vs-intentional test (SPEC §7.3).

* :func:`cochran_mantel_haenszel` — the harmonic test (SPEC §8.5), which needs
  more care than a pooled comparison. Shuffle raises the skip rate *and*
  produces more clashing transitions, so it is a common cause of both variables:
  pooling everything would credit shuffle's skips to bad harmony. CMH tests the
  association within each shuffle stratum and combines the strata, which is the
  cheapest honest answer short of a regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy import stats

# 95% two-sided normal quantile, the default for the Wilson interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the textbook normal approximation because it stays inside
    [0, 1] and does not collapse to a zero-width interval when the observed
    proportion is 0 or 1 — exactly the cases that dominate a long-tailed crate.

    ``trials == 0`` returns the fully uninformative [0, 1].
    """
    if trials < 0 or successes < 0:
        raise ValueError("counts must be non-negative")
    if successes > trials:
        raise ValueError(f"successes ({successes}) exceeds trials ({trials})")
    if trials == 0:
        return Interval(0.0, 0.0, 1.0)

    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = p + z2 / (2.0 * trials)
    margin = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return Interval(p, max(0.0, (centre - margin) / denom),
                    min(1.0, (centre + margin) / denom))


@dataclass(frozen=True)
class ZTest:
    rate_a: float
    rate_b: float
    difference: float
    z: float
    p_one_sided: float
    cohens_h: float
    n_a: int
    n_b: int
    x_a: int
    x_b: int


def two_proportion_z(x_a: int, n_a: int, x_b: int, n_b: int) -> ZTest:
    """Pooled two-proportion z-test of H1: rate_a > rate_b.

    Also returns Cohen's h, because a z of 25 on 8,000 plays says the effect is
    real, not that it is large — the two questions need separate numbers.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both groups need at least one trial")

    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z = 0.0 if se == 0 else (p_a - p_b) / se
    # Cohen's h: the arcsine-transformed difference, which unlike a raw
    # percentage-point gap is comparable across baseline rates.
    h = 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b))
    return ZTest(
        rate_a=p_a, rate_b=p_b, difference=p_a - p_b,
        z=z, p_one_sided=float(stats.norm.sf(z)), cohens_h=h,
        n_a=n_a, n_b=n_b, x_a=x_a, x_b=x_b,
    )


@dataclass(frozen=True)
class Stratum:
    """One 2x2 table: exposure (clash vs harmonic) by outcome (skip vs hold)."""

    label: str
    x_exposed: int      # skips among the exposed (clashing) transitions
    n_exposed: int
    x_control: int      # skips among the control (harmonic) transitions
    n_control: int

    @property
    def rate_exposed(self) -> float:
        return self.x_exposed / self.n_exposed if self.n_exposed else float("nan")

    @property
    def rate_control(self) -> float:
        return self.x_control / self.n_control if self.n_control else float("nan")

    @property
    def odds_ratio(self) -> float:
        """Within-stratum odds ratio, Haldane-corrected so a zero cell is usable."""
        a, b = self.x_exposed, self.n_exposed - self.x_exposed
        c, d = self.x_control, self.n_control - self.x_control
        if 0 in (a, b, c, d):
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        return (a * d) / (b * c)


@dataclass(frozen=True)
class CMHResult:
    statistic: float
    p_value: float
    odds_ratio: float
    or_low: float
    or_high: float
    strata: tuple
    pooled: ZTest

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def crude_odds_ratio(self) -> float:
        """The odds ratio you get by ignoring the strata and pooling.

        Reported next to the adjusted one so the confound has a size rather
        than just a mention: the gap between the two IS what stratifying bought.
        """
        a = sum(s.x_exposed for s in self.strata)
        b = sum(s.n_exposed - s.x_exposed for s in self.strata)
        c = sum(s.x_control for s in self.strata)
        d = sum(s.n_control - s.x_control for s in self.strata)
        if 0 in (b, c):
            return math.inf if a and d else float("nan")
        return (a * d) / (b * c)

    @property
    def confounding_pct(self) -> float:
        """How far the crude estimate is off, as a percentage of the adjusted.

        Over ~10% is the conventional threshold for calling a variable a
        confounder worth adjusting for.
        """
        if not math.isfinite(self.crude_odds_ratio) or self.odds_ratio in (0, math.inf):
            return float("nan")
        return 100.0 * (self.crude_odds_ratio - self.odds_ratio) / self.odds_ratio


def cochran_mantel_haenszel(strata: Sequence[Stratum]) -> CMHResult:
    """Test exposure-outcome association, holding the strata fixed.

    Returns the CMH chi-square (1 df, continuity-corrected), its p-value, and
    the Mantel-Haenszel common odds ratio with a Robins-Breslow-Greenland 95%
    interval. An odds ratio above 1 means the exposed group skips more.

    A zero cell in one stratum among several is absorbed by the pooling and
    still yields a finite odds ratio. Complete separation across *every*
    stratum returns ``inf``, because that is the honest answer.

    Strata with an empty arm contribute nothing and are dropped, so a stratum
    that happens to contain no clashing transitions cannot produce a NaN.
    """
    usable = [s for s in strata if s.n_exposed > 0 and s.n_control > 0
              and (s.n_exposed + s.n_control) > 1]
    if not usable:
        raise ValueError("no stratum has data in both arms")

    sum_a = sum_e = sum_v = 0.0
    num = den = 0.0
    # Robins-Breslow-Greenland variance accumulators for the log odds ratio.
    rbg_pr = rbg_pspr = rbg_sqs = 0.0

    for s in usable:
        a, b = s.x_exposed, s.n_exposed - s.x_exposed
        c, d = s.x_control, s.n_control - s.x_control
        n = float(a + b + c + d)
        r1, r2 = a + b, c + d          # exposed / control totals
        c1, c2 = a + c, b + d          # skip / hold totals

        sum_a += a
        sum_e += r1 * c1 / n
        sum_v += (r1 * r2 * c1 * c2) / (n * n * (n - 1.0))

        r, s_ = (a * d) / n, (b * c) / n
        num += r
        den += s_
        p_, q_ = (a + d) / n, (b + c) / n
        rbg_pr += p_ * r
        rbg_pspr += p_ * s_ + q_ * r
        rbg_sqs += q_ * s_

    # Continuity-corrected CMH statistic; clamped at 0 when |sum_a - sum_e| < 0.5.
    if sum_v <= 0:
        statistic = 0.0
    else:
        statistic = max(0.0, abs(sum_a - sum_e) - 0.5) ** 2 / sum_v
    p_value = float(stats.chi2.sf(statistic, df=1))

    if den == 0 and num > 0:
        # Complete separation: every exposed transition skipped and no control
        # one did (or vice versa). The common odds ratio really is unbounded —
        # report that rather than a NaN that reads like "computation failed" or
        # a Haldane-corrected number that reads like real evidence.
        odds_ratio, or_low, or_high = math.inf, float("nan"), math.inf
    elif den > 0 and num > 0:
        odds_ratio = num / den
        var_log = (rbg_pr / (2 * num * num)
                   + rbg_pspr / (2 * num * den)
                   + rbg_sqs / (2 * den * den))
        se = math.sqrt(var_log)
        or_low = odds_ratio * math.exp(-Z_95 * se)
        or_high = odds_ratio * math.exp(Z_95 * se)
    else:
        odds_ratio, or_low, or_high = float("nan"), float("nan"), float("nan")

    pooled = two_proportion_z(
        sum(s.x_exposed for s in usable), sum(s.n_exposed for s in usable),
        sum(s.x_control for s in usable), sum(s.n_control for s in usable),
    )
    return CMHResult(statistic, p_value, odds_ratio, or_low, or_high,
                     tuple(usable), pooled)
