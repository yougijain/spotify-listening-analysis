"""Tests for the inferential statistics.

The CMH expectations below were cross-checked once against
``statsmodels.stats.contingency_tables.StratifiedTable`` and agreed to six
decimal places on the statistic, p-value, common odds ratio and the
Robins-Breslow-Greenland interval. They are pinned here so the check survives
without adding statsmodels as a dependency.
"""

from __future__ import annotations

import math

import pytest

from src.stats import (
    Stratum,
    cochran_mantel_haenszel,
    two_proportion_z,
    wilson_interval,
)

# --- Wilson ----------------------------------------------------------------

def test_wilson_brackets_the_point_estimate():
    iv = wilson_interval(30, 100)
    assert iv.low < iv.point < iv.high
    assert iv.point == pytest.approx(0.30)


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    for successes, trials in [(0, 1), (1, 1), (0, 5), (5, 5), (0, 1000), (1000, 1000)]:
        iv = wilson_interval(successes, trials)
        assert 0.0 <= iv.low <= iv.high <= 1.0


def test_wilson_does_not_collapse_at_zero_successes():
    # The whole point: a never-skipped track is not proven un-skippable.
    assert wilson_interval(0, 3).high > 0.5


def test_wilson_tightens_as_evidence_accumulates():
    widths = [wilson_interval(n // 2, n).high - wilson_interval(n // 2, n).low
              for n in (10, 100, 1000, 10000)]
    assert widths == sorted(widths, reverse=True)


def test_wilson_shrinks_optimistic_small_samples_the_most():
    # Two tracks both at a 0% skip rate; the well-evidenced one ranks higher on
    # the lower bound of its hold rate. This is what stops n=2 flukes topping
    # the crate ranking.
    thin = wilson_interval(2, 2)
    thick = wilson_interval(200, 200)
    assert thin.point == thick.point == 1.0
    assert thin.low < thick.low


def test_wilson_with_no_trials_is_maximally_uninformative():
    iv = wilson_interval(0, 0)
    assert (iv.low, iv.high) == (0.0, 1.0)


def test_wilson_is_symmetric_under_relabelling():
    a = wilson_interval(30, 100)
    b = wilson_interval(70, 100)
    assert a.low == pytest.approx(1 - b.high)
    assert a.high == pytest.approx(1 - b.low)


@pytest.mark.parametrize("successes,trials", [(-1, 10), (5, -1), (11, 10)])
def test_wilson_rejects_impossible_counts(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


# --- two-proportion z ------------------------------------------------------

def test_two_proportion_z_reproduces_the_shuffle_result():
    # Locked to the numbers this project published before the refactor.
    r = two_proportion_z(1248, 3578, 512, 4400)
    assert r.z == pytest.approx(24.9, abs=0.05)
    assert r.p_one_sided == pytest.approx(3.65e-137, rel=1e-2)
    assert r.cohens_h == pytest.approx(0.567, abs=0.001)


def test_two_proportion_z_is_signed_by_direction():
    assert two_proportion_z(50, 100, 10, 100).z > 0
    assert two_proportion_z(10, 100, 50, 100).z < 0


def test_identical_rates_give_no_evidence():
    r = two_proportion_z(50, 100, 50, 100)
    assert r.z == pytest.approx(0.0)
    assert r.p_one_sided == pytest.approx(0.5)
    assert r.cohens_h == pytest.approx(0.0)


def test_effect_size_is_independent_of_sample_size():
    small = two_proportion_z(30, 100, 10, 100)
    large = two_proportion_z(3000, 10000, 1000, 10000)
    assert small.cohens_h == pytest.approx(large.cohens_h)
    assert large.z > small.z          # only the evidence grows


def test_two_proportion_z_needs_trials_in_both_arms():
    with pytest.raises(ValueError):
        two_proportion_z(0, 0, 5, 10)


# --- CMH -------------------------------------------------------------------

TWO_STRATA = [
    Stratum("shuffle", 300, 1000, 200, 1000),
    Stratum("intentional", 120, 1000, 80, 1000),
]
THREE_STRATA = [
    Stratum("s1", 10, 50, 3, 40),
    Stratum("s2", 25, 80, 30, 150),
    Stratum("s3", 5, 20, 8, 60),
]


def test_cmh_matches_the_reference_implementation_two_strata():
    r = cochran_mantel_haenszel(TWO_STRATA)
    assert r.statistic == pytest.approx(34.795206, abs=1e-6)
    assert r.p_value == pytest.approx(3.662742e-09, rel=1e-6)
    assert r.odds_ratio == pytest.approx(1.665399, abs=1e-6)
    assert (r.or_low, r.or_high) == (pytest.approx(1.4061, abs=1e-4),
                                     pytest.approx(1.9725, abs=1e-4))


def test_cmh_matches_the_reference_implementation_three_strata():
    r = cochran_mantel_haenszel(THREE_STRATA)
    assert r.statistic == pytest.approx(6.812686, abs=1e-6)
    assert r.p_value == pytest.approx(9.051252e-03, rel=1e-6)
    assert r.odds_ratio == pytest.approx(2.038981, abs=1e-6)


def test_cmh_matches_the_reference_implementation_single_stratum():
    r = cochran_mantel_haenszel([Stratum("only", 7, 12, 2, 15)])
    assert r.statistic == pytest.approx(4.0625, abs=1e-6)
    assert r.odds_ratio == pytest.approx(9.1, abs=1e-6)


def test_confidence_interval_brackets_the_odds_ratio():
    r = cochran_mantel_haenszel(TWO_STRATA)
    assert r.or_low < r.odds_ratio < r.or_high


def test_cmh_survives_simpsons_paradox():
    """The reason this test exists rather than a pooled comparison.

    Both strata show clashes skipping *less*, but the clash transitions are
    concentrated in the high-skip stratum. Pooling reverses the sign; CMH keeps
    it, which is exactly the confound shuffle introduces in the real data.
    """
    strata = [
        # High-skip stratum, almost all the clashes live here.
        Stratum("shuffle", 380, 1000, 45, 100),
        # Low-skip stratum, almost all the harmonic transitions live here.
        Stratum("intentional", 4, 100, 90, 1000),
    ]
    r = cochran_mantel_haenszel(strata)
    pooled_exposed = (380 + 4) / (1000 + 100)
    pooled_control = (45 + 90) / (100 + 1000)
    assert pooled_exposed > pooled_control       # pooling says clashes skip more
    assert r.odds_ratio < 1.0                    # stratified says the opposite
    for s in strata:
        assert s.rate_exposed < s.rate_control   # ...matching every stratum


def test_no_association_gives_an_odds_ratio_of_one():
    r = cochran_mantel_haenszel([Stratum("a", 100, 500, 100, 500)])
    assert r.odds_ratio == pytest.approx(1.0)
    assert not r.significant


def test_empty_arms_are_dropped_not_propagated_as_nan():
    r = cochran_mantel_haenszel([
        Stratum("usable", 300, 1000, 200, 1000),
        Stratum("no clashes here", 0, 0, 50, 300),
    ])
    assert len(r.strata) == 1
    assert not math.isnan(r.statistic)
    assert not math.isnan(r.odds_ratio)


def test_cmh_needs_at_least_one_usable_stratum():
    with pytest.raises(ValueError, match="no stratum"):
        cochran_mantel_haenszel([Stratum("empty", 0, 0, 0, 0)])


def test_a_zero_cell_in_one_stratum_still_pools_to_a_finite_answer():
    # The realistic case: one thin stratum is fully separated, the rest are not.
    r = cochran_mantel_haenszel([
        Stratum("thin and separated", 6, 6, 0, 6),
        Stratum("ordinary", 300, 1000, 200, 1000),
    ])
    assert math.isfinite(r.odds_ratio)
    assert r.odds_ratio > 1.0


def test_complete_separation_reports_an_infinite_odds_ratio():
    # Every exposed transition skipped, no control one did. The odds ratio is
    # genuinely unbounded; reporting NaN would read as "computation failed".
    r = cochran_mantel_haenszel([Stratum("perfect split", 10, 10, 0, 10)])
    assert math.isinf(r.odds_ratio)
    assert math.isfinite(r.statistic) and r.significant


def test_stratum_rates_and_odds_ratio():
    s = Stratum("x", 300, 1000, 200, 1000)
    assert s.rate_exposed == pytest.approx(0.30)
    assert s.rate_control == pytest.approx(0.20)
    # (300/700) / (200/800)
    assert s.odds_ratio == pytest.approx((300 * 800) / (700 * 200))


def test_cmh_also_reports_the_pooled_comparison_for_contrast():
    r = cochran_mantel_haenszel(TWO_STRATA)
    assert r.pooled.n_a == 2000 and r.pooled.n_b == 2000
    assert r.pooled.rate_a == pytest.approx(420 / 2000)
