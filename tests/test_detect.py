import numpy as np
import pytest
from scipy import stats

from src.detect import (
    aot,
    aot_power_bound,
    apply_permutation,
    block_permutation,
    cat,
    circular_serial_corr,
    mean_pairwise_spearman,
    partial_permutation,
    ww_moments,
)


def ar1(n, rho, rng):
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho**2) * rng.normal()
    return x


def test_circular_serial_corr_of_constant_step_sequence():
    # a linear ramp is almost perfectly serially correlated except for the wrap
    x = np.arange(1000, dtype=float)
    r = circular_serial_corr(x)
    assert 0.99 < r < 1.0


def test_circular_serial_corr_is_scale_and_shift_invariant():
    rng = np.random.default_rng(0)
    x = ar1(500, 0.8, rng)
    a = circular_serial_corr(x)
    b = circular_serial_corr(3.5 * x + 12.0)
    assert abs(a - b) < 1e-10


@pytest.mark.parametrize("n,kind", [(60, "normal"), (200, "lognormal"), (500, "chisquare")])
def test_ww_moments_match_monte_carlo(n, kind):
    """Reference test for the exact randomisation moments (Wald-Wolfowitz)."""
    rng = np.random.default_rng(42)
    x = {
        "normal": lambda: rng.normal(size=n),
        "lognormal": lambda: rng.lognormal(size=n),
        "chisquare": lambda: rng.chisquare(2, size=n),
    }[kind]()
    draws = np.array([circular_serial_corr(rng.permutation(x)) for _ in range(20000)])
    mean, var = ww_moments(x)
    assert abs(draws.mean() - mean) < 4 * draws.std(ddof=1) / np.sqrt(draws.size)
    assert abs(np.sqrt(var) / draws.std(ddof=1) - 1.0) < 0.05


def test_aot_certifies_serially_dependent_sequence():
    rng = np.random.default_rng(1)
    x = ar1(2000, 0.9, rng)
    res = aot(x)
    assert res["r"] > 0.8
    assert res["p"] < 1e-12


def test_aot_does_not_certify_a_permuted_sequence():
    rng = np.random.default_rng(2)
    x = ar1(2000, 0.9, rng)
    res = aot(apply_permutation(x, rng.permutation(x.size)))
    assert abs(res["r"]) < 0.1
    assert res["p"] > 0.01


def test_aot_false_certification_rate_matches_alpha():
    """The audit rule "certify iff p < alpha" mis-certifies at rate <= alpha."""
    rng = np.random.default_rng(3)
    base = ar1(400, 0.95, rng)
    ps = np.array([aot(rng.permutation(base))["p"] for _ in range(2000)])
    assert 0.03 < (ps < 0.05).mean() < 0.07
    assert stats.kstest(ps, "uniform").pvalue > 0.01


def test_aot_normal_approximation_agrees_with_permutation_p():
    rng = np.random.default_rng(4)
    x = rng.permutation(ar1(300, 0.9, rng))
    res = aot(x, n_perm=4000, rng=np.random.default_rng(5))
    assert abs(res["p"] - res["p_perm"]) < 0.05


def test_aot_power_bound_is_monotone_and_bounded():
    assert aot_power_bound(0.1, 100) < aot_power_bound(0.1, 10000)
    assert aot_power_bound(0.01, 50) < 0.5
    assert aot_power_bound(0.9, 5000) > 0.999


def test_block_permutation_keeps_blocks_intact():
    rng = np.random.default_rng(6)
    p = block_permutation(100, 10, rng)
    assert np.array_equal(np.sort(p), np.arange(100))
    for i in range(0, 100, 10):
        blk = p[i : i + 10]
        assert np.array_equal(blk, np.arange(blk[0], blk[0] + blk.size))


def test_block_shuffle_is_still_detected_when_blocks_are_smooth():
    rng = np.random.default_rng(7)
    x = ar1(4000, 0.99, rng)
    p = aot(apply_permutation(x, block_permutation(x.size, 64, rng)))["p"]
    assert p < 1e-6


def test_partial_permutation_touches_only_a_fraction():
    rng = np.random.default_rng(8)
    p = partial_permutation(1000, 0.1, rng)
    assert np.array_equal(np.sort(p), np.arange(1000))
    assert (p != np.arange(1000)).sum() <= 100


def test_cat_certifies_agreeing_arms():
    rng = np.random.default_rng(9)
    common = np.abs(ar1(800, 0.95, rng))
    mat = np.stack([common + 0.1 * rng.normal(size=800) for _ in range(4)])
    res = cat(mat, n_perm=500, rng=np.random.default_rng(10))
    assert res["rho_bar"] > 0.8
    assert res["p"] < 0.01


def test_cat_does_not_certify_independently_permuted_arms():
    rng = np.random.default_rng(11)
    common = np.abs(ar1(800, 0.95, rng))
    mat = np.stack([common + 0.1 * rng.normal(size=800) for _ in range(4)])
    defect = np.stack([apply_permutation(r, rng.permutation(800)) for r in mat])
    res = cat(defect, n_perm=500, rng=np.random.default_rng(12))
    assert abs(res["rho_bar"]) < 0.1
    assert res["p"] > 0.05


def test_cat_works_without_serial_structure_where_aot_cannot():
    """The i.i.d. regime: CAT still has power, AOT has none by construction."""
    rng = np.random.default_rng(13)
    difficulty = rng.gamma(2.0, 1.0, size=600)
    mat = np.stack([difficulty * np.exp(0.1 * rng.normal(size=600)) for _ in range(3)])
    assert aot(mat[0])["p"] > 0.01
    assert cat(mat, n_perm=500, rng=np.random.default_rng(14))["p"] < 0.01


def test_mean_pairwise_spearman_edge_cases():
    assert np.isnan(mean_pairwise_spearman(np.zeros((1, 10))))
    x = np.arange(10.0)
    assert abs(mean_pairwise_spearman(np.stack([x, x])) - 1.0) < 1e-12
    assert abs(mean_pairwise_spearman(np.stack([x, -x])) + 1.0) < 1e-12


def test_aot_returns_nan_on_degenerate_input():
    assert np.isnan(aot(np.zeros(3))["r"])
    assert np.isnan(aot(np.ones(100))["r"])


def test_ww_moments_match_brute_force_enumeration():
    """The randomisation moments are claimed to be exact, not asymptotic.

    Enumerate every permutation for small n and compare. This pins the variance
    formula: an error here would invalidate every z-score in the paper.
    """
    import itertools

    rng = np.random.default_rng(0)
    for n in (5, 6, 7, 8):
        x = rng.normal(size=n)
        vals = np.array([circular_serial_corr(np.array(p)) for p in itertools.permutations(x)])
        mean, var = ww_moments(x)
        assert mean == pytest.approx(vals.mean(), abs=1e-12)
        assert var == pytest.approx(vals.var(), rel=1e-10)


def test_ww_moments_undefined_for_constant_vector():
    """A constant vector must not be certifiable: variance is undefined, not tiny."""
    mean, var = ww_moments(np.full(64, 3.0))
    assert np.isnan(var)
    out = aot(np.full(64, 3.0))
    assert np.isnan(out["p"])


def test_ww_moments_undefined_for_near_constant_vector():
    x = np.full(64, 1.0)
    x[0] = 1.0 + 1e-13
    out = aot(x)
    assert not (out["p"] < 1e-3), "a near-constant vector must not be certified as ordered"
