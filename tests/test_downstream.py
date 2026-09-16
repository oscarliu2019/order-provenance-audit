import numpy as np
import pytest
from scipy import stats

from src.downstream import (
    arm_selection,
    association,
    difficulty_regression,
    equivalence_test,
    purged_split,
)

COLS = ["f_std", "f_range"]


def test_purged_split_leaves_a_gap():
    tr, ho = purged_split(1000, purge=50)
    assert tr.max() < ho.min()
    assert ho.min() - tr.max() > 50


def test_association_recovers_a_monotone_relation():
    rng = np.random.default_rng(0)
    f = rng.gamma(2.0, 1.0, size=2000)
    y = f**1.5
    x = np.stack([f, rng.normal(size=2000)], axis=1)
    res = association(x, y, COLS)
    assert res["assoc_rho_fstd"] > 0.99
    assert res["decile_ratio_fstd"] > 5


def test_association_collapses_under_a_permuted_error_vector():
    rng = np.random.default_rng(1)
    f = rng.gamma(2.0, 1.0, size=4000)
    y = f**1.5
    x = np.stack([f, rng.normal(size=4000)], axis=1)
    res = association(x, y[rng.permutation(4000)], COLS)
    assert abs(res["assoc_rho_fstd"]) < 0.05
    assert abs(res["decile_ratio_fstd"] - 1.0) < 0.25


def test_defect_and_feature_shuffled_control_are_equal_in_distribution():
    """Proposition 3 in the paper, checked empirically.

    ``(X, y o sigma)`` and ``(X o pi, y)`` are the same random object as far as
    any row-permutation-equivariant procedure is concerned, so the canonical
    negative control has no power against the defect.
    """
    rng = np.random.default_rng(2)
    n = 800
    f = rng.gamma(2.0, 1.0, size=n)
    y = f**1.5
    x = np.stack([f, rng.normal(size=n)], axis=1)
    defect, control = [], []
    for _ in range(300):
        defect.append(association(x, y[rng.permutation(n)], COLS)["assoc_rho_fstd"])
        control.append(association(x[rng.permutation(n)], y, COLS)["assoc_rho_fstd"])
    d, c = np.array(defect), np.array(control)
    assert stats.ks_2samp(d, c).pvalue > 0.01
    assert abs(d.mean() - c.mean()) < 0.01


def test_equivalence_test_declares_identical_samples_equivalent():
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    res = equivalence_test(a, b, margin=0.5)
    assert res["equivalent_at_05"] == 1.0


def test_equivalence_test_rejects_a_real_shift():
    rng = np.random.default_rng(4)
    a = rng.normal(size=200) + 1.0
    b = rng.normal(size=200)
    res = equivalence_test(a, b, margin=0.1)
    assert res["equivalent_at_05"] == 0.0


def test_difficulty_regression_index_beats_defect_and_control():
    rng = np.random.default_rng(5)
    n, m = 1500, 600
    f = np.abs(np.cumsum(rng.normal(size=n))) + 1.0
    y = f * np.exp(0.2 * rng.normal(size=n))
    x = np.stack([f, rng.normal(size=n)], axis=1)
    ft = np.abs(np.cumsum(rng.normal(size=m))) + 1.0
    yt = ft * np.exp(0.2 * rng.normal(size=m))
    xt = np.stack([ft, rng.normal(size=m)], axis=1)
    res = difficulty_regression(x, y, xt, yt, rng.permutation(n), COLS, seed=0, purge=20)
    assert res["index"]["assoc_rho_fstd"] > 0.8
    assert abs(res["defect"]["assoc_rho_fstd"]) < 0.1
    assert abs(res["control"]["assoc_rho_fstd"]) < 0.1
    assert res["index"]["spearman_test"] > res["defect"]["spearman_test"]


def test_arm_selection_oracle_is_a_lower_bound():
    rng = np.random.default_rng(6)
    n, m, k = 900, 400, 3
    x = rng.normal(size=(n, 2))
    xt = rng.normal(size=(m, 2))
    val = np.abs(rng.normal(size=(k, n))) + 0.5
    test = np.abs(rng.normal(size=(k, m))) + 0.5
    perms = np.stack([rng.permutation(n) for _ in range(k)])
    res = arm_selection(x, val, xt, test, perms, ["a", "b", "c"], seed=0)
    for cond in res:
        assert res[cond]["oracle_mse"] <= res[cond]["best_fixed_mse"] + 1e-9
        assert res[cond]["oracle_headroom_pct"] >= -1e-9


def test_arm_selection_learns_a_planted_rule_only_with_the_correct_join():
    rng = np.random.default_rng(7)
    n, m = 1200, 600
    gate = rng.normal(size=n)
    gate_t = rng.normal(size=m)
    x = np.stack([gate, rng.normal(size=n)], axis=1)
    xt = np.stack([gate_t, rng.normal(size=m)], axis=1)
    val = np.stack([np.where(gate > 0, 0.5, 1.5), np.where(gate > 0, 1.5, 0.5)])
    test = np.stack([np.where(gate_t > 0, 0.5, 1.5), np.where(gate_t > 0, 1.5, 0.5)])
    perms = np.stack([rng.permutation(n) for _ in range(2)])
    res = arm_selection(x, val, xt, test, perms, ["a", "b"], seed=0)
    assert res["index"]["rel_gain_vs_best_fixed"] > 40
    assert res["defect"]["rel_gain_vs_best_fixed"] < 10
    assert res["control"]["rel_gain_vs_best_fixed"] < 10


def test_arm_selection_handles_a_degenerate_single_label_block():
    rng = np.random.default_rng(8)
    n, m = 300, 200
    x = rng.normal(size=(n, 2))
    xt = rng.normal(size=(m, 2))
    val = np.stack([np.full(n, 0.1), np.full(n, 1.0)])
    test = np.stack([np.full(m, 0.1), np.full(m, 1.0)])
    perms = np.stack([rng.permutation(n) for _ in range(2)])
    res = arm_selection(x, val, xt, test, perms, ["a", "b"], seed=0)
    assert np.isnan(res["index"]["oracle_match_acc"])
    assert abs(res["index"]["rel_gain_vs_best_fixed"]) < 1e-9


def test_association_is_invariant_to_a_monotone_transform_of_the_error():
    rng = np.random.default_rng(9)
    f = rng.gamma(2.0, 1.0, size=500)
    y = f**1.5
    x = np.stack([f, rng.normal(size=500)], axis=1)
    a = association(x, y, COLS)["assoc_rho_fstd"]
    b = association(x, np.log(y), COLS)["assoc_rho_fstd"]
    assert abs(a - b) < 1e-12


@pytest.mark.parametrize("n", [50, 200])
def test_association_z_and_p_are_consistent(n):
    rng = np.random.default_rng(10)
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    res = association(x, y, COLS)
    expected = 2 * stats.norm.sf(abs(res["assoc_z_fstd"]))
    assert abs(res["assoc_p_fstd"] - expected) < 1e-12
