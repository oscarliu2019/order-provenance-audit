"""Downstream analyses that consume a per-window error vector by positional join.

Two tasks, deliberately chosen to sit on opposite sides of the null:

T1 *difficulty regression* -- predict a window's error from the window's own
   input features. This has a large, real effect: volatile windows are harder.
   It is the positive control of the study.

T2 *per-window arm selection* -- pick, per window, which plugin to apply. Prior
   work reports this as not learnable, i.e. its honest answer is a null.

Each task is run under three conditions:

``index``    the validation error vector is joined in index order (correct);
``defect``   the validation error vector is in loader order, i.e. permuted, and
             joined positionally anyway;
``control``  the canonical negative control: correct errors, but the feature
             rows are shuffled.

The comparison of ``defect`` with ``control`` is the substance of the paper: if
they coincide, the negative control has no power to reveal the defect, and a
pipeline whose headline result is a null cannot be validated by it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.features import FEATURE_COLUMNS

CONDITIONS = ("index", "defect", "control")


def _fit_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=200,
        learning_rate=0.08,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _fit_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.08,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def purged_split(n: int, purge: int, frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/holdout split with a purge gap between the two parts.

    Adjacent windows overlap by ``seq_len + pred_len - 1`` points, so a random
    split would leak holdout windows into training through their neighbours.
    """
    cut = int(n * frac)
    train = np.arange(0, max(cut - purge, 1))
    hold = np.arange(min(cut + purge, n - 1), n)
    return train, hold


def association(
    x: np.ndarray, y: np.ndarray, columns: list[str], y_raw: np.ndarray | None = None
) -> dict[str, float]:
    """Descriptive per-window association statistics of a joined table.

    This is the form of per-sample analysis that appears in papers: correlate a
    per-sample loss with a per-sample covariate, or bin the loss by covariate
    decile. No model is fitted, so nothing here depends on a learner's
    inductive bias -- only on whether row ``i`` of the feature table and entry
    ``i`` of the error vector describe the same window.
    """
    n = len(y)
    rho = np.array([stats.spearmanr(x[:, j], y).statistic for j in range(x.shape[1])])
    best = int(np.nanargmax(np.abs(rho)))
    fstd = columns.index("f_std")
    raw = y if y_raw is None else np.asarray(y_raw, dtype=np.float64)
    q = np.quantile(x[:, fstd], [0.1, 0.9])
    low = raw[x[:, fstd] <= q[0]]
    high = raw[x[:, fstd] >= q[1]]
    out = {
        "assoc_rho_best": float(rho[best]),
        "assoc_rho_best_feature": columns[best],
        "assoc_rho_fstd": float(rho[fstd]),
        "assoc_abs_rho_mean": float(np.nanmean(np.abs(rho))),
        "assoc_z_fstd": float(rho[fstd] * np.sqrt(max(n - 1, 1))),
        "assoc_p_fstd": float(2 * stats.norm.sf(abs(rho[fstd]) * np.sqrt(max(n - 1, 1)))),
        "decile_ratio_fstd": float(high.mean() / low.mean()) if low.size and low.mean() > 0 else float("nan"),
        "n": int(n),
    }
    for j, c in enumerate(columns):
        out[f"rho_{c}"] = float(rho[j])
    return out


def difficulty_regression(
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    perm: np.ndarray,
    columns: list[str],
    seed: int = 0,
    purge: int = 200,
    control_perm: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """T1 under all three conditions. ``perm`` is the loader visiting order.

    ``control_perm`` is the feature-row shuffle of the negative control. Pass an
    independent draw per cell: deriving it from ``seed`` alone would reuse the
    same shuffle across every cell of equal length, which correlates the control
    arm across cells and understates its spread.

    Two flavours of the same question are reported. The *descriptive* flavour
    (``association``) is what a paper prints in a table; the *predictive*
    flavour fits a gradient-boosted model on the joined validation table and
    measures rank agreement on the untouched test split, which is what a paper
    claims when it says window difficulty is predictable. Rank agreement rather
    than R2 because the validation and test splits cover different calendar
    periods and their error levels differ by construction.

    The target is ``log`` error: per-window MSE is heavy tailed and the join
    question concerns ordering information, which a log transform preserves.
    """
    if control_perm is None:
        control_perm = np.random.default_rng(seed).permutation(len(x_val))
    control_perm = np.asarray(control_perm, dtype=np.int64)
    y_val_log = np.log(np.maximum(y_val, 1e-12))
    y_test_log = np.log(np.maximum(y_test, 1e-12))
    variants = {
        "index": (x_val, y_val_log),
        "defect": (x_val, y_val_log[perm]),
        "control": (x_val[control_perm], y_val_log),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (xv, yv) in variants.items():
        stats_d = association(xv, yv, columns, y_raw=np.exp(yv))
        model = _fit_regressor(seed)
        model.fit(xv, yv)
        pred_test = model.predict(x_test)
        tr, ho = purged_split(len(yv), purge)
        m2 = _fit_regressor(seed)
        m2.fit(xv[tr], yv[tr])
        pred_hold = m2.predict(xv[ho])
        out[name] = {
            **stats_d,
            "spearman_test": float(stats.spearmanr(pred_test, y_test_log).statistic),
            "r2_test": _r2(y_test_log, pred_test),
            "spearman_holdout": float(stats.spearmanr(pred_hold, yv[ho]).statistic),
            "n_val": int(len(yv)),
            "n_test": int(len(y_test_log)),
        }
    return out


def arm_selection(
    x_val: np.ndarray,
    val_errs: np.ndarray,
    x_test: np.ndarray,
    test_errs: np.ndarray,
    perms: np.ndarray,
    arms: list[str],
    seed: int = 0,
    control_perm: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """T2 under all three conditions.

    ``val_errs`` / ``test_errs`` are ``(n_arms, n_windows)`` in index order.
    ``perms`` is ``(n_arms, n_val_windows)``: one loader order per arm, which is
    what a defective pipeline produces -- the permutations are independent
    because each arm is a separate process with its own loader.
    """
    if control_perm is None:
        control_perm = np.random.default_rng(seed).permutation(len(x_val))
    control_perm = np.asarray(control_perm, dtype=np.int64)
    k = val_errs.shape[0]
    fixed_val = val_errs.mean(axis=1)
    best_fixed = int(np.argmin(fixed_val))
    per_window_test = test_errs
    oracle = per_window_test.min(axis=0).mean()
    best_fixed_mse = per_window_test[best_fixed].mean()
    random_mse = per_window_test.mean(axis=0).mean()

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "index": (x_val, val_errs),
        "defect": (x_val, np.stack([val_errs[a][perms[a]] for a in range(k)])),
        "control": (x_val[control_perm], val_errs),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (xv, ve) in variants.items():
        labels = np.argmin(ve, axis=0)
        if len(np.unique(labels)) < 2:
            realised = best_fixed_mse
            acc = float("nan")
        else:
            clf = _fit_classifier(seed)
            clf.fit(xv, labels)
            pick = clf.predict(x_test)
            realised = float(per_window_test[pick, np.arange(per_window_test.shape[1])].mean())
            acc = float((pick == np.argmin(per_window_test, axis=0)).mean())
        out[name] = {
            "realised_mse": float(realised),
            "rel_gain_vs_best_fixed": float(100.0 * (best_fixed_mse - realised) / best_fixed_mse),
            "oracle_headroom_pct": float(100.0 * (best_fixed_mse - oracle) / best_fixed_mse),
            "best_fixed_mse": float(best_fixed_mse),
            "oracle_mse": float(oracle),
            "random_arm_mse": float(random_mse),
            "best_fixed_arm": arms[best_fixed],
            "oracle_match_acc": acc,
            "n_arms": int(k),
        }
    return out


def paired_summary(df: pd.DataFrame, value: str, by: str = "condition") -> pd.DataFrame:
    """Paired comparison of every condition against ``index``, block by block."""
    wide = df.pivot_table(index="block_id", columns=by, values=value)
    rows = []
    for cond in wide.columns:
        if cond == "index":
            continue
        d = (wide[cond] - wide["index"]).dropna()
        if len(d) < 3:
            continue
        t = stats.wilcoxon(d) if np.any(d != 0) else None
        rows.append(
            {
                "condition": cond,
                "n_blocks": int(len(d)),
                "mean_index": float(wide["index"].reindex(d.index).mean()),
                "mean_condition": float(wide[cond].reindex(d.index).mean()),
                "mean_diff": float(d.mean()),
                "median_diff": float(d.median()),
                "wilcoxon_p": float(t.pvalue) if t is not None else 1.0,
            }
        )
    return pd.DataFrame(rows)


def equivalence_test(a: np.ndarray, b: np.ndarray, margin: float) -> dict[str, float]:
    """Two one-sided tests for equivalence of paired samples within ``margin``.

    Needed because the claim about ``defect`` versus ``control`` is a claim of
    *indistinguishability*, and a non-significant difference is not evidence of
    that on its own.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = d.size
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 1e-300
    t_lower = (d.mean() + margin) / se
    t_upper = (d.mean() - margin) / se
    p_lower = stats.t.sf(t_lower, df=n - 1)
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    return {
        "n": float(n),
        "mean_diff": float(d.mean()),
        "sd_diff": float(sd),
        "margin": float(margin),
        "p_tost": float(max(p_lower, p_upper)),
        "equivalent_at_05": float(max(p_lower, p_upper) < 0.05),
        "ci95_low": float(d.mean() - stats.t.isf(0.025, n - 1) * se),
        "ci95_high": float(d.mean() + stats.t.isf(0.025, n - 1) * se),
    }


def equivalence_test_clustered(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    margin: float,
) -> dict[str, float]:
    """Cluster-level TOST: collapse each cluster to its mean paired difference first.

    Cells that share a dataset also share an evaluation sample set, so treating
    them as independent replicates understates the standard error. Averaging
    within cluster and testing across cluster means is the conservative reading:
    the effective sample size is the number of clusters, not the number of cells.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    clusters = np.asarray(clusters)
    keep = np.isfinite(d)
    d, clusters = d[keep], clusters[keep]
    labels = np.unique(clusters)
    means = np.array([d[clusters == c].mean() for c in labels])
    out = equivalence_test(means, np.zeros_like(means), margin)
    out["n_clusters"] = float(len(labels))
    out["n_cells"] = float(d.size)
    return out


__all__ = [
    "CONDITIONS",
    "FEATURE_COLUMNS",
    "arm_selection",
    "difficulty_regression",
    "equivalence_test",
    "equivalence_test_clustered",
    "paired_summary",
]
