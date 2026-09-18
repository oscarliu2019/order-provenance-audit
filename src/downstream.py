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


PURGE_MODES = ("labels", "disjoint")
MIN_HOLDOUT_WINDOWS = 200


def required_origin_gap(seq_len: int, pred_len: int, purge_mode: str = "disjoint") -> int:
    """Minimum window-origin separation that makes a train/holdout pair honest.

    ``labels``   the last training target must close before the first holdout
                 forecast origin, i.e. no shared *target* point: gap >= H.
    ``disjoint`` no shared time point at all, input included: gap >= L + H.
    """
    if purge_mode not in PURGE_MODES:
        raise ValueError(f"purge_mode must be one of {PURGE_MODES}, got {purge_mode!r}")
    L, H = int(seq_len), int(pred_len)
    return H if purge_mode == "labels" else L + H


def purge_for_mode(seq_len: int, pred_len: int, purge_mode: str = "disjoint") -> int:
    """``purged_split`` leaves ``2 * purge + 1`` origins between the two parts."""
    gap = required_origin_gap(seq_len, pred_len, purge_mode)
    return int(np.ceil((gap - 1) / 2.0))


def window_span(origin: int, seq_len: int, pred_len: int) -> dict[str, int]:
    """Half-open interval endpoints of one sliding window, in raw time points."""
    o, L, H = int(origin), int(seq_len), int(pred_len)
    return {
        "input_start": o,
        "input_end": o + L,
        "target_start": o + L,
        "target_end": o + L + H,
        "cover_start": o,
        "cover_end": o + L + H,
    }


def purge_boundary_audit(
    train: np.ndarray,
    hold: np.ndarray,
    seq_len: int,
    pred_len: int,
    purge_mode: str = "disjoint",
    strict: bool = True,
) -> dict[str, float]:
    """Validate a purged split from the interval endpoints it actually produced.

    Nothing here reads the purge constant back: the boundary numbers are derived
    from ``max(train)`` and ``min(hold)`` so a wrong purge cannot pass silently.
    """
    L, H = int(seq_len), int(pred_len)
    out: dict[str, float] = {
        "purge_mode": purge_mode,
        "required_origin_gap": float(required_origin_gap(L, H, purge_mode)),
        "n_train_windows": float(len(train)),
        "n_eval_windows": float(len(hold)),
    }
    if len(train) == 0 or len(hold) == 0:
        out.update(
            {
                "last_train_origin": float("nan"),
                "first_eval_origin": float("nan"),
                "min_origin_gap_actual": float("nan"),
                "last_train_target_end": float("nan"),
                "first_eval_input_start": float("nan"),
                "first_eval_target_start": float("nan"),
                "shared_timepoints_at_boundary": float("nan"),
                "shared_target_points_at_boundary": float("nan"),
                "boundary_ok": 0.0,
            }
        )
        return out
    a = window_span(int(np.max(train)), L, H)
    b = window_span(int(np.min(hold)), L, H)
    gap = b["input_start"] - a["input_start"]
    shared_all = max(0, a["cover_end"] - b["cover_start"])
    shared_tgt = max(0, a["target_end"] - b["target_start"])
    ok = shared_tgt == 0 if purge_mode == "labels" else shared_all == 0
    out.update(
        {
            "last_train_origin": float(a["input_start"]),
            "first_eval_origin": float(b["input_start"]),
            "min_origin_gap_actual": float(gap),
            "last_train_target_end": float(a["target_end"]),
            "first_eval_input_start": float(b["input_start"]),
            "first_eval_target_start": float(b["target_start"]),
            "shared_timepoints_at_boundary": float(shared_all),
            "shared_target_points_at_boundary": float(shared_tgt),
            "boundary_ok": float(bool(ok)),
        }
    )
    if strict and not ok:
        raise AssertionError(
            f"purge_mode={purge_mode!r} violated: origin gap {gap} < "
            f"{out['required_origin_gap']:.0f}, shared target points {shared_tgt}, "
            f"shared time points {shared_all}"
        )
    return out


def purged_split_audited(
    n: int,
    seq_len: int,
    pred_len: int,
    purge_mode: str = "disjoint",
    frac: float = 0.5,
    min_windows: int = MIN_HOLDOUT_WINDOWS,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Chronological split whose purge is derived from the window span itself.

    The purge is never shrunk to make a cell usable: a cell whose sequence is
    too short for the rule is reported as ``holdout_eligible = 0`` instead.
    """
    n = int(n)
    purge = purge_for_mode(seq_len, pred_len, purge_mode)
    cut = int(n * frac)
    lo = cut - purge
    hi = cut + purge
    train = np.arange(0, max(lo, 0))
    hold = np.arange(min(hi, n), n)
    audit = purge_boundary_audit(train, hold, seq_len, pred_len, purge_mode, strict=False)
    eligible = (
        len(train) >= min_windows
        and len(hold) >= min_windows
        and audit["boundary_ok"] == 1.0
    )
    audit.update(
        {
            "n": float(n),
            "seq_len": float(seq_len),
            "pred_len": float(pred_len),
            "window_cover": float(int(seq_len) + int(pred_len)),
            "purge_origins": float(purge),
            "min_windows_required": float(min_windows),
            "holdout_eligible": float(bool(eligible)),
        }
    )
    if eligible:
        purge_boundary_audit(train, hold, seq_len, pred_len, purge_mode, strict=True)
    return train, hold, audit


def block_length_for_overlap(n: int, seq_len: int, pred_len: int) -> dict[str, float]:
    """Block length for a moving-block resample of a stride-1 sliding-window series.

    Two windows are statistically dependent while their spans intersect, i.e. up
    to ``L + H - 1`` origins apart, so ``L + H`` is the smallest block that can
    carry the whole dependence range; ``ceil(n^(1/3))`` is the usual consistency
    floor for block bootstrap and takes over only for very long series.
    """
    n = int(n)
    cover = int(seq_len) + int(pred_len)
    floor_rule = int(np.ceil(n ** (1.0 / 3.0)))
    want = max(cover, floor_rule)
    used = max(1, min(want, n // 2))
    return {
        "block_len": float(used),
        "block_len_requested": float(want),
        "block_len_cover": float(cover),
        "block_len_floor": float(floor_rule),
        "block_len_capped": float(used < want),
        "n_blocks": float(int(np.ceil(n / used))),
    }


def _rank(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a, axis=-1)


def _corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / den


def _mbb_indices(n: int, block_len: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]


def _cbp_indices(n: int, block_len: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    base = np.arange(n)
    bounds = list(range(0, n, block_len)) + [n]
    out = np.empty((reps, n), dtype=np.int64)
    for r in range(reps):
        rolled = np.roll(base, int(rng.integers(n)))
        order = rng.permutation(len(bounds) - 1)
        out[r] = np.concatenate([rolled[bounds[k]: bounds[k + 1]] for k in order])
    return out


def association_dep_aware(
    f: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    pred_len: int,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 0,
    chunk: int = 100,
) -> dict[str, float]:
    """Dependence-aware inference for the (window feature, window error) association.

    ``assoc_rho_fstd`` is unchanged and stays purely descriptive. Uncertainty is
    read off a *circular moving-block bootstrap* of the joined pairs, and the
    p-value ``assoc_p_block`` comes from a *circular block permutation* null: the
    error series is cut into whole blocks whose order is shuffled after a random
    circular shift, which preserves each series' autocorrelation exactly while
    destroying only the window-to-window alignment that the join asserts. The
    iid formula is kept as ``assoc_p_iid_naive`` for contrast only.
    """
    f = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(f.size)
    rho = float(stats.spearmanr(f, y).statistic)
    bl = block_length_for_overlap(n, seq_len, pred_len)
    b = int(bl["block_len"])
    rng = np.random.default_rng(seed)
    rf, ry = stats.rankdata(f), stats.rankdata(y)

    boot = np.empty(n_boot, dtype=np.float64)
    for c0 in range(0, n_boot, chunk):
        c1 = min(c0 + chunk, n_boot)
        idx = _mbb_indices(n, b, c1 - c0, rng)
        boot[c0:c1] = _corr_rows(_rank(rf[idx]), _rank(ry[idx]))
    boot_ok = boot[np.isfinite(boot)]

    perm = np.empty(n_perm, dtype=np.float64)
    for c0 in range(0, n_perm, chunk):
        c1 = min(c0 + chunk, n_perm)
        idx = _cbp_indices(n, b, c1 - c0, rng)
        perm[c0:c1] = _corr_rows(np.broadcast_to(rf, (c1 - c0, n)), ry[idx])
    perm_ok = perm[np.isfinite(perm)]

    lo, hi = (
        (float(np.percentile(boot_ok, 2.5)), float(np.percentile(boot_ok, 97.5)))
        if boot_ok.size
        else (float("nan"), float("nan"))
    )
    p_block = float((1 + int((np.abs(perm_ok) >= abs(rho)).sum())) / (perm_ok.size + 1))
    side = min((boot_ok <= 0).mean(), (boot_ok >= 0).mean()) if boot_ok.size else float("nan")
    p_iid = float(2 * stats.norm.sf(abs(rho) * np.sqrt(max(n - 1, 1))))
    return {
        "n": int(n),
        "assoc_rho_fstd": rho,
        "assoc_rho_ci_lo": lo,
        "assoc_rho_ci_hi": hi,
        "assoc_rho_ci_excludes_zero": float(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
        "assoc_p_block": p_block,
        "assoc_p_boot_2sided": float(max(2 * side, 1.0 / max(boot_ok.size, 1))),
        "assoc_p_iid_naive": p_iid,
        "assoc_p_fstd": p_iid,
        "boot_rho_mean": float(boot_ok.mean()) if boot_ok.size else float("nan"),
        "n_boot": int(boot_ok.size),
        "n_perm": int(perm_ok.size),
        **bl,
    }


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    stat=np.median,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Resample whole clusters, not rows: overlapping cells are not replicates."""
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    keep = np.isfinite(v)
    v, c = v[keep], c[keep]
    labels = np.unique(c)
    groups = [v[c == k] for k in labels]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    k = len(groups)
    for r in range(n_boot):
        pick = rng.integers(0, k, size=k)
        draws[r] = stat(np.concatenate([groups[i] for i in pick]))
    return {
        "point": float(stat(v)),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "boot_median": float(np.median(draws)),
        "n_clusters": int(k),
        "n_cells": int(v.size),
        "n_boot": int(n_boot),
    }


def association(
    x: np.ndarray, y: np.ndarray, columns: list[str], y_raw: np.ndarray | None = None
) -> dict[str, float]:
    """Descriptive per-window association statistics of a joined table.

    This is the form of per-sample analysis that appears in papers: correlate a
    per-sample loss with a per-sample covariate, or bin the loss by covariate
    decile. No model is fitted, so nothing here depends on a learner's
    inductive bias -- only on whether row ``i`` of the feature table and entry
    ``i`` of the error vector describe the same window.

    ``assoc_p_fstd`` (alias ``assoc_p_iid_naive``) treats the strongly
    overlapping windows as iid pairs and is therefore anti-conservative; use
    :func:`association_dep_aware` for inference and keep this key for contrast.
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
        "assoc_p_iid_naive": float(2 * stats.norm.sf(abs(rho[fstd]) * np.sqrt(max(n - 1, 1)))),
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
    purge: int | None = 200,
    control_perm: np.ndarray | None = None,
    seq_len: int | None = None,
    pred_len: int | None = None,
    purge_mode: str = "disjoint",
    min_holdout_windows: int = MIN_HOLDOUT_WINDOWS,
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

    ``purge=None`` switches the holdout split to :func:`purged_split_audited`,
    whose gap is derived from ``seq_len`` / ``pred_len`` under ``purge_mode``; an
    ineligible cell reports ``holdout_eligible=0`` and a NaN holdout statistic
    rather than a shrunken purge.
    """
    if control_perm is None:
        control_perm = np.random.default_rng(seed).permutation(len(x_val))
    control_perm = np.asarray(control_perm, dtype=np.int64)
    if purge is None and (seq_len is None or pred_len is None):
        raise ValueError("purge=None requires both seq_len and pred_len")
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
        if purge is None:
            tr, ho, audit = purged_split_audited(
                len(yv), seq_len, pred_len, purge_mode, min_windows=min_holdout_windows
            )
        else:
            tr, ho = purged_split(len(yv), purge)
            audit = {"holdout_eligible": 1.0, "purge_origins": float(purge)}
        if audit["holdout_eligible"] == 1.0:
            m2 = _fit_regressor(seed)
            m2.fit(xv[tr], yv[tr])
            pred_hold = m2.predict(xv[ho])
            rho_hold = float(stats.spearmanr(pred_hold, yv[ho]).statistic)
        else:
            rho_hold = float("nan")
        extra = {f"purge_{k}": v for k, v in audit.items()} if purge is None else {}
        out[name] = {
            **stats_d,
            "spearman_test": float(stats.spearmanr(pred_test, y_test_log).statistic),
            "r2_test": _r2(y_test_log, pred_test),
            "spearman_holdout": rho_hold,
            "n_val": int(len(yv)),
            "n_test": int(len(y_test_log)),
            **extra,
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
    "MIN_HOLDOUT_WINDOWS",
    "PURGE_MODES",
    "arm_selection",
    "association",
    "association_dep_aware",
    "block_length_for_overlap",
    "cluster_bootstrap",
    "difficulty_regression",
    "equivalence_test",
    "equivalence_test_clustered",
    "paired_summary",
    "purge_boundary_audit",
    "purge_for_mode",
    "purged_split",
    "purged_split_audited",
    "required_origin_gap",
    "window_span",
]
