"""Two post-hoc tests for the order provenance of a per-window error vector.

Neither test needs the model, the data or a rerun: they read the stored vectors
only. That is the point -- an artefact that has already been published can be
audited by whoever downloaded it.

AOT, the autocorrelation order test
-----------------------------------
Sliding windows overlap: consecutive windows of a split share
``seq_len + pred_len - 1`` of their points, so consecutive per-window errors are
strongly positively dependent. A random permutation destroys that dependence.
The statistic is the circular serial correlation, whose randomisation
distribution has exact first two moments (Wald and Wolfowitz, 1943) and is
asymptotically normal, so a p-value is available in O(n) with no resampling.

CAT, the cross-arm agreement test
---------------------------------
When several arms (plugins, hyper-parameters, seeds) are evaluated on the same
windows, their per-window errors agree strongly, because window difficulty
dominates arm identity. A defective pipeline draws an *independent* permutation
per arm, so the agreement collapses. CAT needs no serial structure at all, which
makes it the applicable test for i.i.d. data where AOT has no signal.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# AOT
# --------------------------------------------------------------------------- #
def circular_serial_corr(x: np.ndarray) -> float:
    """Circular lag-1 serial correlation of ``x`` (the Wald-Wolfowitz statistic).

    Circular rather than linear because the wrap-around term makes the
    randomisation moments exact instead of asymptotic.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        return float("nan")
    c = x - x.mean()
    num = float((c * np.roll(c, -1)).sum())
    den = float((c * c).sum())
    return num / den if den > 0 else float("nan")


def ww_moments(x: np.ndarray) -> tuple[float, float]:
    """Exact mean and variance of the circular serial correlation under a random
    permutation of the *observed values* (Wald and Wolfowitz, 1943).

    Working on the centred vector makes the power sums that appear in the
    formula the central moments, and the statistic is invariant to centring.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    c = x - x.mean()
    s2 = float((c**2).sum())
    s3 = float((c**3).sum())
    s4 = float((c**4).sum())
    if n < 4 or s2 <= 0:
        return float("nan"), float("nan")
    mean = -1.0 / (n - 1)
    # Variance of R = sum_i c_i c_{i+1} (circular), then scaled by s2^2.
    var_r = (
        (s2**2 - s4) / (n - 1)
        + (s2**2 - 2.0 * s4 + 4.0 * s2 * 0.0 + 4.0 * s3 * 0.0) / ((n - 1) * (n - 2))
        - (s2**2) / (n - 1) ** 2
    )
    var = var_r / (s2**2)
    return float(mean), float(max(var, 1e-300))


def aot(x: np.ndarray, n_perm: int = 0, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Autocorrelation order test.

    Returns the statistic, a one-sided normal-approximation p-value against
    "the vector is in a random order", and optionally a Monte-Carlo permutation
    p-value for calibration checks.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    r = circular_serial_corr(x)
    mean, var = ww_moments(x)
    out: dict[str, float] = {"n": float(n), "r": r, "null_mean": mean, "null_sd": float(np.sqrt(var))}
    if not np.isfinite(r) or not np.isfinite(var):
        out["z"] = float("nan")
        out["p"] = float("nan")
        return out
    z = (r - mean) / np.sqrt(var)
    out["z"] = float(z)
    out["p"] = float(stats.norm.sf(z))
    if n_perm:
        rng = rng or np.random.default_rng(0)
        stat = np.empty(n_perm)
        for b in range(n_perm):
            stat[b] = circular_serial_corr(rng.permutation(x))
        out["p_perm"] = float((1.0 + np.sum(stat >= r)) / (n_perm + 1.0))
    return out


def aot_power_bound(rho: float, n: int, alpha: float = 0.05) -> float:
    """Asymptotic power of AOT for a sequence with true serial correlation rho.

    Under the alternative the statistic concentrates near ``rho`` while the null
    spread is ``~n^{-1/2}``, so power is ``Phi(rho*sqrt(n) - z_alpha)``. Useful
    as the smallest detectable dependence at a given length.
    """
    z_alpha = stats.norm.isf(alpha)
    return float(stats.norm.cdf(rho * np.sqrt(n) - z_alpha))


# --------------------------------------------------------------------------- #
# CAT
# --------------------------------------------------------------------------- #
def mean_pairwise_spearman(mat: np.ndarray) -> float:
    """Mean pairwise Spearman correlation between the rows of ``mat``."""
    mat = np.asarray(mat, dtype=np.float64)
    k = mat.shape[0]
    if k < 2:
        return float("nan")
    ranks = np.apply_along_axis(stats.rankdata, 1, mat)
    ranks = ranks - ranks.mean(axis=1, keepdims=True)
    norm = np.sqrt((ranks**2).sum(axis=1))
    corr = (ranks @ ranks.T) / np.outer(norm, norm)
    iu = np.triu_indices(k, k=1)
    return float(corr[iu].mean())


def cat(
    mat: np.ndarray, n_perm: int = 2000, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """Cross-arm agreement test on a ``(n_arms, n_windows)`` matrix.

    The null is "each row was independently permuted", which is exactly what a
    defective pipeline produces, so the Monte-Carlo null is the honest one: draw
    independent permutations of each row.
    """
    mat = np.asarray(mat, dtype=np.float64)
    k, n = mat.shape
    obs = mean_pairwise_spearman(mat)
    rng = rng or np.random.default_rng(0)
    stat = np.empty(n_perm)
    ranks = np.apply_along_axis(stats.rankdata, 1, mat)
    for b in range(n_perm):
        perm = np.stack([rng.permutation(row) for row in ranks])
        stat[b] = mean_pairwise_spearman(perm)
    p = float((1.0 + np.sum(stat >= obs)) / (n_perm + 1.0))
    # analytic reference: Spearman under independence has variance 1/(n-1)
    z = obs * np.sqrt(n - 1)
    return {
        "n_arms": float(k),
        "n": float(n),
        "rho_bar": obs,
        "p": p,
        "z_analytic": float(z),
        "p_analytic": float(stats.norm.sf(z)),
        "null_mean": float(stat.mean()),
        "null_sd": float(stat.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
# permutation generators used by the experiments
# --------------------------------------------------------------------------- #
def full_permutation(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def block_permutation(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle the order of contiguous blocks but keep each block intact.

    This is the milder variant produced by shuffling batch order only (e.g. a
    ``BatchSampler`` over a ``SequentialSampler``), and it is the interesting
    stress case for AOT: within-block dependence survives.
    """
    idx = np.arange(n)
    blocks = [idx[i : i + block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def partial_permutation(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Permute a random ``frac`` subset of positions, leave the rest in place."""
    idx = np.arange(n)
    m = int(round(frac * n))
    if m < 2:
        return idx
    pos = rng.choice(n, size=m, replace=False)
    idx[np.sort(pos)] = pos[rng.permutation(m)]
    return idx


def apply_permutation(values: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Return the vector a loader with visiting order ``perm`` would record."""
    return np.asarray(values)[np.asarray(perm, dtype=np.int64)]
