"""CAT revision: observed permutation sharing, sharing conditions, corrected analytic reference."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from math import comb, sqrt
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402
from src.detect import aot, apply_permutation, cat  # noqa: E402
from src.seeds import stable_rng  # noqa: E402
from src.winerr import ORDER_INDEX, load_series  # noqa: E402
from tools.make_artifacts import ALPHA, _art, load_runs, win_dir  # noqa: E402

CONDITIONS = ("intact", "independent", "all_shared", "shared_k3", "shared_k2", "two_pairs")


def sharing_labels(condition: str, k: int) -> list[int] | None:
    """Group label per arm; arms sharing a label share one permutation."""
    if condition == "intact":
        return None
    if condition == "independent":
        return list(range(k))
    if condition == "all_shared":
        return [0] * k
    if condition == "shared_k3":
        m = min(3, k)
        return [0] * m + list(range(1, k - m + 1))
    if condition == "shared_k2":
        m = min(2, k)
        return [0] * m + list(range(1, k - m + 1))
    if condition == "two_pairs":
        if k < 4:
            return None
        return [0, 0, 1, 1] + list(range(2, k - 2))
    raise ValueError(f"unknown condition {condition!r}")


def inject(mat: np.ndarray, labels: list[int], rng: np.random.Generator) -> np.ndarray:
    """Apply one permutation per distinct label to the intact rows."""
    n = mat.shape[1]
    perms = {lab: rng.permutation(n) for lab in sorted(set(labels))}
    return np.stack([apply_permutation(row, perms[lab]) for row, lab in zip(mat, labels)])


def collect_groups(cfg: GridConfig) -> list[dict]:
    """The real stage-E2 groups: shuffled val runs keyed by (block_id, seed)."""
    runs = load_runs(cfg)
    runs = runs[runs["val_order"] == "shuffled"]
    groups: list[dict] = []
    for (blk, seed), g in runs.groupby(["block_id", "seed"]):
        g = g.sort_values("plugin")
        arms, dirs = [], []
        for _, r in g.iterrows():
            d = win_dir(cfg, r["cell_id"])
            if not (d / "val_loader_order.npy").exists():
                continue
            arms.append(str(r["plugin"]))
            dirs.append(str(d))
        if len(arms) < 2:
            continue
        groups.append({"block_id": str(blk), "seed": int(seed), "arms": arms, "dirs": dirs})
    return groups


def _load_group(grp: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], int]:
    intact, loader, perms, n_min = [], [], [], None
    for d in grp["dirs"]:
        y_idx = load_series(d, "val_index_order", require_order=ORDER_INDEX, legacy_ok=True)
        y_ld = load_series(d, "val_loader_order", require_order=None, legacy_ok=True)
        p = load_series(d, "val_perm", require_order=None, legacy_ok=True).astype(np.int64)
        intact.append(y_idx)
        loader.append(y_ld)
        perms.append(p)
        n_min = y_idx.size if n_min is None else min(n_min, y_idx.size)
    n_min = int(n_min)
    return (
        np.stack([v[:n_min] for v in intact]),
        np.stack([v[:n_min] for v in loader]),
        perms,
        n_min,
    )


def scan_row(grp: dict, intact: np.ndarray, loader: np.ndarray, perms, n: int, n_perm: int) -> dict:
    blk, seed, arms = grp["block_id"], grp["seed"], grp["arms"]
    k = len(arms)
    pair_ident, pair_rho = [], []
    for i, j in combinations(range(k), 2):
        pair_ident.append(bool(np.array_equal(perms[i], perms[j])))
        pair_rho.append(float(stats.spearmanr(perms[i], perms[j]).statistic))
    non_ident = [r for r, e in zip(pair_rho, pair_ident) if not e]
    distinct = {p.tobytes() for p in perms}
    n_distinct = len(distinct)
    if n_distinct == 1:
        sharing = "all_identical"
    elif n_distinct == k:
        sharing = "all_distinct"
    else:
        sharing = "partial_shared"
    n_mis = sum(1 for p in perms if not np.array_equal(p, np.arange(p.size)))
    consistent = all(
        np.allclose(apply_permutation(intact[i], perms[i]), loader[i])
        for i in range(k)
        if perms[i].size == n
    )
    r_obs = cat(loader, n_perm=n_perm, rng=stable_rng("cat-rev-observed", blk, seed))
    r_int = cat(intact, n_perm=n_perm, rng=stable_rng("cat-rev-intact", blk, seed))
    aot_p = [float(aot(loader[i])["p"]) for i in range(k)]
    return {
        "block_id": blk,
        "seed": seed,
        "n_arms": k,
        "n_windows": n,
        "arms": "|".join(arms),
        "n_pairs": len(pair_ident),
        "n_pairs_identical": int(sum(pair_ident)),
        "spearman_perm_mean": float(np.mean(pair_rho)),
        "spearman_perm_max": float(np.max(pair_rho)),
        "spearman_perm_min": float(np.min(pair_rho)),
        "spearman_perm_max_nonidentical": float(np.max(non_ident)) if non_ident else float("nan"),
        "n_distinct_perms": n_distinct,
        "perm_sharing": sharing,
        "n_arms_misordered": int(n_mis),
        "perm_consistent": int(bool(consistent)),
        "cat_rho_bar": r_obs["rho_bar"],
        "cat_p_mc": r_obs["p"],
        "cat_certified": int(r_obs["p"] < ALPHA),
        "cat_p_analytic": r_obs["p_analytic"],
        "cat_rho_bar_intact": r_int["rho_bar"],
        "cat_p_mc_intact": r_int["p"],
        "cat_certified_intact": int(r_int["p"] < ALPHA),
        "aot_p_min": float(np.min(aot_p)),
        "aot_p_max": float(np.max(aot_p)),
        "aot_p_median": float(np.median(aot_p)),
        "n_arms_aot_rejects": int(sum(1 for p in aot_p if p < ALPHA)),
        "n_arms_aot": k,
    }


def condition_rows(grp: dict, intact: np.ndarray, n: int, n_perm: int) -> list[dict]:
    blk, seed, arms = grp["block_id"], grp["seed"], grp["arms"]
    k = len(arms)
    out = []
    for cond in CONDITIONS:
        labels = sharing_labels(cond, k)
        if cond != "intact" and labels is None:
            continue
        if labels is None:
            mat = intact
        else:
            mat = inject(intact, labels, stable_rng("cat-rev-inject", cond, blk, seed))
        res = cat(mat, n_perm=n_perm, rng=stable_rng("cat-rev-null", cond, blk, seed))
        m_pairs = comb(k, 2)
        out.append(
            {
                "block_id": blk,
                "seed": seed,
                "condition": cond,
                "n_arms": k,
                "n": n,
                "n_distinct_perms_injected": 0 if labels is None else len(set(labels)),
                "sharing_labels": "" if labels is None else "|".join(str(x) for x in labels),
                "rho_bar": res["rho_bar"],
                "p_mc": res["p"],
                "certified": int(res["p"] < ALPHA),
                "m_pairs": m_pairs,
                "p_analytic": res["p_analytic"],
                "p_analytic_naive": res["p_analytic_naive"],
                "z_analytic": res["z_analytic"],
                "z_analytic_naive": res["z_analytic_naive"],
                "null_mean_mc": res["null_mean"],
                "null_sd_mc": res["null_sd"],
                "null_sd_theory_corrected": 1.0 / sqrt(m_pairs * (n - 1)),
                "null_sd_theory_naive": 1.0 / sqrt(n - 1),
            }
        )
    return out


def _work(task: tuple) -> tuple[dict, list[dict]]:
    grp, n_perm = task
    intact, loader, perms, n = _load_group(grp)
    scan = scan_row(grp, intact, loader, perms, n, n_perm)
    conds = condition_rows(grp, intact, n, n_perm)
    print(
        f"[group] {grp['block_id']} s{grp['seed']} n={n} sharing={scan['perm_sharing']} "
        f"cat_p_obs={scan['cat_p_mc']:.4g}",
        flush=True,
    )
    return scan, conds


def main() -> None:
    ap = argparse.ArgumentParser(description="CAT revision artefacts")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N groups")
    args = ap.parse_args()

    cfg = GridConfig(args.config)
    groups = collect_groups(cfg)
    if args.limit:
        groups = groups[: args.limit]
    print(f"[setup] {len(groups)} real E2 groups, n_perm={args.n_perm}, jobs={args.jobs}")

    tasks = [(g, args.n_perm) for g in groups]
    if args.jobs > 1 and len(tasks) > 1:
        with Pool(min(args.jobs, len(tasks))) as pool:
            results = pool.map(_work, tasks)
    else:
        results = [_work(t) for t in tasks]

    scan = pd.DataFrame([r[0] for r in results]).sort_values(["block_id", "seed"])
    conds = pd.DataFrame([row for r in results for row in r[1]])
    conds["condition"] = pd.Categorical(conds["condition"], categories=CONDITIONS, ordered=True)
    conds = conds.sort_values(["block_id", "seed", "condition"])

    art = _art(cfg)
    scan.to_csv(art / "observed_perm_scan.csv", index=False)
    an_cols = [
        "block_id", "seed", "condition", "n_arms", "n", "m_pairs", "rho_bar",
        "p_mc", "p_analytic", "p_analytic_naive", "z_analytic", "z_analytic_naive",
        "null_sd_mc", "null_sd_theory_corrected", "null_sd_theory_naive",
    ]
    an = conds[an_cols].copy()
    an["abs_err_corrected"] = (an["p_analytic"] - an["p_mc"]).abs()
    an["abs_err_naive"] = (an["p_analytic_naive"] - an["p_mc"]).abs()
    an["sd_ratio_corrected"] = an["null_sd_mc"] / an["null_sd_theory_corrected"]
    an["sd_ratio_naive"] = an["null_sd_mc"] / an["null_sd_theory_naive"]
    an.to_csv(art / "cat_analytic_check.csv", index=False)
    conds.drop(columns=["m_pairs"]).to_csv(art / "cat_conditions.csv", index=False)

    print("\n=== A. observed permutation sharing (real E2 groups) ===")
    print(f"groups={len(scan)}  arms/group={sorted(scan['n_arms'].unique())}  "
          f"n_windows={sorted(scan['n_windows'].unique())}")
    print(scan["perm_sharing"].value_counts().to_string())
    print("n_pairs_identical distribution:")
    print(scan["n_pairs_identical"].value_counts().sort_index().to_string())
    print(f"n_distinct_perms: {scan['n_distinct_perms'].value_counts().sort_index().to_dict()}")
    print(
        f"spearman_perm_mean range [{scan['spearman_perm_mean'].min():.4f}, "
        f"{scan['spearman_perm_mean'].max():.4f}] median {scan['spearman_perm_mean'].median():.4f}; "
        f"spearman_perm_max range [{scan['spearman_perm_max'].min():.4f}, "
        f"{scan['spearman_perm_max'].max():.4f}]"
    )
    print(
        f"max Spearman among non-identical permutation pairs = "
        f"{scan['spearman_perm_max_nonidentical'].max():.4f} (identical pairs excluded)"
    )
    print(
        f"arms with non-identity observed permutation: "
        f"{int(scan['n_arms_misordered'].sum())}/{int(scan['n_arms'].sum())}; "
        f"loader_order == perm(index_order) for all arms in "
        f"{int(scan['perm_consistent'].sum())}/{len(scan)} groups"
    )

    print("\n=== B. CAT on the real loader-order vectors (misordered ground truth) ===")
    cert = int(scan["cat_certified"].sum())
    print(
        f"certified (p<{ALPHA}) = {cert}/{len(scan)} = {100 * cert / len(scan):.1f}%  "
        f"median rho_bar={scan['cat_rho_bar'].median():.4f} "
        f"[{scan['cat_rho_bar'].min():.4f}, {scan['cat_rho_bar'].max():.4f}]  "
        f"median p={scan['cat_p_mc'].median():.4g} max p={scan['cat_p_mc'].max():.4g}"
    )
    ci = int(scan["cat_certified_intact"].sum())
    print(
        f"control on the intact val_index_order vectors: certified {ci}/{len(scan)} "
        f"median rho_bar={scan['cat_rho_bar_intact'].median():.4f} "
        f"median p={scan['cat_p_mc_intact'].median():.4g}"
    )
    by_share = scan.groupby("perm_sharing").agg(
        groups=("cat_certified", "size"),
        certified=("cat_certified", "sum"),
        median_rho=("cat_rho_bar", "median"),
    )
    print(by_share.to_string())

    print("\n=== C. AOT on the real loader-order vectors ===")
    tot_arms = int(scan["n_arms_aot"].sum())
    rej = int(scan["n_arms_aot_rejects"].sum())
    print(
        f"arm-vectors rejecting the random-order null (p<{ALPHA}) = {rej}/{tot_arms} "
        f"= {100 * rej / tot_arms:.1f}%  p_min={scan['aot_p_min'].min():.4g} "
        f"median of per-group median p={scan['aot_p_median'].median():.4f}"
    )
    print(f"groups with >=1 AOT-rejecting arm: {int((scan['n_arms_aot_rejects'] > 0).sum())}/{len(scan)}")

    print("\n=== D. CAT certification rate by permutation-sharing condition ===")
    tab = conds.groupby("condition", observed=True).agg(
        groups=("certified", "size"),
        certified=("certified", "sum"),
        median_rho_bar=("rho_bar", "median"),
        median_p_mc=("p_mc", "median"),
    )
    tab["cert_rate"] = tab["certified"] / tab["groups"]
    for cond, r in tab.iterrows():
        print(
            f"  {str(cond):12s} certified {int(r['certified'])}/{int(r['groups'])} "
            f"({100 * r['cert_rate']:5.1f}%)  median rho_bar={r['median_rho_bar']:+.4f}  "
            f"median p_mc={r['median_p_mc']:.4g}"
        )

    print("\n=== E. analytic reference vs Monte-Carlo p ===")
    for label, sub in [("all rows", an)] + [(f"cond={c}", an[an["condition"] == c]) for c in CONDITIONS]:
        if not len(sub):
            continue
        print(
            f"  {label:18s} n={len(sub):3d} corrected max|dp|={sub['abs_err_corrected'].max():.4f} "
            f"median|dp|={sub['abs_err_corrected'].median():.4f} | naive max|dp|="
            f"{sub['abs_err_naive'].max():.4f} median|dp|={sub['abs_err_naive'].median():.4f}"
        )
    print(
        f"  null_sd_mc / theory_corrected: median={an['sd_ratio_corrected'].median():.4f} "
        f"[{an['sd_ratio_corrected'].min():.4f}, {an['sd_ratio_corrected'].max():.4f}]"
    )
    print(
        f"  null_sd_mc / theory_naive:     median={an['sd_ratio_naive'].median():.4f} "
        f"[{an['sd_ratio_naive'].min():.4f}, {an['sd_ratio_naive'].max():.4f}]"
    )
    ind = an[an["condition"] == "independent"]
    if len(ind):
        print(
            f"  reviewer baseline subset (condition=independent, n={len(ind)}): naive max|dp|="
            f"{ind['abs_err_naive'].max():.4f} (ref 0.2119), median sd_ratio_corrected="
            f"{ind['sd_ratio_corrected'].median():.4f} (ref 1.0023)"
        )

    base = scan[(scan["block_id"] == "ETTh1_h336_DLinear") & (scan["seed"] == 2021)]
    if len(base) == 1:
        r = base.iloc[0]
        print("\n=== F. regression baseline DLinear_ETTh1_h336_*_s2021_shuf ===")
        print(
            f"  cat_rho_bar={r['cat_rho_bar']:.10f} cat_p_mc={r['cat_p_mc']:.10f} "
            f"n_windows={int(r['n_windows'])} sharing={r['perm_sharing']} "
            f"n_pairs_identical={int(r['n_pairs_identical'])}"
        )
        ok = (
            abs(r["cat_rho_bar"] - 0.4824092858) < 1e-6
            and abs(r["cat_p_mc"] - 0.0004997501) < 1e-6
        )
        print(f"  matches expected rho_bar=0.4824092858 / p=0.0004997501: {ok}")

    print(
        f"\n[out] {art / 'observed_perm_scan.csv'}\n[out] {art / 'cat_conditions.csv'}"
        f"\n[out] {art / 'cat_analytic_check.csv'}"
    )


if __name__ == "__main__":
    main()
