"""Major-revision artefacts for T1: dependence-aware association and dynamic purge.

Subcommand-free: one pass over every finished cell writes three CSVs into
``artifacts/``.

``t1_difficulty_depaware.csv``  per cell x condition, Spearman rho with a
    circular moving-block bootstrap CI and a circular block permutation p-value,
    next to the retained (anti-conservative) iid p-value.
``t1_purge_audit.csv``          per cell, the purge implied by the window span
    under both rules, with the realised boundary numbers and eligibility.
``t1_cluster_summary.csv``      cluster bootstrap over sample sets and over
    datasets, because 712 sequences are not 712 independent experiments.

Nothing is subsampled: every cell in results/cells is processed.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402
from src.downstream import (  # noqa: E402
    CONDITIONS,
    MIN_HOLDOUT_WINDOWS,
    PURGE_MODES,
    association_dep_aware,
    cluster_bootstrap,
    purge_boundary_audit,
    purged_split,
    purged_split_audited,
)
from src.seeds import stable_rng, stable_seed  # noqa: E402
from src.winerr import load_series  # noqa: E402

from tools.make_artifacts import _art, feature_matrix, load_runs, win_dir  # noqa: E402

LEGACY_PURGE = 200
IDENT = ["cell_id", "dataset", "backbone", "plugin", "pred_len", "seq_len", "seed", "val_order"]
DEPAWARE_COLS = [
    "cell_id", "dataset", "backbone", "plugin", "pred_len", "seq_len", "seed", "val_order",
    "condition", "n", "assoc_rho_fstd", "assoc_rho_ci_lo", "assoc_rho_ci_hi",
    "assoc_rho_ci_excludes_zero", "assoc_p_block", "assoc_p_boot_2sided", "assoc_p_iid_naive",
    "block_len", "block_len_requested", "block_len_cover", "block_len_floor",
    "block_len_capped", "n_blocks", "n_boot", "n_perm",
]

_CFG: GridConfig | None = None
_FEAT: dict[tuple, np.ndarray] = {}
_OPTS: dict = {}


def _init(config: str | None, opts: dict) -> None:
    global _CFG
    _CFG = GridConfig(config)
    _OPTS.update(opts)


def _fstd(dataset: str, pred_len: int, split: str) -> np.ndarray:
    key = (dataset, pred_len, split)
    if key not in _FEAT:
        _FEAT[key] = feature_matrix(_CFG, dataset, pred_len, split)["f_std"].to_numpy(np.float64)
    return _FEAT[key]


def _cell(rec: dict) -> dict:
    """One cell: dependence-aware association per condition plus the purge audit."""
    cid = rec["cell_id"]
    L, H = int(rec["seq_len"]), int(rec["pred_len"])
    d = win_dir(_CFG, cid)
    yv = load_series(d, "val_index_order", legacy_ok=True)
    f = _fstd(rec["dataset"], H, "val")
    if f.size != yv.size:
        raise AssertionError(f"{cid}: feature/error length mismatch {f.size}/{yv.size}")
    n = int(yv.size)
    if rec["val_order"] == "shuffled":
        perm = load_series(d, "val_perm", require_order=None, legacy_ok=True).astype(np.int64)
    else:
        perm = stable_rng("t1", cid).permutation(n)
    control_perm = stable_rng("t1-control", cid).permutation(n)
    y_log = np.log(np.maximum(yv, 1e-12))
    variants = {
        "index": (f, y_log),
        "defect": (f, y_log[perm]),
        "control": (f[control_perm], y_log),
    }
    ident = {k: rec[k] for k in IDENT}
    rows = []
    for cond, (fx, fy) in variants.items():
        res = association_dep_aware(
            fx, fy, L, H,
            n_boot=_OPTS["n_boot"], n_perm=_OPTS["n_perm"],
            seed=stable_seed("t1-depaware", cid, cond) % (2**32),
        )
        rows.append({**ident, "condition": cond, **res})

    audit = {**ident, "n": n, "window_cover": L + H}
    for mode in PURGE_MODES:
        _, _, a = purged_split_audited(n, L, H, mode, min_windows=_OPTS["min_windows"])
        for k, v in a.items():
            if k in ("n", "seq_len", "pred_len", "window_cover", "purge_mode"):
                continue
            audit[f"{mode}_{k}"] = v
    tr, ho = purged_split(n, LEGACY_PURGE)
    legacy = purge_boundary_audit(tr, ho, L, H, "disjoint", strict=False)
    audit["legacy_purge_origins"] = float(LEGACY_PURGE)
    for k in (
        "min_origin_gap_actual", "shared_timepoints_at_boundary",
        "shared_target_points_at_boundary", "n_train_windows", "n_eval_windows",
    ):
        audit[f"legacy_{k}"] = legacy[k]
    return {"assoc": rows, "audit": audit}


def _sample_set(df: pd.DataFrame) -> pd.Series:
    return (
        df["dataset"] + "|L" + df["seq_len"].astype(str) + "|H" + df["pred_len"].astype(str)
        + "|val"
    )


def _cluster_rows(wide: pd.DataFrame, n_boot: int) -> list[dict]:
    stats_spec = [
        ("index_median_rho", wide["rho_index"].to_numpy(), np.median),
        ("defect_minus_index_median", (wide["rho_defect"] - wide["rho_index"]).to_numpy(), np.median),
        ("control_minus_index_median", (wide["rho_control"] - wide["rho_index"]).to_numpy(), np.median),
        ("defect_minus_control_median", (wide["rho_defect"] - wide["rho_control"]).to_numpy(), np.median),
        ("index_frac_p_block_lt_05", wide["sig_block_index"].to_numpy(), np.mean),
        ("index_frac_ci_excludes_zero", wide["ci_excl_index"].to_numpy(), np.mean),
        ("index_frac_p_iid_lt_05", wide["sig_iid_index"].to_numpy(), np.mean),
    ]
    out = []
    for cdef, col in (("sample_set", "cluster_sample_set"), ("dataset", "cluster_dataset")):
        for name, vals, fn in stats_spec:
            r = cluster_bootstrap(
                vals, wide[col].to_numpy(), stat=fn, n_boot=n_boot,
                seed=stable_seed("cluster", cdef, name) % (2**32),
            )
            out.append({"cluster_def": cdef, "statistic": name, **r})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-cluster-boot", type=int, default=4000)
    ap.add_argument("--min-windows", type=int, default=MIN_HOLDOUT_WINDOWS)
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = GridConfig(args.config)
    runs = load_runs(cfg)
    keep = [
        r for _, r in runs.iterrows()
        if (win_dir(cfg, r["cell_id"]) / "val_index_order.npy").exists()
    ]
    recs = [{k: (int(r[k]) if k in ("pred_len", "seq_len", "seed") else r[k]) for k in IDENT}
            for r in keep]
    print(f"[rev] {len(recs)} cells, jobs={args.jobs}, boot={args.n_boot}, perm={args.n_perm}")

    opts = {"n_boot": args.n_boot, "n_perm": args.n_perm, "min_windows": args.min_windows}
    assoc_rows: list[dict] = []
    audit_rows: list[dict] = []
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.config, opts)) as ex:
        for i, res in enumerate(ex.map(_cell, recs, chunksize=1), 1):
            assoc_rows.extend(res["assoc"])
            audit_rows.append(res["audit"])
            if i % 25 == 0 or i == len(recs):
                print(f"[rev] {i}/{len(recs)} cells  {time.time() - t0:.0f}s", flush=True)

    art = _art(cfg)
    dep = pd.DataFrame(assoc_rows)
    dep = dep[DEPAWARE_COLS + [c for c in dep.columns if c not in DEPAWARE_COLS]]
    dep.to_csv(art / "t1_difficulty_depaware.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(art / "t1_purge_audit.csv", index=False)

    ix = dep[dep.condition == "index"].copy()
    wide = ix[["cell_id", "dataset", "seq_len", "pred_len"]].copy()
    wide["cluster_sample_set"] = _sample_set(ix)
    wide["cluster_dataset"] = ix["dataset"].to_numpy()
    for cond in CONDITIONS:
        s = dep[dep.condition == cond].set_index("cell_id")["assoc_rho_fstd"]
        wide[f"rho_{cond}"] = s.reindex(ix["cell_id"]).to_numpy()
    wide["sig_block_index"] = (ix["assoc_p_block"] < 0.05).astype(float).to_numpy()
    wide["sig_iid_index"] = (ix["assoc_p_iid_naive"] < 0.05).astype(float).to_numpy()
    wide["ci_excl_index"] = ix["assoc_rho_ci_excludes_zero"].to_numpy()
    clus = pd.DataFrame(_cluster_rows(wide, args.n_cluster_boot))
    clus.to_csv(art / "t1_cluster_summary.csv", index=False)

    r = ix["assoc_rho_fstd"].to_numpy()
    q1, q3 = np.percentile(r, [25, 75])
    head = {
        "n_cells": int(len(ix)),
        "n_sequences_val_plus_test": int(2 * len(ix)),
        "n_distinct_sample_sets": int(
            len(audit[["dataset", "seq_len", "pred_len"]].drop_duplicates()) * 2
        ),
        "n_distinct_sample_sets_val": int(wide["cluster_sample_set"].nunique()),
        "n_datasets": int(audit["dataset"].nunique()),
        "rho_median": float(np.median(r)),
        "rho_q1": float(q1),
        "rho_q3": float(q3),
        "rho_iqr": float(q3 - q1),
        "rho_min": float(r.min()),
        "rho_max": float(r.max()),
        "frac_p_block_lt_05": float((ix["assoc_p_block"] < 0.05).mean()),
        "frac_ci_excludes_zero": float(ix["assoc_rho_ci_excludes_zero"].mean()),
        "frac_p_iid_naive_lt_05": float((ix["assoc_p_iid_naive"] < 0.05).mean()),
        "block_len_median": float(ix["block_len"].median()),
        "block_len_capped_cells": int(ix["block_len_capped"].sum()),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "n_cluster_boot": args.n_cluster_boot,
        "min_holdout_windows": args.min_windows,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for mode in PURGE_MODES:
        head[f"{mode}_eligible"] = int(audit[f"{mode}_holdout_eligible"].sum())
        head[f"{mode}_ineligible"] = int((audit[f"{mode}_holdout_eligible"] == 0).sum())
    (art / "t1_depaware_summary.json").write_text(
        json.dumps(head, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("[rev] headline:", json.dumps(head, indent=2, sort_keys=True))
    print("[rev] cluster bootstrap:")
    print(clus.to_string(index=False))
    for mode in PURGE_MODES:
        bad = audit[audit[f"{mode}_holdout_eligible"] == 0]
        print(f"[rev] {mode}: ineligible cells by (dataset, pred_len):")
        if len(bad):
            print(bad.groupby(["dataset", "pred_len"]).size().to_string())
    print(f"[rev] done in {time.time() - t0:.0f}s -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
