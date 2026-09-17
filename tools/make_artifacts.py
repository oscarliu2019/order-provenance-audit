"""Build every artefact the paper cites, from the raw per-window vectors.

Subcommands
-----------
``runs``      aggregate results/cells/*.json into artifacts/runs.csv
``features``  per-window feature tables (index order) for every (dataset, h)
``aot``       autocorrelation order test over the whole sequence bank
``cat``       cross-arm agreement test, block by block
``downstream``T1 difficulty regression and T2 arm selection, three conditions
``rngtax``    paired deterministic/shuffled runs from stage E2
``all``       everything, in dependency order

Every number that appears in the paper is written here and re-checked by
tools/verify_paper_numbers.py, so a claim in the text can always be traced to a
row in a CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402
from src.detect import (  # noqa: E402
    aot,
    ww_moments,
    aot_power_bound,
    apply_permutation,
    block_permutation,
    cat,
    partial_permutation,
)
from src.downstream import (  # noqa: E402
    FEATURE_COLUMNS,
    arm_selection,
    difficulty_regression,
    equivalence_test,
    equivalence_test_clustered,
    paired_summary,
)
from src.seeds import stable_rng  # noqa: E402
from src.winerr import ORDER_INDEX, load_series  # noqa: E402

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _art(cfg: GridConfig) -> Path:
    p = cfg.path("artifacts_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_runs(cfg: GridConfig) -> pd.DataFrame:
    rows = []
    for p in sorted(cfg.path("cells_dir").glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no finished cells found")
    df = pd.DataFrame(rows)
    df["block_id"] = df["dataset"] + "_h" + df["pred_len"].astype(str) + "_" + df["backbone"]
    return df.sort_values("cell_id").reset_index(drop=True)


def feature_matrix(cfg: GridConfig, dataset: str, pred_len: int, split: str) -> pd.DataFrame:
    cache_dir = _art(cfg) / "window_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{dataset}_h{pred_len}_{split}.csv"
    if p.exists():
        return pd.read_csv(p)
    from src.features import split_features

    df = split_features(cfg, dataset, pred_len, split)
    df.to_csv(p, index=False)
    return df


def win_dir(cfg: GridConfig, cell_id: str) -> Path:
    return cfg.path("winerr_dir") / cell_id


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def cmd_runs(cfg: GridConfig, args) -> None:
    df = load_runs(cfg)
    out = _art(cfg) / "runs.csv"
    df.to_csv(out, index=False)
    print(f"[runs] {len(df)} runs -> {out}")
    print(df.groupby(["backbone", "val_order"]).size().to_string())


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def cmd_features(cfg: GridConfig, args) -> None:
    n = 0
    for ds in cfg.raw["datasets"]:
        for h in ds["horizons"]:
            for split in ("val", "test"):
                df = feature_matrix(cfg, ds["name"], int(h), split)
                n += len(df)
    print(f"[features] cached {n} window rows")


# --------------------------------------------------------------------------- #
# AOT
# --------------------------------------------------------------------------- #
def cmd_aot(cfg: GridConfig, args) -> None:
    runs = load_runs(cfg)
    rng = np.random.default_rng(20260916)
    rows: list[dict] = []
    for _, r in runs.iterrows():
        d = win_dir(cfg, r["cell_id"])
        for split in ("val", "test"):
            name = f"{split}_index_order"
            try:
                y = load_series(d, name, require_order=ORDER_INDEX)
            except FileNotFoundError:
                continue
            base = {
                "cell_id": r["cell_id"],
                "dataset": r["dataset"],
                "backbone": r["backbone"],
                "plugin": r["plugin"],
                "pred_len": r["pred_len"],
                "seed": r["seed"],
                "val_order": r["val_order"],
                "split": split,
                "n": int(y.size),
            }
            res = aot(y)
            rows.append({**base, "variant": "index_order", **_aot_row(res)})
            # what the same run would have written under a shuffled loader
            perm = rng.permutation(y.size)
            res_p = aot(apply_permutation(y, perm))
            rows.append({**base, "variant": "full_permutation", **_aot_row(res_p)})
            if split == "val" and r["val_order"] == "shuffled":
                # the *observed* loader order of a real shuffled run
                obs = load_series(d, "val_loader_order", require_order=None)
                rows.append(
                    {**base, "variant": "observed_loader_order", **_aot_row(aot(obs))}
                )
    df = pd.DataFrame(rows)
    out = _art(cfg) / "aot_sequences.csv"
    df.to_csv(out, index=False)
    print(f"[aot] {len(df)} tests -> {out}")
    for v, g in df.groupby("variant"):
        print(
            f"  {v:22s} n_seq={len(g):4d} median r={g['r'].median():+.4f} "
            f"reject@{ALPHA}={100 * (g['p'] < ALPHA).mean():.1f}%"
        )

    # calibration of the normal approximation against a Monte-Carlo permutation
    # null on a stratified subsample
    sub = df[df["variant"] == "index_order"].sample(
        n=min(args.calib, len(df[df["variant"] == "index_order"])),
        random_state=0,
    )
    calib_rows = []
    for _, r in sub.iterrows():
        d = win_dir(cfg, r["cell_id"])
        y = load_series(d, f"{r['split']}_index_order", require_order=ORDER_INDEX)
        perm = stable_rng("aot-calib", r["cell_id"]).permutation(y.size)
        shuffled = apply_permutation(y, perm)
        res = aot(shuffled, n_perm=args.n_perm, rng=np.random.default_rng(7))
        calib_rows.append(
            {
                "cell_id": r["cell_id"],
                "split": r["split"],
                "n": int(y.size),
                "p_normal": res["p"],
                "p_perm": res["p_perm"],
            }
        )
    cal = pd.DataFrame(calib_rows)
    cal.to_csv(_art(cfg) / "aot_calibration.csv", index=False)
    if len(cal):
        print(
            f"[aot] calibration on {len(cal)} shuffled sequences: "
            f"false-positive rate normal={100 * (cal['p_normal'] < ALPHA).mean():.2f}% "
            f"permutation={100 * (cal['p_perm'] < ALPHA).mean():.2f}% "
            f"max|p_normal-p_perm|={np.abs(cal['p_normal'] - cal['p_perm']).max():.4f}"
        )

    # power against milder corruptions: block-order shuffling and partial shuffles
    sens_rows = []
    pool = df[(df["variant"] == "index_order") & (df["split"] == "val")]
    pool = pool.sample(n=min(args.sens, len(pool)), random_state=1)
    for _, r in pool.iterrows():
        d = win_dir(cfg, r["cell_id"])
        y = load_series(d, "val_index_order", require_order=ORDER_INDEX)
        g = stable_rng("sens", r["cell_id"])
        for b in args.blocks:
            if b >= y.size:
                continue
            p = aot(apply_permutation(y, block_permutation(y.size, b, g)))["p"]
            sens_rows.append(
                {"cell_id": r["cell_id"], "n": int(y.size), "kind": "block",
                 "param": int(b), "p": p, "reject": int(p < ALPHA)}
            )
        for f in args.fracs:
            p = aot(apply_permutation(y, partial_permutation(y.size, f, g)))["p"]
            sens_rows.append(
                {"cell_id": r["cell_id"], "n": int(y.size), "kind": "partial",
                 "param": float(f), "p": p, "reject": int(p < ALPHA)}
            )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(_art(cfg) / "aot_sensitivity.csv", index=False)
    if len(sens):
        print("[aot] sensitivity (detection rate):")
        print(
            sens.groupby(["kind", "param"])["reject"]
            .agg(["mean", "size"])
            .assign(mean=lambda x: (100 * x["mean"]).round(1))
            .to_string()
        )

    # theoretical smallest detectable serial correlation per sequence length
    theory = pd.DataFrame(
        [
            {
                "n": int(n),
                "rho_min_detectable": float(
                    np.sqrt(1.0 / n) * 1.6448536269514722 + np.sqrt(1.0 / n) * 1.2815515655446004
                ),
                "power_at_rho_0_5": aot_power_bound(0.5, int(n)),
                "power_at_rho_0_1": aot_power_bound(0.1, int(n)),
            }
            for n in (50, 100, 200, 500, 1000, 2000, 5000, 11000)
        ]
    )
    theory.to_csv(_art(cfg) / "aot_theory.csv", index=False)
    print(f"[aot] theory table -> {_art(cfg) / 'aot_theory.csv'}")


def cmd_aottail(cfg: GridConfig, args) -> None:
    """Measure the deep tail of the normal approximation, instead of trusting it.

    The randomisation moments in ww_moments are exact, but the normal reference
    is asymptotic, so a p-value quoted at 1e-6 is an extrapolation of a two-moment
    approximation. Here we take a stratified set of real sequences, draw
    ``--tail-perm`` independent uniform permutations of each (which is exactly the
    null of the test), and report how often the normal rule would reject at each
    nominal level. With B draws the resolution is 1/B, so B = 10^6 checks the rule
    down to about 1e-5 and bounds it at 1e-6.
    """
    seq = pd.read_csv(_art(cfg) / "aot_sequences.csv")
    clean = seq[(seq["variant"] == "index_order")].drop_duplicates("cell_id")
    # stratify by length: the approximation is worst for short sequences
    qs = np.quantile(clean["n"], [0.0, 0.25, 0.5, 0.75, 1.0])
    picks = []
    for q in qs:
        row = clean.iloc[(clean["n"] - q).abs().argsort().iloc[0]]
        if row["cell_id"] not in [p_["cell_id"] for p_ in picks]:
            picks.append(row)
    B = int(args.tail_perm)
    chunk = max(1, min(4096, 20_000_000 // max(int(max(p_["n"] for p_ in picks)), 1)))
    alphas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    rows = []
    for row in picks:
        d = win_dir(cfg, row["cell_id"])
        y = load_series(d, f"{row['split']}_index_order", require_order=ORDER_INDEX)
        n = int(y.size)
        mean, var = ww_moments(y)
        sd = float(np.sqrt(var))
        thresholds = np.array([mean + stats.norm.isf(a) * sd for a in alphas])
        counts = np.zeros(len(alphas), dtype=np.int64)
        c = y - y.mean()
        s2 = float((c * c).sum())
        rng = stable_rng("aot-tail", row["cell_id"], B)
        done = 0
        zmax = -np.inf
        while done < B:
            m = min(chunk, B - done)
            block = np.tile(c, (m, 1))
            block = rng.permuted(block, axis=1)
            num = (block * np.roll(block, -1, axis=1)).sum(axis=1)
            r = num / s2
            counts += (r[:, None] >= thresholds[None, :]).sum(axis=0)
            zmax = max(zmax, float(((r - mean) / sd).max()))
            done += m
        for a, k in zip(alphas, counts):
            rows.append(
                {
                    "cell_id": row["cell_id"],
                    "split": row["split"],
                    "n": n,
                    "kurtosis": float(stats.kurtosis(y, fisher=False)),
                    "n_perm": B,
                    "alpha": a,
                    "n_reject": int(k),
                    "fpr_empirical": float(k) / B,
                    "ratio_to_nominal": float(k) / B / a,
                    "max_z_observed": zmax,
                }
            )
        print(
            f"[aottail] {row['cell_id']} n={n} maxz={zmax:.2f} "
            + " ".join(f"a={a:.0e}:{k}" for a, k in zip(alphas, counts)),
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(_art(cfg) / "aot_tail.csv", index=False)
    print(df.to_string(index=False))


def _aot_row(res: dict) -> dict:
    return {
        "r": res["r"],
        "z": res["z"],
        "p": res["p"],
        "null_sd": res["null_sd"],
        "reject": int(res["p"] < ALPHA) if np.isfinite(res["p"]) else 0,
    }


# --------------------------------------------------------------------------- #
# CAT
# --------------------------------------------------------------------------- #
def cmd_cat(cfg: GridConfig, args) -> None:
    runs = load_runs(cfg)
    rows = []
    for (blk, seed, order), g in runs.groupby(["block_id", "seed", "val_order"]):
        if len(g) < 2:
            continue
        arms, mats = [], []
        n_min = None
        for _, r in g.sort_values("plugin").iterrows():
            try:
                y = load_series(win_dir(cfg, r["cell_id"]), "val_index_order")
            except FileNotFoundError:
                continue
            arms.append(r["plugin"])
            mats.append(y)
            n_min = y.size if n_min is None else min(n_min, y.size)
        if len(mats) < 2:
            continue
        mat = np.stack([m[:n_min] for m in mats])
        g_rng = stable_rng("cat", blk, seed, order)
        defect = np.stack([apply_permutation(m, g_rng.permutation(n_min)) for m in mat])
        base = {
            "block_id": blk,
            "seed": int(seed),
            "val_order": order,
            "n_arms": len(arms),
            "arms": "|".join(arms),
            "n": int(n_min),
        }
        r_idx = cat(mat, n_perm=args.n_perm)
        r_def = cat(defect, n_perm=args.n_perm)
        rows.append({**base, "variant": "index_order", **{k: v for k, v in r_idx.items()}})
        rows.append({**base, "variant": "full_permutation", **{k: v for k, v in r_def.items()}})
        # pairwise detail for the index-order case: how much do arms agree?
        for i, j in combinations(range(len(arms)), 2):
            from scipy import stats as _st

            rows.append(
                {
                    **base,
                    "variant": f"pair:{arms[i]}~{arms[j]}",
                    "rho_bar": float(_st.spearmanr(mat[i], mat[j]).statistic),
                    "p": float(_st.spearmanr(mat[i], mat[j]).pvalue),
                    "n_arms": 2,
                }
            )
    df = pd.DataFrame(rows)
    out = _art(cfg) / "cat_blocks.csv"
    df.to_csv(out, index=False)
    print(f"[cat] {len(df)} rows -> {out}")
    for v, g in df[df["variant"].isin(["index_order", "full_permutation"])].groupby("variant"):
        print(
            f"  {v:18s} blocks={len(g):4d} median rho_bar={g['rho_bar'].median():+.4f} "
            f"reject@{ALPHA}={100 * (g['p'] < ALPHA).mean():.1f}%"
        )


# --------------------------------------------------------------------------- #
# downstream
# --------------------------------------------------------------------------- #
def cmd_downstream(cfg: GridConfig, args) -> None:
    runs = load_runs(cfg)
    t1_rows: list[dict] = []
    for _, r in runs.iterrows():
        d = win_dir(cfg, r["cell_id"])
        try:
            yv = load_series(d, "val_index_order")
            yt = load_series(d, "test_index_order")
        except FileNotFoundError:
            continue
        fv = feature_matrix(cfg, r["dataset"], int(r["pred_len"]), "val")
        ft = feature_matrix(cfg, r["dataset"], int(r["pred_len"]), "test")
        if len(fv) != yv.size or len(ft) != yt.size:
            raise AssertionError(
                f"{r['cell_id']}: feature/error length mismatch "
                f"val {len(fv)}/{yv.size} test {len(ft)}/{yt.size}"
            )
        xv = fv[FEATURE_COLUMNS].to_numpy()
        xt = ft[FEATURE_COLUMNS].to_numpy()
        if r["val_order"] == "shuffled":
            perm = load_series(d, "val_perm", require_order=None).astype(np.int64)
            perm_source = "observed"
        else:
            perm = stable_rng("t1", r["cell_id"]).permutation(yv.size)
            perm_source = "simulated"
        res = difficulty_regression(
            xv, yv, xt, yt, perm, FEATURE_COLUMNS, seed=int(r["seed"]),
            control_perm=stable_rng("t1-control", r["cell_id"]).permutation(yv.size),
        )
        for cond, m in res.items():
            t1_rows.append(
                {
                    "cell_id": r["cell_id"],
                    "block_id": r["block_id"],
                    "dataset": r["dataset"],
                    "backbone": r["backbone"],
                    "plugin": r["plugin"],
                    "pred_len": int(r["pred_len"]),
                    "seed": int(r["seed"]),
                    "val_order": r["val_order"],
                    "perm_source": perm_source,
                    "condition": cond,
                    **m,
                }
            )
        print(
            f"[t1] {r['cell_id']} rho_fstd index={res['index']['assoc_rho_fstd']:+.3f} "
            f"defect={res['defect']['assoc_rho_fstd']:+.3f} control={res['control']['assoc_rho_fstd']:+.3f}",
            flush=True,
        )
    t1 = pd.DataFrame(t1_rows)
    t1.to_csv(_art(cfg) / "t1_difficulty.csv", index=False)
    print(f"[t1] {len(t1)} rows; association and transfer by condition:")
    print(
        t1.groupby("condition")[
            ["assoc_rho_fstd", "assoc_rho_best", "decile_ratio_fstd", "spearman_test",
             "spearman_holdout"]
        ]
        .agg(["mean", "median"])
        .to_string()
    )

    # T2 needs all arms of a block at once
    t2_rows: list[dict] = []
    for (blk, seed, order), g in runs.groupby(["block_id", "seed", "val_order"]):
        g = g.sort_values("plugin")
        arms, vs, ts = [], [], []
        for _, r in g.iterrows():
            d = win_dir(cfg, r["cell_id"])
            try:
                vs.append(load_series(d, "val_index_order"))
                ts.append(load_series(d, "test_index_order"))
            except FileNotFoundError:
                continue
            arms.append(r["plugin"])
        if len(arms) < 2:
            continue
        nv = min(v.size for v in vs)
        nt = min(t.size for t in ts)
        val_errs = np.stack([v[:nv] for v in vs])
        test_errs = np.stack([t[:nt] for t in ts])
        ds, h, bk = g.iloc[0]["dataset"], int(g.iloc[0]["pred_len"]), g.iloc[0]["backbone"]
        fv = feature_matrix(cfg, ds, h, "val")
        ft = feature_matrix(cfg, ds, h, "test")
        xv = fv[FEATURE_COLUMNS].to_numpy()[:nv]
        xt = ft[FEATURE_COLUMNS].to_numpy()[:nt]
        rg = stable_rng("t2", blk, seed, order)
        perms = np.stack([rg.permutation(nv) for _ in arms])
        res = arm_selection(
            xv, val_errs, xt, test_errs, perms, arms, seed=int(seed),
            control_perm=stable_rng("t2-control", blk, seed, order).permutation(nv),
        )
        for cond, m in res.items():
            t2_rows.append(
                {
                    "block_id": blk,
                    "dataset": ds,
                    "backbone": bk,
                    "pred_len": h,
                    "seed": int(seed),
                    "val_order": order,
                    "condition": cond,
                    "arms": "|".join(arms),
                    **m,
                }
            )
        print(
            f"[t2] {blk} s{seed} {order} index={res['index']['rel_gain_vs_best_fixed']:+.2f}% "
            f"defect={res['defect']['rel_gain_vs_best_fixed']:+.2f}% "
            f"control={res['control']['rel_gain_vs_best_fixed']:+.2f}%",
            flush=True,
        )
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(_art(cfg) / "t2_selection.csv", index=False)
    print(f"[t2] {len(t2)} rows; mean rel_gain by condition:")
    print(
        t2.groupby("condition")["rel_gain_vs_best_fixed"]
        .agg(["mean", "median", "size"])
        .to_string()
    )

    # paired contrasts and equivalence tests
    summ = {}
    for tag, df, value in (
        ("t1_rho", t1, "assoc_rho_fstd"),
        ("t1_transfer", t1, "spearman_test"),
        ("t2_gain", t2, "rel_gain_vs_best_fixed"),
    ):
        key = "block_id" if tag == "t2_gain" else "cell_id"
        d = df.copy()
        d["block_id"] = d[key] + "_" + d["seed"].astype(str) + "_" + d["val_order"]
        ps = paired_summary(d, value)
        ps.insert(0, "task", tag)
        summ[tag] = ps
        wide = d.pivot_table(index="block_id", columns="condition", values=value)
        wide = wide.dropna()
        margin = args.margin_gain if tag == "t2_gain" else args.margin_rho
        eq = equivalence_test(wide["defect"].to_numpy(), wide["control"].to_numpy(), margin)
        eq_row = {"task": tag, **eq}
        # conservative reading: cells that share a dataset share a sample set
        ds = d.drop_duplicates("block_id").set_index("block_id")["dataset"]
        eqc = equivalence_test_clustered(
            wide["defect"].to_numpy(),
            wide["control"].to_numpy(),
            ds.reindex(wide.index).to_numpy(),
            margin,
        )
        for k, v in eqc.items():
            eq_row[f"clustered_{k}"] = v
        pd.DataFrame([eq_row]).to_csv(
            _art(cfg) / f"equivalence_{tag}.csv", index=False
        )
        print(f"[equiv] {tag}: defect vs control {eq}")
        print(f"[equiv] {tag}: dataset-clustered {eqc}")
    pd.concat(summ.values(), ignore_index=True).to_csv(
        _art(cfg) / "downstream_paired.csv", index=False
    )


# --------------------------------------------------------------------------- #
# replicated-draw equivalence: the single-draw TOST is a Monte Carlo experiment
# --------------------------------------------------------------------------- #
def _eq_draw_cell(job: tuple) -> list[dict]:
    """One cell of the replicated equivalence experiment.

    The defect is a random permutation and so is the control shuffle, so a TOST
    computed from one draw of each is itself a Monte Carlo experiment: its p-value
    has a sampling distribution. This recomputes the two T1 statistics over
    ``draws`` independent draws of both conditions, which lets us report the
    spread of the single-draw test and a draw-averaged estimate whose Monte Carlo
    component is ``draws`` times smaller.
    """
    from src.downstream import _fit_regressor, association

    cell_id, xv, yv, xt, yt, seed, draws, fstd_col = job
    y_val_log = np.log(np.maximum(yv, 1e-12))
    y_test_log = np.log(np.maximum(yt, 1e-12))
    out = []
    for d in range(draws):
        dperm = stable_rng("eq-defect", cell_id, d).permutation(y_val_log.size)
        cperm = stable_rng("eq-control", cell_id, d).permutation(y_val_log.size)
        for cond, (x_, y_) in (
            ("defect", (xv, y_val_log[dperm])),
            ("control", (xv[cperm], y_val_log)),
        ):
            rho = association(x_, y_, fstd_col, y_raw=np.exp(y_))["assoc_rho_fstd"]
            m = _fit_regressor(seed)
            m.fit(x_, y_)
            tr = float(stats.spearmanr(m.predict(xt), y_test_log).statistic)
            out.append(
                {
                    "cell_id": cell_id,
                    "draw": d,
                    "condition": cond,
                    "assoc_rho_fstd": rho,
                    "spearman_test": tr,
                }
            )
    return out


def cmd_eqdraws(cfg: GridConfig, args) -> None:
    import os
    from concurrent.futures import ProcessPoolExecutor

    runs = load_runs(cfg)
    jobs = []
    for _, r in runs.iterrows():
        d = win_dir(cfg, r["cell_id"])
        try:
            yv = load_series(d, "val_index_order")
            yt = load_series(d, "test_index_order")
        except FileNotFoundError:
            continue
        fv = feature_matrix(cfg, r["dataset"], int(r["pred_len"]), "val")
        ft = feature_matrix(cfg, r["dataset"], int(r["pred_len"]), "test")
        jobs.append(
            (
                r["cell_id"],
                fv[FEATURE_COLUMNS].to_numpy(),
                yv,
                ft[FEATURE_COLUMNS].to_numpy(),
                yt,
                int(r["seed"]),
                int(args.eq_draws),
                FEATURE_COLUMNS,
            )
        )
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=int(args.eq_jobs)) as ex:
        for i, res in enumerate(ex.map(_eq_draw_cell, jobs), 1):
            rows.extend(res)
            if i % 25 == 0 or i == len(jobs):
                print(f"[eqdraws] {i}/{len(jobs)} cells", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(_art(cfg) / "equivalence_draws_raw.csv", index=False)

    summary = []
    for stat, margin in (
        ("assoc_rho_fstd", args.margin_rho),
        ("spearman_test", args.margin_rho),
    ):
        per_draw = []
        for d, g in df.groupby("draw"):
            w = g.pivot_table(index="cell_id", columns="condition", values=stat).dropna()
            eq = equivalence_test(w["defect"].to_numpy(), w["control"].to_numpy(), margin)
            per_draw.append({"stat": stat, "draw": int(d), **eq})
        pd_df = pd.DataFrame(per_draw)
        mean_cell = (
            df.groupby(["cell_id", "condition"])[stat].mean().unstack("condition").dropna()
        )
        eq_mean = equivalence_test(
            mean_cell["defect"].to_numpy(), mean_cell["control"].to_numpy(), margin
        )
        summary.append(
            {
                "stat": stat,
                "draws": int(args.eq_draws),
                "margin": float(margin),
                "single_p_min": float(pd_df["p_tost"].min()),
                "single_p_max": float(pd_df["p_tost"].max()),
                "single_p_median": float(pd_df["p_tost"].median()),
                "single_certify_frac": float(pd_df["equivalent_at_05"].mean()),
                "single_absdiff_max": float(pd_df["mean_diff"].abs().max()),
                "mean_n": eq_mean["n"],
                "mean_diff": eq_mean["mean_diff"],
                "mean_ci95_low": eq_mean["ci95_low"],
                "mean_ci95_high": eq_mean["ci95_high"],
                "mean_p_tost": eq_mean["p_tost"],
                "mean_equivalent_at_05": eq_mean["equivalent_at_05"],
            }
        )
        print(f"[eqdraws] {stat}: {summary[-1]}", flush=True)
    pd.DataFrame(summary).to_csv(_art(cfg) / "equivalence_draws.csv", index=False)


# --------------------------------------------------------------------------- #
# label law: why a multi-arm control is optimistic rather than merely powerless
# --------------------------------------------------------------------------- #
def cmd_labellaw(cfg: GridConfig, args) -> None:
    """Per-window best-arm labels under the correct join and under the defect.

    The shuffled-feature control keeps the arms aligned, so its label vector is a
    permutation of the correct one and its majority class is unchanged. A defective
    pipeline permutes each arm independently, so the label is an argmin over
    misaligned vectors. This records the label law of both, with no model fitting,
    so the mechanism behind the T2 gap is checkable on its own.
    """
    runs = load_runs(cfg)
    rows = []
    for (blk, seed, order), g in runs.groupby(["block_id", "seed", "val_order"]):
        g = g.sort_values("plugin")
        arms, vs = [], []
        for _, r in g.iterrows():
            try:
                vs.append(load_series(win_dir(cfg, r["cell_id"]), "val_index_order"))
            except FileNotFoundError:
                continue
            arms.append(r["plugin"])
        if len(arms) < 2:
            continue
        n = min(v.size for v in vs)
        val = np.stack([v[:n] for v in vs])
        best_fixed = int(np.argmin(val.mean(axis=1)))
        rg = stable_rng("t2", blk, seed, order)
        perms = np.stack([rg.permutation(n) for _ in arms])
        defect = np.stack([val[a][perms[a]] for a in range(len(arms))])

        def law(mat: np.ndarray) -> dict[str, float]:
            lab = np.argmin(mat, axis=0)
            freq = np.bincount(lab, minlength=len(arms)) / lab.size
            nz = freq[freq > 0]
            return {
                "modal_arm": int(np.argmax(freq)),
                "modal_share": float(freq.max()),
                "best_fixed_share": float(freq[best_fixed]),
                "modal_is_best_fixed": bool(int(np.argmax(freq)) == best_fixed),
                "entropy_bits": float(-(nz * np.log2(nz)).sum()),
            }

        for cond, mat in (("index", val), ("defect", defect)):
            rows.append(
                {
                    "block_id": blk,
                    "seed": int(seed),
                    "val_order": order,
                    "condition": cond,
                    "n_arms": len(arms),
                    "n": int(n),
                    **law(mat),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(_art(cfg) / "label_law.csv", index=False)
    print(f"[labellaw] {len(df)} rows")
    print(
        df.groupby("condition")[
            ["modal_share", "best_fixed_share", "modal_is_best_fixed", "entropy_bits"]
        ]
        .mean()
        .to_string()
    )


# --------------------------------------------------------------------------- #
# RNG tax
# --------------------------------------------------------------------------- #
def cmd_rngtax(cfg: GridConfig, args) -> None:
    runs = load_runs(cfg)
    key = ["dataset", "backbone", "plugin", "pred_len", "seed"]
    det = runs[runs["val_order"] == "deterministic"].set_index(key)
    shuf = runs[runs["val_order"] == "shuffled"].set_index(key)
    common = det.index.intersection(shuf.index)
    rows = []
    for k in common:
        a, b = det.loc[k], shuf.loc[k]
        rows.append(
            {
                "dataset": k[0], "backbone": k[1], "plugin": k[2],
                "pred_len": k[3], "seed": k[4],
                "mse_deterministic": float(a["mse"]),
                "mse_shuffled": float(b["mse"]),
                "abs_rel_diff_pct": float(100 * abs(b["mse"] - a["mse"]) / a["mse"]),
                "rel_diff_pct": float(100 * (b["mse"] - a["mse"]) / a["mse"]),
                "val_mse_deterministic": float(a["val_mse"]),
                "val_mse_shuffled": float(b["val_mse"]),
                "best_epoch_deterministic": int(a["best_epoch"]),
                "best_epoch_shuffled": int(b["best_epoch"]),
                "same_best_epoch": int(a["best_epoch"] == b["best_epoch"]),
            }
        )
    df = pd.DataFrame(rows)
    out = _art(cfg) / "rng_tax.csv"
    df.to_csv(out, index=False)
    print(f"[rngtax] {len(df)} paired cells -> {out}")
    if len(df):
        from scipy import stats as _st

        print(
            f"  median |rel diff| = {df['abs_rel_diff_pct'].median():.3f}%  "
            f"p90 = {df['abs_rel_diff_pct'].quantile(0.9):.3f}%  "
            f"max = {df['abs_rel_diff_pct'].max():.3f}%"
        )
        print(f"  same best epoch: {100 * df['same_best_epoch'].mean():.1f}%")
        w = _st.wilcoxon(df["mse_shuffled"], df["mse_deterministic"])
        print(f"  wilcoxon on signed diff: p={w.pvalue:.4f} (no systematic direction expected)")


# --------------------------------------------------------------------------- #
# battery: where each test works
# --------------------------------------------------------------------------- #
def cmd_battery(cfg: GridConfig, args) -> None:
    """AOT and CAT as a function of how much serial structure survives.

    Sub-sampling every ``s``-th window reduces the overlap between neighbours,
    which is the knob that separates the two tests: AOT lives on serial
    dependence, CAT does not need any. At a stride beyond ``seq_len + pred_len``
    the windows are disjoint, which is the i.i.d. regime where only CAT applies.
    """
    runs = load_runs(cfg)
    rows = []
    for (blk, seed, order), g in runs.groupby(["block_id", "seed", "val_order"]):
        g = g.sort_values("plugin")
        arms, mats = [], []
        for _, r in g.iterrows():
            try:
                mats.append(load_series(win_dir(cfg, r["cell_id"]), "val_index_order"))
            except FileNotFoundError:
                continue
            arms.append(r["plugin"])
        if len(mats) < 2:
            continue
        n_min = min(m.size for m in mats)
        mat = np.stack([m[:n_min] for m in mats])
        seq_len = int(g.iloc[0]["seq_len"])
        h = int(g.iloc[0]["pred_len"])
        for s in args.strides:
            sub = mat[:, ::s]
            if sub.shape[1] < 40:
                continue
            rg = stable_rng("battery", blk, seed, order, s)
            defect = np.stack([apply_permutation(row, rg.permutation(sub.shape[1])) for row in sub])
            for variant, m in (("index_order", sub), ("full_permutation", defect)):
                a_p = [aot(row)["p"] for row in m]
                c = cat(m, n_perm=args.n_perm)
                rows.append(
                    {
                        "block_id": blk,
                        "seed": int(seed),
                        "val_order": order,
                        "variant": variant,
                        "stride": int(s),
                        "overlap_ratio": float(max(0.0, 1.0 - s / (seq_len + h))),
                        "n": int(sub.shape[1]),
                        "n_arms": int(sub.shape[0]),
                        "aot_median_p": float(np.median(a_p)),
                        "aot_certify_rate": float(np.mean([p < ALPHA for p in a_p])),
                        "cat_rho_bar": c["rho_bar"],
                        "cat_p": c["p"],
                        "cat_certify": int(c["p"] < ALPHA),
                    }
                )
    df = pd.DataFrame(rows)
    out = _art(cfg) / "battery.csv"
    df.to_csv(out, index=False)
    print(f"[battery] {len(df)} rows -> {out}")
    if len(df):
        piv = df.pivot_table(
            index=["variant", "stride"],
            values=["overlap_ratio", "aot_certify_rate", "cat_certify", "n"],
            aggfunc="mean",
        )
        print(piv.round(3).to_string())


# --------------------------------------------------------------------------- #
# certification thresholds
# --------------------------------------------------------------------------- #
def cmd_certify(cfg: GridConfig, args) -> None:
    """Operating characteristics of the audit rule "certify iff p < alpha".

    The alarming error is certifying a *corrupted* artefact, and that error is
    exactly the size of the test, so it is controllable by construction: the
    table shows how far alpha can be pushed down before clean artefacts start
    failing to certify.
    """
    seq = pd.read_csv(_art(cfg) / "aot_sequences.csv")
    clean = seq[seq["variant"] == "index_order"]
    dirty = seq[seq["variant"] == "full_permutation"]
    rows = []
    for alpha in args.alphas:
        rows.append(
            {
                "alpha": alpha,
                "n_clean": int(len(clean)),
                "n_corrupted": int(len(dirty)),
                "clean_certified_pct": float(100 * (clean["p"] < alpha).mean()),
                "corrupted_certified_pct": float(100 * (dirty["p"] < alpha).mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(_art(cfg) / "certification.csv", index=False)
    print(df.to_string(index=False))
    margin = {
        "min_z_clean": float(clean["z"].min()),
        "median_z_clean": float(clean["z"].median()),
        "max_z_corrupted": float(dirty["z"].max()),
        "min_r_clean": float(clean["r"].min()),
        "median_r_clean": float(clean["r"].median()),
        "median_r_corrupted": float(dirty["r"].median()),
        "min_n_clean": int(clean["n"].min()),
    }
    (_art(cfg) / "certification_margin.json").write_text(json.dumps(margin, indent=2))
    print(json.dumps(margin, indent=2))


# --------------------------------------------------------------------------- #
def cmd_all(cfg: GridConfig, args) -> None:
    cmd_runs(cfg, args)
    cmd_features(cfg, args)
    cmd_aot(cfg, args)
    cmd_aottail(cfg, args)
    cmd_certify(cfg, args)
    cmd_cat(cfg, args)
    cmd_battery(cfg, args)
    cmd_downstream(cfg, args)
    cmd_eqdraws(cfg, args)
    cmd_labellaw(cfg, args)
    cmd_rngtax(cfg, args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "cmd",
        choices=["runs", "features", "aot", "aottail", "certify", "cat", "battery",
                 "downstream", "eqdraws", "labellaw", "rngtax", "all"],
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--calib", type=int, default=120)
    ap.add_argument("--tail-perm", type=int, default=1_000_000)
    ap.add_argument("--sens", type=int, default=60)
    ap.add_argument("--blocks", type=int, nargs="*", default=[8, 32, 128, 512])
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--strides", type=int, nargs="*", default=[1, 4, 16, 64, 192, 512])
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.05, 0.01, 1e-3, 1e-6, 1e-12])
    ap.add_argument("--eq-draws", type=int, default=8)
    ap.add_argument("--eq-jobs", type=int, default=12)
    ap.add_argument("--margin-rho", type=float, default=0.03)
    ap.add_argument("--margin-gain", type=float, default=0.25)
    a = ap.parse_args(argv)
    cfg = GridConfig(a.config)
    {
        "runs": cmd_runs, "features": cmd_features, "aot": cmd_aot,
        "aottail": cmd_aottail, "certify": cmd_certify,
        "cat": cmd_cat, "battery": cmd_battery, "downstream": cmd_downstream,
        "eqdraws": cmd_eqdraws,
        "labellaw": cmd_labellaw, "rngtax": cmd_rngtax, "all": cmd_all,
    }[a.cmd](cfg, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
