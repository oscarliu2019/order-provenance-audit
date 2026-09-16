"""Generate every number, table and figure the manuscript contains.

The manuscript never hard-codes a numeric value: it uses LaTeX macros from
``paper/numbers.tex``, which this script writes together with a machine-readable
``paper/numbers.json``. ``tools/verify_paper_numbers.py`` recomputes each value
from the artefacts and compares, so a stale number cannot survive a build.

Usage
-----
    python tools/paper_assets.py numbers
    python tools/paper_assets.py tables
    python tools/paper_assets.py figs
    python tools/paper_assets.py all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402
from src.detect import aot  # noqa: E402
from src.winerr import load_series  # noqa: E402

ALPHA = 0.05


def art(cfg: GridConfig) -> Path:
    return cfg.path("artifacts_dir")


def paper(cfg: GridConfig) -> Path:
    from src.config import REPO_ROOT

    p = REPO_ROOT / "paper"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read(cfg: GridConfig, name: str) -> pd.DataFrame:
    return pd.read_csv(art(cfg) / name)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def collect_numbers(cfg: GridConfig) -> dict[str, object]:
    runs = _read(cfg, "runs.csv")
    seq = _read(cfg, "aot_sequences.csv")
    cal = _read(cfg, "aot_calibration.csv")
    sens = _read(cfg, "aot_sensitivity.csv")
    theory = _read(cfg, "aot_theory.csv")
    cert = _read(cfg, "certification.csv")
    catb = _read(cfg, "cat_blocks.csv")
    bat = _read(cfg, "battery.csv")
    t1 = _read(cfg, "t1_difficulty.csv")
    t2 = _read(cfg, "t2_selection.csv")
    tax = _read(cfg, "rng_tax.csv")
    eq_rho = _read(cfg, "equivalence_t1_rho.csv").iloc[0]
    eq_tr = _read(cfg, "equivalence_t1_transfer.csv").iloc[0]
    eq_gain = _read(cfg, "equivalence_t2_gain.csv").iloc[0]

    n: dict[str, object] = {}

    # ---- study scale ---- #
    n["NumRuns"] = int(len(runs))
    n["NumRunsDet"] = int((runs["val_order"] == "deterministic").sum())
    n["NumRunsShuf"] = int((runs["val_order"] == "shuffled").sum())
    n["NumDatasets"] = int(runs["dataset"].nunique())
    n["NumBackbones"] = int(runs["backbone"].nunique())
    n["NumArms"] = int(runs["plugin"].nunique())
    n["NumBlocks"] = int(runs["block_id"].nunique())
    n["NumGpuHours"] = round(float(runs["train_seconds"].sum()) / 3600.0, 1)
    n["NumTrainedEpochs"] = int(runs["epochs_run"].sum())

    clean = seq[seq["variant"] == "index_order"]
    dirty = seq[seq["variant"] == "full_permutation"]
    obs = seq[seq["variant"] == "observed_loader_order"]
    n["NumSequences"] = int(len(clean))
    n["NumWindowsAudited"] = int(clean["n"].sum())
    n["NumWindowsMin"] = int(clean["n"].min())
    n["NumWindowsMax"] = int(clean["n"].max())

    # ---- masking ---- #
    mask = masking_check(cfg, runs)
    n.update(mask)

    # ---- AOT ---- #
    n["NumRMedianClean"] = round(float(clean["r"].median()), 4)
    n["NumRMinClean"] = round(float(clean["r"].min()), 4)
    n["NumRMedianCorrupt"] = round(float(dirty["r"].median()), 4)
    n["NumRMaxCorrupt"] = round(float(dirty["r"].max()), 4)
    n["NumZMedianClean"] = round(float(clean["z"].median()), 1)
    n["NumZMinClean"] = round(float(clean["z"].min()), 1)
    n["NumZMaxCorrupt"] = round(float(dirty["z"].max()), 2)
    n["NumCleanCertPct"] = round(float(100 * (clean["p"] < ALPHA).mean()), 1)
    n["NumCorruptCertPct"] = round(float(100 * (dirty["p"] < ALPHA).mean()), 2)
    if len(obs):
        n["NumObservedSeq"] = int(len(obs))
        n["NumObservedRMedian"] = round(float(obs["r"].median()), 4)
        n["NumObservedCertPct"] = round(float(100 * (obs["p"] < ALPHA).mean()), 2)
    for a, tag in ((1e-3, "AlphaThree"), (1e-6, "AlphaSix"), (1e-12, "AlphaTwelve")):
        row = cert[np.isclose(cert["alpha"], a)]
        if len(row):
            n[f"NumCleanCert{tag}"] = round(float(row["clean_certified_pct"].iloc[0]), 1)
            n[f"NumCorruptCert{tag}"] = round(float(row["corrupted_certified_pct"].iloc[0]), 2)

    n["NumCalibSeq"] = int(len(cal))
    n["NumCalibMaxAbsDiff"] = round(float((cal["p_normal"] - cal["p_perm"]).abs().max()), 4)
    n["NumCalibFprNormal"] = round(float(100 * (cal["p_normal"] < ALPHA).mean()), 2)
    n["NumCalibFprPerm"] = round(float(100 * (cal["p_perm"] < ALPHA).mean()), 2)

    blk = sens[sens["kind"] == "block"]
    par = sens[sens["kind"] == "partial"]
    n["NumSensSeq"] = int(sens["cell_id"].nunique())
    if len(blk):
        big = blk[blk["param"] == blk["param"].max()]
        n["NumSensBlockMax"] = int(big["param"].iloc[0])
        n["NumSensBlockMaxRate"] = round(float(100 * big["reject"].mean()), 1)
    if len(par):
        small = par[par["param"] == par["param"].min()]
        n["NumSensPartialMin"] = round(float(100 * small["param"].iloc[0]), 1)
        n["NumSensPartialMinRate"] = round(float(100 * small["reject"].mean()), 1)
        full = par[np.isclose(par["param"], 1.0)]
        if len(full):
            n["NumSensPartialFullRate"] = round(float(100 * full["reject"].mean()), 1)
    for nn, tag in ((100, "Hundred"), (1000, "Thousand")):
        row = theory[theory["n"] == nn]
        if len(row):
            n[f"NumRhoMinN{tag}"] = round(float(row["rho_min_detectable"].iloc[0]), 4)

    # ---- CAT ---- #
    cc = catb[catb["variant"] == "index_order"]
    cd = catb[catb["variant"] == "full_permutation"]
    pairs = catb[catb["variant"].str.startswith("pair:")]
    n["NumCatBlocks"] = int(len(cc))
    n["NumCatRhoMedianClean"] = round(float(cc["rho_bar"].median()), 4)
    n["NumCatRhoMinClean"] = round(float(cc["rho_bar"].min()), 4)
    n["NumCatRhoMedianCorrupt"] = round(float(cd["rho_bar"].median()), 4)
    n["NumCatCertCleanPct"] = round(float(100 * (cc["p"] < ALPHA).mean()), 1)
    n["NumCatCertCorruptPct"] = round(float(100 * (cd["p"] < ALPHA).mean()), 2)
    n["NumCatPairs"] = int(len(pairs))
    n["NumCatPairRhoMedian"] = round(float(pairs["rho_bar"].median()), 4)

    # ---- battery ---- #
    if len(bat):
        bi = bat[bat["variant"] == "index_order"]
        bd = bat[bat["variant"] == "full_permutation"]
        s1 = bi[bi["stride"] == 1]
        n["NumBatAotStrideOne"] = round(float(100 * s1["aot_certify_rate"].mean()), 1)
        n["NumBatCatStrideOne"] = round(float(100 * s1["cat_certify"].mean()), 1)
        dd = bi[bi["overlap_ratio"] <= 0.0]
        if len(dd):
            n["NumBatStrideDisjoint"] = int(dd["stride"].min())
            n["NumBatDisjointRows"] = int(len(dd))
            n["NumBatAotDisjoint"] = round(float(100 * dd["aot_certify_rate"].mean()), 1)
            n["NumBatCatDisjoint"] = round(float(100 * dd["cat_certify"].mean()), 1)
            n["NumBatNDisjoint"] = int(round(dd["n"].mean()))
        n["NumBatCorruptAotMax"] = round(float(100 * bd["aot_certify_rate"].max()), 1)
        n["NumBatCorruptCatMax"] = round(float(100 * bd["cat_certify"].max()), 1)
        n["NumBatCorruptAotPct"] = round(float(100 * bd["aot_certify_rate"].mean()), 1)
        n["NumBatCorruptCatPct"] = round(float(100 * bd["cat_certify"].mean()), 1)
        n["NumBatGroups"] = int(len(s1))
        n["NumBatRows"] = int(len(bat))

    # ---- T1 ---- #
    n["NumTaskOneCells"] = int(t1["cell_id"].nunique())
    for cond, tag in (("index", "Index"), ("defect", "Defect"), ("control", "Control")):
        g = t1[t1["condition"] == cond]
        n[f"NumTaskOneRho{tag}Mean"] = round(float(g["assoc_rho_fstd"].mean()), 4)
        n[f"NumTaskOneRho{tag}Median"] = round(float(g["assoc_rho_fstd"].median()), 4)
        n[f"NumTaskOneRhoBest{tag}Median"] = round(float(g["assoc_rho_best"].abs().median()), 4)
        n[f"NumTaskOneDecile{tag}Median"] = round(float(g["decile_ratio_fstd"].median()), 3)
        n[f"NumTaskOneSig{tag}Pct"] = round(float(100 * (g["assoc_p_fstd"] < ALPHA).mean()), 1)
        n[f"NumTaskOneTransfer{tag}Mean"] = round(float(g["spearman_test"].mean()), 4)
        n[f"NumTaskOneTransfer{tag}Median"] = round(float(g["spearman_test"].median()), 4)
        n[f"NumTaskOneHoldout{tag}Median"] = round(float(g["spearman_holdout"].median()), 4)
    n["NumTaskOneTostMargin"] = float(eq_rho["margin"])
    n["NumTaskOneTostMeanDiff"] = round(float(eq_rho["mean_diff"]), 5)
    n["NumTaskOneTostCiLow"] = round(float(eq_rho["ci95_low"]), 5)
    n["NumTaskOneTostCiHigh"] = round(float(eq_rho["ci95_high"]), 5)
    n["NumTaskOneTostP"] = float(f"{eq_rho['p_tost']:.3g}")
    n["NumTaskOneTransferTostMargin"] = float(eq_tr["margin"])
    n["NumTaskOneTransferTostCiLow"] = round(float(eq_tr["ci95_low"]), 4)
    n["NumTaskOneTransferTostCiHigh"] = round(float(eq_tr["ci95_high"]), 4)

    # ---- T2 ---- #
    n["NumTaskTwoBlocks"] = int(len(t2[t2["condition"] == "index"]))
    for cond, tag in (("index", "Index"), ("defect", "Defect"), ("control", "Control")):
        g = t2[t2["condition"] == cond]
        n[f"NumTaskTwoGain{tag}Median"] = round(float(g["rel_gain_vs_best_fixed"].median()), 3)
        n[f"NumTaskTwoGain{tag}Mean"] = round(float(g["rel_gain_vs_best_fixed"].mean()), 3)
        n[f"NumTaskTwoGain{tag}PosPct"] = round(
            float(100 * (g["rel_gain_vs_best_fixed"] > 0).mean()), 1
        )
    idx = t2[t2["condition"] == "index"]["rel_gain_vs_best_fixed"].to_numpy()
    n["NumTaskTwoOracleHeadroomMedian"] = round(
        float(t2[t2["condition"] == "index"]["oracle_headroom_pct"].median()), 2
    )
    n["NumTaskTwoSignTestP"] = float(f"{stats.wilcoxon(idx).pvalue:.3g}")
    n["NumTaskTwoTostCiLow"] = round(float(eq_gain["ci95_low"]), 3)
    n["NumTaskTwoTostCiHigh"] = round(float(eq_gain["ci95_high"]), 3)

    # ---- RNG tax ---- #
    n["NumTaxPairs"] = int(len(tax))
    n["NumTaxMedianAbs"] = round(float(tax["abs_rel_diff_pct"].median()), 3)
    n["NumTaxPNinetyAbs"] = round(float(tax["abs_rel_diff_pct"].quantile(0.9)), 3)
    n["NumTaxMaxAbs"] = round(float(tax["abs_rel_diff_pct"].max()), 3)
    n["NumTaxSameEpochPct"] = round(float(100 * tax["same_best_epoch"].mean()), 1)
    n["NumTaxOverTenthPct"] = round(float(100 * (tax["abs_rel_diff_pct"] > 0.1).mean()), 1)
    n["NumTaxWilcoxP"] = float(
        f"{stats.wilcoxon(tax['mse_shuffled'], tax['mse_deterministic']).pvalue:.3g}"
    )
    n["NumTaxIdenticalPct"] = round(float(100 * (tax["abs_rel_diff_pct"] < 1e-9).mean()), 1)
    return n


def masking_check(cfg: GridConfig, runs: pd.DataFrame) -> dict[str, object]:
    """Verify on real artefacts that the permutation leaves the aggregate intact."""
    rel, same_multiset, same_len, ncheck = [], [], [], 0
    for _, r in runs[runs["val_order"] == "shuffled"].iterrows():
        d = cfg.path("winerr_dir") / r["cell_id"]
        try:
            rec = load_series(d, "val_loader_order", require_order="loader")
            idx = load_series(d, "val_index_order", require_order="index")
        except FileNotFoundError:
            continue
        ncheck += 1
        rel.append(abs(rec.mean() - idx.mean()) / abs(idx.mean()))
        same_multiset.append(bool(np.array_equal(np.sort(rec), np.sort(idx))))
        same_len.append(rec.size == idx.size)
    if not ncheck:
        return {"NumMaskChecked": 0}
    return {
        "NumMaskChecked": int(ncheck),
        "NumMaskMaxRelMeanDiff": float(f"{max(rel):.2g}"),
        "NumMaskMultisetPct": round(float(100 * np.mean(same_multiset)), 1),
        "NumMaskLengthPct": round(float(100 * np.mean(same_len)), 1),
    }


def fmt(v: object) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", "\\,")
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e6):
            m, e = f"{v:.2e}".split("e")
            return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def cmd_numbers(cfg: GridConfig, args) -> None:
    n = collect_numbers(cfg)
    (paper(cfg) / "numbers.json").write_text(json.dumps(n, indent=2, sort_keys=True))
    lines = [
        "% Generated by tools/paper_assets.py numbers -- do not edit by hand.",
        "% Every numeric claim in main.tex expands one of these macros.",
        "",
    ]
    for k in sorted(n):
        lines.append(f"\\newcommand{{\\{k}}}{{{fmt(n[k])}}}")
    (paper(cfg) / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"[numbers] {len(n)} macros -> paper/numbers.tex")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _tabular(header: list[str], rows: list[list[str]], align: str) -> str:
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def cmd_tables(cfg: GridConfig, args) -> None:
    out = paper(cfg) / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # T1: certification operating characteristics
    cert = _read(cfg, "certification.csv")
    rows = []
    for _, r in cert.iterrows():
        a = r["alpha"]
        atxt = f"{a:.2f}" if a >= 0.01 else f"$10^{{{int(round(np.log10(a)))}}}$"
        rows.append(
            [atxt, f"{r['clean_certified_pct']:.1f}", f"{r['corrupted_certified_pct']:.2f}"]
        )
    (out / "certification.tex").write_text(
        _tabular(
            [r"$\alpha$", r"index-order certified (\%)", r"permuted certified (\%)"],
            rows,
            "lrr",
        )
    )

    # T2: sensitivity to milder corruptions
    sens = _read(cfg, "aot_sensitivity.csv")
    rows = []
    for kind, label in (("block", "block-order shuffle, block size"),
                        ("partial", "partial shuffle, fraction moved")):
        g = sens[sens["kind"] == kind]
        for p, gg in g.groupby("param"):
            ptxt = f"{int(p)}" if kind == "block" else f"{p:.2f}"
            rows.append(
                [label, ptxt, f"{len(gg)}", f"{100 * (1 - gg['reject'].mean()):.1f}"]
            )
    (out / "sensitivity.tex").write_text(
        _tabular(
            ["corruption", "parameter", "sequences", r"detected (\%)"],
            [[r[0], r[1], r[2], f"{100 - float(r[3]):.1f}"] for r in rows],
            "llrr",
        )
    )

    # T3: the battery against loss of serial structure
    bat = _read(cfg, "battery.csv")
    bi = bat[bat["variant"] == "index_order"]
    bd = bat[bat["variant"] == "full_permutation"]
    rows = []
    for s in sorted(bi["stride"].unique()):
        gi = bi[bi["stride"] == s]
        gd = bd[bd["stride"] == s]
        rows.append(
            [
                f"{int(s)}",
                f"{gi['overlap_ratio'].mean():.2f}",
                f"{int(round(gi['n'].mean()))}",
                f"{100 * gi['aot_certify_rate'].mean():.1f}",
                f"{100 * gd['aot_certify_rate'].mean():.1f}",
                f"{100 * gi['cat_certify'].mean():.1f}",
                f"{100 * gd['cat_certify'].mean():.1f}",
            ]
        )
    (out / "battery.tex").write_text(
        _tabular(
            ["stride", "overlap", "$n$", "AOT clean", "AOT perm.", "CAT clean", "CAT perm."],
            rows,
            "rrrrrrr",
        )
    )

    # T4: downstream collapse
    t1 = _read(cfg, "t1_difficulty.csv")
    metrics = [
        ("assoc_rho_fstd", r"$\rho$(volatility, error)", 3),
        ("assoc_rho_best", r"$\rho$ of the best feature", 3),
        ("decile_ratio_fstd", "top/bottom decile error ratio", 2),
        ("spearman_test", "transfer rank agreement on test", 3),
    ]
    rows = []
    for col, label, dec in metrics:
        cells = [label]
        for cond in ("index", "defect", "control"):
            g = t1[t1["condition"] == cond][col]
            cells.append(f"{g.median():.{dec}f}")
        rows.append(cells)
    g = t1.groupby("condition")["assoc_p_fstd"]
    rows.append(
        [r"significant association at $\alpha=0.05$ (\%)"]
        + [f"{100 * (t1[t1['condition'] == c]['assoc_p_fstd'] < ALPHA).mean():.1f}"
           for c in ("index", "defect", "control")]
    )
    (out / "downstream_t1.tex").write_text(
        _tabular(["statistic (median over runs)", "index order", "defect", "control"],
                 rows, "lrrr")
    )

    # T5: arm selection
    t2 = _read(cfg, "t2_selection.csv")
    rows = []
    for col, label, dec in (
        ("rel_gain_vs_best_fixed", r"gain over best fixed arm (\%)", 2),
        ("oracle_headroom_pct", r"per-window oracle headroom (\%)", 2),
        ("oracle_match_acc", "agreement with the per-window oracle", 3),
    ):
        cells = [label]
        for cond in ("index", "defect", "control"):
            cells.append(f"{t2[t2['condition'] == cond][col].median():.{dec}f}")
        rows.append(cells)
    rows.append(
        [r"blocks with a positive gain (\%)"]
        + [f"{100 * (t2[t2['condition'] == c]['rel_gain_vs_best_fixed'] > 0).mean():.1f}"
           for c in ("index", "defect", "control")]
    )
    (out / "downstream_t2.tex").write_text(
        _tabular(["statistic (median over blocks)", "index order", "defect", "control"],
                 rows, "lrrr")
    )

    # T6: RNG coupling tax
    tax = _read(cfg, "rng_tax.csv")
    rows = []
    for ds, g in tax.groupby("dataset"):
        rows.append(
            [
                ds,
                f"{len(g)}",
                f"{g['abs_rel_diff_pct'].median():.3f}",
                f"{g['abs_rel_diff_pct'].max():.3f}",
                f"{100 * g['same_best_epoch'].mean():.0f}",
            ]
        )
    rows.append(
        [
            r"\textbf{all}",
            f"{len(tax)}",
            f"{tax['abs_rel_diff_pct'].median():.3f}",
            f"{tax['abs_rel_diff_pct'].max():.3f}",
            f"{100 * tax['same_best_epoch'].mean():.0f}",
        ]
    )
    (out / "rng_tax.tex").write_text(
        _tabular(
            ["dataset", "pairs", r"median $|\Delta|$ (\%)", r"max $|\Delta|$ (\%)",
             r"same best epoch (\%)"],
            rows,
            "lrrrr",
        )
    )

    # T7: grid summary
    runs = _read(cfg, "runs.csv")
    rows = []
    for bk, g in runs.groupby("backbone"):
        rows.append(
            [
                bk,
                f"{g['dataset'].nunique()}",
                f"{g['pred_len'].nunique()}",
                f"{g['plugin'].nunique()}",
                f"{len(g)}",
                f"{g['train_seconds'].sum() / 3600:.1f}",
            ]
        )
    rows.append(
        [
            r"\textbf{all}",
            f"{runs['dataset'].nunique()}",
            f"{runs['pred_len'].nunique()}",
            f"{runs['plugin'].nunique()}",
            f"{len(runs)}",
            f"{runs['train_seconds'].sum() / 3600:.1f}",
        ]
    )
    (out / "grid.tex").write_text(
        _tabular(
            ["backbone", "datasets", "horizons", "arms", "runs", "GPU hours"],
            rows,
            "lrrrrr",
        )
    )
    print(f"[tables] 7 tables -> {out}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def cmd_figs(cfg: GridConfig, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )
    figs = paper(cfg) / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    runs = _read(cfg, "runs.csv")

    # ---- fig 1: a real sequence, both orders ---- #
    pick = runs[(runs["backbone"] == "DLinear") & (runs["dataset"] == "ETTh1")
                & (runs["pred_len"] == 96) & (runs["plugin"] == "none")]
    if not len(pick):
        pick = runs
    cell = pick.iloc[0]["cell_id"]
    d = cfg.path("winerr_dir") / cell
    y = load_series(d, "val_index_order")
    perm = np.random.default_rng(0).permutation(y.size)
    yp = y[perm]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    ax[0].plot(y, lw=0.4, color="#1f4e79")
    ax[0].set_title(f"index order  ($r={aot(y)['r']:.3f}$)")
    ax[0].set_xlabel("window index")
    ax[0].set_ylabel("per-window MSE")
    ax[1].plot(yp, lw=0.4, color="#a33")
    ax[1].set_title(f"loader order  ($r={aot(yp)['r']:.3f}$)")
    ax[1].set_xlabel("loader position")
    ax[2].scatter(y[:-1], y[1:], s=0.7, alpha=0.35, color="#1f4e79", label="index order")
    ax[2].scatter(yp[:-1], yp[1:], s=0.7, alpha=0.35, color="#a33", label="loader order")
    ax[2].set_xlabel("error at position $i$")
    ax[2].set_ylabel("error at position $i+1$")
    ax[2].legend(markerscale=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "sequence.pdf")
    plt.close(fig)

    # ---- fig 2: separation of the statistic ---- #
    seq = _read(cfg, "aot_sequences.csv")
    clean = seq[seq["variant"] == "index_order"]
    dirty = seq[seq["variant"] == "full_permutation"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax[0].hist(dirty["r"], bins=40, color="#a33", alpha=0.75, label="permuted")
    ax[0].hist(clean["r"], bins=40, color="#1f4e79", alpha=0.75, label="index order")
    ax[0].set_xlabel("circular serial correlation $r$")
    ax[0].set_ylabel("sequences")
    ax[0].legend(frameon=False)
    zc = np.maximum(clean["z"], 1e-3)
    zd = np.maximum(dirty["z"], 1e-3)
    ax[1].hist(np.log10(zd), bins=40, color="#a33", alpha=0.75, label="permuted")
    ax[1].hist(np.log10(zc), bins=40, color="#1f4e79", alpha=0.75, label="index order")
    ax[1].axvline(np.log10(stats.norm.isf(0.05)), color="k", lw=0.8, ls="--")
    ax[1].axvline(np.log10(stats.norm.isf(1e-12)), color="k", lw=0.8, ls=":")
    ax[1].set_xlabel(r"$\log_{10} z$ (dashed $\alpha=0.05$, dotted $\alpha=10^{-12}$)")
    fig.tight_layout()
    fig.savefig(figs / "separation.pdf")
    plt.close(fig)

    # ---- fig 3: the battery against loss of overlap ---- #
    bat = _read(cfg, "battery.csv")
    bi = bat[bat["variant"] == "index_order"].groupby("stride").mean(numeric_only=True)
    bd = bat[bat["variant"] == "full_permutation"].groupby("stride").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(bi.index, 100 * bi["aot_certify_rate"], "o-", color="#1f4e79", label="AOT, clean")
    ax.plot(bi.index, 100 * bi["cat_certify"], "s-", color="#2e7d32", label="CAT, clean")
    ax.plot(bd.index, 100 * bd["aot_certify_rate"], "o--", color="#a33", label="AOT, permuted")
    ax.plot(bd.index, 100 * bd["cat_certify"], "s--", color="#e08a1e", label="CAT, permuted")
    ax.set_xscale("log")
    ax.set_xlabel("sub-sampling stride (windows)")
    ax.set_ylabel(r"certified at $\alpha=0.05$ (\%)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(figs / "battery.pdf")
    plt.close(fig)

    # ---- fig 4: downstream collapse ---- #
    t1 = _read(cfg, "t1_difficulty.csv")
    piv = t1.pivot_table(index="cell_id", columns="condition", values="assoc_rho_fstd")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    parts = [piv["index"].dropna(), piv["defect"].dropna(), piv["control"].dropna()]
    ax[0].violinplot(parts, showmedians=True)
    ax[0].set_xticks([1, 2, 3])
    ax[0].set_xticklabels(["index", "defect", "control"])
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel(r"$\rho$(volatility, per-window error)")
    dec = t1.pivot_table(index="cell_id", columns="condition", values="decile_ratio_fstd")
    ax[1].violinplot(
        [dec["index"].dropna(), dec["defect"].dropna(), dec["control"].dropna()],
        showmedians=True,
    )
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels(["index", "defect", "control"])
    ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_ylabel("top/bottom decile error ratio")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(figs / "downstream.pdf")
    plt.close(fig)

    # ---- fig 5: the repair is not free ---- #
    tax = _read(cfg, "rng_tax.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax[0].scatter(tax["mse_deterministic"], tax["mse_shuffled"], s=6, alpha=0.6,
                  color="#1f4e79")
    lim = [min(tax["mse_deterministic"].min(), tax["mse_shuffled"].min()) * 0.95,
           max(tax["mse_deterministic"].max(), tax["mse_shuffled"].max()) * 1.05]
    ax[0].plot(lim, lim, "k--", lw=0.7)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("test MSE, deterministic validation loader")
    ax[0].set_ylabel("test MSE, shuffled validation loader")
    ax[1].hist(tax["rel_diff_pct"], bins=40, color="#1f4e79", alpha=0.85)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel(r"relative change in reported test MSE (\%)")
    ax[1].set_ylabel("paired cells")
    fig.tight_layout()
    fig.savefig(figs / "rngtax.pdf")
    plt.close(fig)
    print(f"[figs] 5 figures -> {figs}")


def cmd_all(cfg: GridConfig, args) -> None:
    cmd_numbers(cfg, args)
    cmd_tables(cfg, args)
    cmd_figs(cfg, args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["numbers", "tables", "figs", "all"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = GridConfig(a.config)
    {"numbers": cmd_numbers, "tables": cmd_tables, "figs": cmd_figs, "all": cmd_all}[a.cmd](
        cfg, a
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
