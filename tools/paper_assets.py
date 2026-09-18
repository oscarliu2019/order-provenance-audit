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
from src.winerr import load_series as _load_series  # noqa: E402


def load_series(win_dir, name, require_order="index", **kw):
    """Legacy-tolerant read: the archived cells predate the provenance contract."""
    return _load_series(win_dir, name, require_order, legacy_ok=True, **kw)

ALPHA = 0.05

# the six controlled sharing conditions of artifacts/cat_conditions.csv
CAT_CONDITIONS = [
    ("intact", "Intact", "intact (no permutation injected)"),
    ("independent", "Independent", "independent permutation per arm"),
    ("all_shared", "AllShared", "one permutation shared by all four arms"),
    ("shared_k3", "SharedKThree", "three arms share, one independent"),
    ("shared_k2", "SharedKTwo", "two arms share, two independent"),
    ("two_pairs", "TwoPairs", "two disjoint pairs, each sharing"),
]

# statistics of artifacts/t1_cluster_summary.csv exported as cluster intervals
CLUSTER_STATS = [
    ("index_median_rho", "DepRho", 1.0, 4),
    ("defect_minus_control_median", "DepDefectMinusControl", 1.0, 6),
    ("index_frac_p_iid_lt_05", "DepIidSig", 100.0, 2),
    ("index_frac_p_block_lt_05", "DepBlockSig", 100.0, 2),
    ("index_frac_ci_excludes_zero", "DepBootCiExcl", 100.0, 2),
]


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
    eqd = _read(cfg, "equivalence_draws.csv").set_index("stat")

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
    _sets = clean.drop_duplicates(subset=["dataset", "pred_len", "split"])
    n["NumSampleSets"] = int(len(_sets))
    n["NumDistinctWindows"] = int(_sets["n"].sum())
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
    n["NumZMinCleanN"] = int(clean.loc[clean["z"].idxmin(), "n"])
    long_clean = clean[clean["n"] >= 500]
    if len(long_clean):
        n["NumZMinCleanLong"] = round(float(long_clean["z"].min()), 1)
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

    # sequences the strictest level fails to certify are the shortest ones
    strict = clean[clean["p"] >= 1e-12]
    n["NumCertTwelveMissSeq"] = int(len(strict))
    n["NumCertTwelveMissMaxN"] = int(strict["n"].max()) if len(strict) else 0

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
    # ---- measured tail behaviour of the normal reference ---- #
    tail = _read(cfg, "aot_tail.csv")
    if len(tail):
        n["NumTailSeq"] = int(tail["cell_id"].nunique())
        n["NumTailPerm"] = int(tail["n_perm"].iloc[0])
        n["NumTailNullDraws"] = int(tail["n_perm"].iloc[0] * tail["cell_id"].nunique())
        n["NumTailKurtMax"] = round(float(tail["kurtosis"].max()), 1)
        n["NumTailMaxZ"] = round(float(tail["max_z_observed"].max()), 2)
        for a, tag in ((1e-2, "Two"), (1e-3, "Three"), (1e-4, "Four"), (1e-5, "Five")):
            g = tail[np.isclose(tail["alpha"], a)]
            if len(g):
                n[f"NumTailRatio{tag}"] = round(float(g["ratio_to_nominal"].max()), 2)
        short = tail[tail["n"] == tail["n"].min()]
        n["NumTailShortN"] = int(short["n"].iloc[0])
        n["NumTailShortMaxZ"] = round(float(short["max_z_observed"].max()), 2)

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
    miss = cc[cc["p"] >= ALPHA]
    n["NumCatMissGroups"] = int(len(miss))
    n["NumCatMissMaxN"] = int(miss["n"].max()) if len(miss) else 0
    n["NumCatMinN"] = int(cc["n"].min())

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
    for tag, key in (("Rho", "assoc_rho_fstd"), ("Transfer", "spearman_test")):
        row = eqd.loc[key]
        n[f"NumEqDraw{tag}SinglePMax"] = float(f"{row['single_p_max']:.3g}")
        n[f"NumEqDraw{tag}SinglePMedian"] = float(f"{row['single_p_median']:.3g}")
        n[f"NumEqDraw{tag}CertPct"] = round(100.0 * float(row["single_certify_frac"]), 1)
        n[f"NumEqDraw{tag}AbsDiffMax"] = round(float(row["single_absdiff_max"]), 4)
        n[f"NumEqDraw{tag}MeanDiff"] = round(float(row["mean_diff"]), 4)
        n[f"NumEqDraw{tag}CiLow"] = round(float(row["mean_ci95_low"]), 4)
        n[f"NumEqDraw{tag}CiHigh"] = round(float(row["mean_ci95_high"]), 4)
        n[f"NumEqDraw{tag}P"] = float(f"{row['mean_p_tost']:.3g}")
    n["NumEqDraws"] = int(eqd["draws"].iloc[0])
    n["NumTaskOneTostClusters"] = int(eq_rho["clustered_n_clusters"])
    n["NumTaskOneTostClusterCiLow"] = round(float(eq_rho["clustered_ci95_low"]), 5)
    n["NumTaskOneTostClusterCiHigh"] = round(float(eq_rho["clustered_ci95_high"]), 5)
    n["NumTaskOneTostClusterP"] = float(f"{eq_rho['clustered_p_tost']:.3g}")
    n["NumTaskOneTransferTostMargin"] = float(eq_tr["margin"])
    n["NumTaskOneTransferTostCiLow"] = round(float(eq_tr["ci95_low"]), 4)
    n["NumTaskOneTransferTostCiHigh"] = round(float(eq_tr["ci95_high"]), 4)
    n["NumTaskOneTransferTostN"] = int(eq_tr["n"])
    n["NumTaskOneTransferTostMeanDiff"] = round(float(eq_tr["mean_diff"]), 5)
    n["NumTaskOneTransferTostP"] = float(f"{eq_tr['p_tost']:.3g}")
    n["NumTaskOneTransferTostClusterCiLow"] = round(float(eq_tr["clustered_ci95_low"]), 4)
    n["NumTaskOneTransferTostClusterCiHigh"] = round(float(eq_tr["clustered_ci95_high"]), 4)
    n["NumTaskOneTransferTostClusterP"] = float(f"{eq_tr['clustered_p_tost']:.3g}")

    # ---- T2 ---- #
    n["NumTaskTwoBlocks"] = int(len(t2[t2["condition"] == "index"]))
    for cond, tag in (("index", "Index"), ("defect", "Defect"), ("control", "Control")):
        g = t2[t2["condition"] == cond]
        n[f"NumTaskTwoGain{tag}Median"] = round(float(g["rel_gain_vs_best_fixed"].median()), 3)
        n[f"NumTaskTwoGain{tag}Mean"] = round(float(g["rel_gain_vs_best_fixed"].mean()), 3)
        n[f"NumTaskTwoGain{tag}PosPct"] = round(
            float(100 * (g["rel_gain_vs_best_fixed"] > 0).mean()), 1
        )
    idx_rows = t2[t2["condition"] == "index"]
    idx = idx_rows["rel_gain_vs_best_fixed"].to_numpy()
    n["NumTaskTwoOracleHeadroomMedian"] = round(
        float(t2[t2["condition"] == "index"]["oracle_headroom_pct"].median()), 2
    )
    n["NumTaskTwoSignTestP"] = float(f"{stats.wilcoxon(idx).pvalue:.3g}")
    n["NumTaskTwoTostCiLow"] = round(float(eq_gain["ci95_low"]), 3)
    n["NumTaskTwoTostCiHigh"] = round(float(eq_gain["ci95_high"]), 3)
    n["NumTaskTwoTostMargin"] = float(eq_gain["margin"])
    n["NumTaskTwoTostMeanDiff"] = round(float(eq_gain["mean_diff"]), 3)
    n["NumTaskTwoTostClusterCiLow"] = round(float(eq_gain["clustered_ci95_low"]), 3)
    n["NumTaskTwoTostClusterCiHigh"] = round(float(eq_gain["clustered_ci95_high"]), 3)
    n["NumTaskTwoTostClusterP"] = float(f"{eq_gain['clustered_p_tost']:.3g}")
    n["NumTaskTwoRandomPenaltyMedian"] = round(
        float(
            (
                100
                * (idx_rows["random_arm_mse"] - idx_rows["best_fixed_mse"])
                / idx_rows["best_fixed_mse"]
            ).median()
        ),
        2,
    )

    # ---- label law behind the T2 gap ---- #
    lab = _read(cfg, "label_law.csv")
    n["NumLabelBlocks"] = int(len(lab[lab["condition"] == "index"]))
    n["NumLabelEntropyMax"] = round(float(np.log2(lab["n_arms"].max())), 2)
    for cond, tag in (("index", "Index"), ("defect", "Defect")):
        g = lab[lab["condition"] == cond]
        n[f"NumLabelModal{tag}Pct"] = round(float(100 * g["modal_share"].mean()), 1)
        n[f"NumLabelBestFixedShare{tag}Pct"] = round(float(100 * g["best_fixed_share"].mean()), 1)
        n[f"NumLabelEntropy{tag}"] = round(float(g["entropy_bits"].mean()), 3)

    # ---- RNG tax ---- #
    n["NumTaxPairs"] = int(len(tax))
    n["NumTaxMedianAbs"] = round(float(tax["abs_rel_diff_pct"].median()), 3)
    n["NumTaxPNinetyAbs"] = round(float(tax["abs_rel_diff_pct"].quantile(0.9)), 3)
    n["NumTaxMaxAbs"] = round(float(tax["abs_rel_diff_pct"].max()), 3)
    n["NumTaxSameEpochPct"] = round(float(100 * tax["same_best_epoch"].mean()), 1)
    n["NumTaxSameEpochMedian"] = round(
        float(tax.loc[tax["same_best_epoch"] == 1, "abs_rel_diff_pct"].median()), 4
    )
    n["NumTaxDiffEpochMedian"] = round(
        float(tax.loc[tax["same_best_epoch"] == 0, "abs_rel_diff_pct"].median()), 4
    )
    n["NumTaxDiffEpochPairs"] = int((tax["same_best_epoch"] == 0).sum())
    n["NumTaxOverTenthPct"] = round(float(100 * (tax["abs_rel_diff_pct"] > 0.1).mean()), 1)
    n["NumTaxWilcoxP"] = float(
        f"{stats.wilcoxon(tax['mse_shuffled'], tax['mse_deterministic']).pvalue:.3g}"
    )
    n["NumTaxIdenticalPct"] = round(float(100 * (tax["abs_rel_diff_pct"] < 1e-9).mean()), 1)
    n.update(revision_numbers(cfg, sens, tail))
    return n


def _pytest_counts() -> dict[str, int]:
    """Collect-only run of the suite: total tests and the adversarial contract file."""
    import re
    import subprocess

    from src.config import REPO_ROOT

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if m is None:
        raise RuntimeError("could not read the pytest collection summary")
    adv = sum(1 for ln in out.splitlines() if ln.startswith("tests/test_provenance.py::"))
    return {"total": int(m.group(1)), "adversarial": adv}


def rnd(x: float, k: int) -> float:
    """Half-up rounding, so a .5 boundary reads the way a reader rounds it."""
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-k), rounding=ROUND_HALF_UP))


def revision_numbers(cfg: GridConfig, sens: pd.DataFrame, tail: pd.DataFrame) -> dict[str, object]:
    """Macros added by the revision: A1-A10 of the revision plan."""
    n: dict[str, object] = {}
    obs = _read(cfg, "observed_perm_scan.csv")
    cc = _read(cfg, "cat_conditions.csv")
    ca = _read(cfg, "cat_analytic_check.csv")
    dep = _read(cfg, "t1_difficulty_depaware.csv")
    clus = _read(cfg, "t1_cluster_summary.csv")
    pur = _read(cfg, "t1_purge_audit.csv")
    cov = _read(cfg, "contract_coverage.csv")
    mig = _read(cfg, "backfill_provenance_migration_report.csv")
    summ = json.loads((art(cfg) / "t1_depaware_summary.json").read_text())

    # ---- A1: sensitivity, in the flagged direction ---- #
    # reject=1 rejects "uniformly permuted", i.e. the sequence still looks ordered
    per = 100.0 * (1.0 - sens.groupby(["kind", "param"])["reject"].mean())
    blk_rate = per.loc["block"]
    par_rate = per.loc["partial"]
    n["NumSensNBlockRows"] = int((sens["kind"] == "block").sum())
    n["NumSensNPartialRows"] = int((sens["kind"] == "partial").sum())
    n["NumSensFlagBlockMax"] = round(float(blk_rate.max()), 2)
    n["NumSensFlagPartialMildMax"] = round(float(par_rate[par_rate.index <= 0.10].max()), 2)
    par = sens[sens["kind"] == "partial"]
    for p, tag in ((0.25, "Quarter"), (0.50, "Half"), (1.00, "Full")):
        g = par[np.isclose(par["param"], p)]
        n[f"NumSensFlagPartial{tag}"] = round(float(100 * (1 - g["reject"].mean())), 2)

    # ---- A2: AOT on the real loader permutations ---- #
    n["NumObsPermGroups"] = int(len(obs))
    n["NumObsPermArms"] = int(obs["n_arms_aot"].sum())
    n["NumObsAotMissArms"] = int(obs["n_arms_aot_rejects"].sum())
    n["NumObsAotFlagArms"] = int(n["NumObsPermArms"] - n["NumObsAotMissArms"])
    n["NumObsAotFlagPct"] = rnd(100.0 * n["NumObsAotFlagArms"] / n["NumObsPermArms"], 2)
    n["NumObsAotMissGroups"] = int((obs["n_arms_aot_rejects"] > 0).sum())
    n["NumObsAotPMin"] = float(f"{obs['aot_p_min'].min():.3g}")

    # ---- A3: honest tail calibration of the normal reference ---- #
    n["NumTailRatioSix"] = round(
        float(tail.loc[np.isclose(tail["alpha"], 1e-6), "ratio_to_nominal"].max()), 2
    )
    for a, tag in ((1e-2, "Two"), (1e-3, "Three"), (1e-4, "Four"), (1e-5, "Five"), (1e-6, "Six")):
        g = tail[np.isclose(tail["alpha"], a)]
        n[f"NumTailRatioMedian{tag}"] = round(float(g["ratio_to_nominal"].median()), 4)
    n["NumTailRatioAboveOneRows"] = int((tail["ratio_to_nominal"] > 1.0).sum())
    n["NumTailRatioAboveOnePct"] = round(float(100 * (tail["ratio_to_nominal"] > 1.0).mean()), 1)

    # ---- A4: the sharing structure of the observed permutations ---- #
    share = obs["perm_sharing"].value_counts()
    for key, tag in (
        ("partial_shared", "PartialShared"),
        ("all_identical", "AllIdentical"),
        ("all_distinct", "AllDistinct"),
    ):
        n[f"NumObsPerm{tag}"] = int(share.get(key, 0))
    n["NumObsPermPairs"] = int(obs["n_pairs"].sum())
    ident = obs["n_pairs_identical"].value_counts()
    for k, tag in ((0, "None"), (1, "One"), (2, "Two"), (3, "Three"), (6, "Six")):
        n[f"NumObsPairsIdentical{tag}"] = int(ident.get(k, 0))
    n["NumObsSpearmanMedian"] = round(float(obs["spearman_perm_mean"].median()), 4)
    n["NumObsSpearmanMin"] = round(float(obs["spearman_perm_mean"].min()), 4)
    n["NumObsSpearmanMax"] = round(float(obs["spearman_perm_mean"].max()), 4)
    n["NumObsPermNonIdentityArms"] = int(obs["n_arms_misordered"].sum())
    n["NumObsPermConsistentGroups"] = int(obs["perm_consistent"].sum())
    n["NumObsCatCertGroups"] = int(obs["cat_certified"].sum())
    n["NumObsCatCertPct"] = round(float(100 * obs["cat_certified"].mean()), 2)
    n["NumObsCatRhoMedian"] = round(float(obs["cat_rho_bar"].median()), 4)
    n["NumObsCatCertIntactGroups"] = int(obs["cat_certified_intact"].sum())
    n["NumObsCatCertIntactPct"] = round(float(100 * obs["cat_certified_intact"].mean()), 2)
    n["NumObsCatRhoIntactMedian"] = round(float(obs["cat_rho_bar_intact"].median()), 4)

    # ---- A5: CAT under six controlled sharing conditions ---- #
    n["NumCatCondRows"] = int(len(cc))
    n["NumCatCondGroups"] = int(cc.groupby("condition").size().max())
    for cond, tag, _label in CAT_CONDITIONS:
        g = cc[cc["condition"] == cond]
        n[f"NumCatCond{tag}Cert"] = int(g["certified"].sum())
        n[f"NumCatCond{tag}Rho"] = round(float(g["rho_bar"].median()), 4)

    # ---- A6: the corrected analytic null, on independent permutations only ---- #
    ind = ca[ca["condition"] == "independent"]
    n["NumCatAnalyticRows"] = int(len(ind))
    n["NumCatAnalyticPairs"] = int(ind["m_pairs"].iloc[0])
    n["NumCatAnalyticErrMax"] = round(float(ind["abs_err_corrected"].max()), 4)
    n["NumCatAnalyticErrMedian"] = round(float(ind["abs_err_corrected"].median()), 4)
    n["NumCatAnalyticNaiveErrMax"] = round(float(ind["abs_err_naive"].max()), 4)
    n["NumCatAnalyticNaiveErrMedian"] = round(float(ind["abs_err_naive"].median()), 4)
    n["NumCatSdRatioCorrectedMedian"] = round(float(ind["sd_ratio_corrected"].median()), 4)
    n["NumCatSdRatioNaiveMedian"] = round(float(ind["sd_ratio_naive"].median()), 4)
    n["NumCatSdRatioNaiveTheory"] = round(float(1.0 / np.sqrt(n["NumCatAnalyticPairs"])), 4)

    # ---- A7: dependence-aware downstream association ---- #
    di = dep[dep["condition"] == "index"]
    n["NumDepCells"] = int(di["cell_id"].nunique())
    n["NumDepRhoMedian"] = round(float(di["assoc_rho_fstd"].median()), 4)
    n["NumDepRhoQOne"] = round(float(di["assoc_rho_fstd"].quantile(0.25)), 4)
    n["NumDepRhoQThree"] = round(float(di["assoc_rho_fstd"].quantile(0.75)), 4)
    n["NumDepRhoIqr"] = round(
        float(di["assoc_rho_fstd"].quantile(0.75) - di["assoc_rho_fstd"].quantile(0.25)), 4
    )
    n["NumDepRhoMin"] = round(float(di["assoc_rho_fstd"].min()), 4)
    n["NumDepRhoMax"] = round(float(di["assoc_rho_fstd"].max()), 4)
    n["NumDepBlockSigPct"] = round(float(100 * (di["assoc_p_block"] < ALPHA).mean()), 2)
    n["NumDepBootCiExclPct"] = round(float(100 * di["assoc_rho_ci_excludes_zero"].mean()), 2)
    n["NumDepIidSigPct"] = round(float(100 * (di["assoc_p_iid_naive"] < ALPHA).mean()), 2)
    n["NumDepSequences"] = int(summ["n_sequences_val_plus_test"])
    n["NumDepSampleSets"] = int(summ["n_distinct_sample_sets"])
    n["NumDepSampleSetsVal"] = int(summ["n_distinct_sample_sets_val"])
    n["NumDepDatasets"] = int(summ["n_datasets"])
    n["NumDepBlockLenMedian"] = int(di["block_len"].median())
    n["NumDepBlockLenCappedCells"] = int(di["block_len_capped"].sum())
    n["NumDepNPerm"] = int(di["n_perm"].iloc[0])
    n["NumDepNBoot"] = int(di["n_boot"].iloc[0])
    n["NumDepMinHoldoutWindows"] = int(summ["min_holdout_windows"])

    # ---- A8: cluster bootstrap over sample sets and datasets ---- #
    cl = clus.set_index(["cluster_def", "statistic"])
    n["NumDepClusterBoot"] = int(clus["n_boot"].iloc[0])
    n["NumDepClusterSampleSets"] = int(
        clus.loc[clus["cluster_def"] == "sample_set", "n_clusters"].iloc[0]
    )
    n["NumDepClusterDatasets"] = int(
        clus.loc[clus["cluster_def"] == "dataset", "n_clusters"].iloc[0]
    )
    for cdef, ctag in (("sample_set", "SampleSet"), ("dataset", "Dataset")):
        for stat, tag, scale, dec in CLUSTER_STATS:
            r = cl.loc[(cdef, stat)]
            n[f"Num{tag}{ctag}CiLow"] = round(float(scale * r["ci_lo"]), dec)
            n[f"Num{tag}{ctag}CiHigh"] = round(float(scale * r["ci_hi"]), dec)
    n["NumDepDefectMinusControlMedian"] = round(
        float(cl.loc[("sample_set", "defect_minus_control_median"), "point"]), 6
    )

    # ---- A9: purge audit at the boundary of the held-out half ---- #
    n["NumPurgeCells"] = int(len(pur))
    for mode, tag in (("labels", "Labels"), ("disjoint", "Disjoint")):
        n[f"NumPurge{tag}Eligible"] = int(pur[f"{mode}_holdout_eligible"].sum())
        n[f"NumPurge{tag}Ineligible"] = int(len(pur) - pur[f"{mode}_holdout_eligible"].sum())
    n["NumPurgeBucketSeqLen"] = 96
    n["NumPurgeBucketPredLen"] = 720
    b = pur[
        (pur["seq_len"] == n["NumPurgeBucketSeqLen"])
        & (pur["pred_len"] == n["NumPurgeBucketPredLen"])
        & (pur["labels_holdout_eligible"] == 1)
    ]
    n["NumPurgeBucketCells"] = int(len(b))
    for mode, tag in (("labels", "Labels"), ("disjoint", "Disjoint")):
        n[f"NumPurge{tag}Origins"] = int(b[f"{mode}_purge_origins"].median())
        n[f"NumPurge{tag}RequiredGap"] = int(b[f"{mode}_required_origin_gap"].median())
        n[f"NumPurge{tag}Gap"] = int(b[f"{mode}_min_origin_gap_actual"].median())
        n[f"NumPurge{tag}SharedTime"] = int(b[f"{mode}_shared_timepoints_at_boundary"].median())
        n[f"NumPurge{tag}SharedTarget"] = int(
            b[f"{mode}_shared_target_points_at_boundary"].median()
        )
    n["NumPurgeLegacyOrigins"] = int(b["legacy_purge_origins"].median())
    n["NumPurgeLegacyGap"] = int(b["legacy_min_origin_gap_actual"].median())
    n["NumPurgeLegacySharedTime"] = int(b["legacy_shared_timepoints_at_boundary"].median())
    n["NumPurgeLegacySharedTarget"] = int(b["legacy_shared_target_points_at_boundary"].median())
    n["NumPurgeLegacyLeakCells"] = int(
        (pur["legacy_shared_target_points_at_boundary"] > 0).sum()
    )

    # ---- A10: what the provenance contract does and does not catch ---- #
    n["NumContractFaults"] = int(len(cov))
    n["NumContractDetected"] = int((cov["detected"] == 1).sum())
    n["NumContractUndetected"] = int((cov["detected"] == 0).sum())
    tests = _pytest_counts()
    n["NumContractAdversarialTests"] = int(tests["adversarial"])
    n["NumContractRepoTests"] = int(tests["total"])
    n["NumBackfillDirs"] = int((mig["status"] == "ok").sum())
    n["NumBackfillSidecars"] = int(mig["n_sidecars_planned"].sum())
    n["NumBackfillFailures"] = int((~mig["status"].isin(["ok", "skipped"])).sum())
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


def _tex(s: object) -> str:
    """Escape an artefact string so it can be typeset verbatim in a table cell."""
    t = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("|", r"\textbar{}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        t = t.replace(a, b)
    return t


def _mono(s: object, allow_break: bool = False) -> str:
    t = _tex(s)
    if allow_break:
        t = t.replace(r"\_", r"\_\allowbreak{}").replace("+", r"+\allowbreak{}")
    return r"\texttt{" + t + "}"


def _ci(lo: float, hi: float, dec: int) -> str:
    return f"$[{lo:.{dec}f},\\,{hi:.{dec}f}]$"


def _num(v: float, dec: int) -> str:
    """Negative cells need math mode so the minus sign is a minus, not a hyphen."""
    return f"${v:.{dec}f}$" if v < 0 else f"{v:.{dec}f}"


def revision_tables(cfg: GridConfig, out: Path) -> int:
    """The six tables added by the revision; captions stay in main.tex."""
    obs = _read(cfg, "observed_perm_scan.csv")
    cc = _read(cfg, "cat_conditions.csv")
    ca = _read(cfg, "cat_analytic_check.csv")
    dep = _read(cfg, "t1_difficulty_depaware.csv")
    clus = _read(cfg, "t1_cluster_summary.csv").set_index(["cluster_def", "statistic"])
    pur = _read(cfg, "t1_purge_audit.csv")
    cov = _read(cfg, "contract_coverage.csv")

    # ---- observed loader permutations, and what the detectors do with them ---- #
    arms = int(obs["n_arms_aot"].sum())
    miss = int(obs["n_arms_aot_rejects"].sum())
    share = obs["perm_sharing"].value_counts()
    ident = obs["n_pairs_identical"].value_counts().sort_index()
    ng = len(obs)
    rows = [
        ["groups of arms over a common sample set", f"{ng}"],
        ["arms (group $\\times$ plugin)", f"{arms}"],
        ["arms whose validation order is not the identity",
         f"{int(obs['n_arms_misordered'].sum())}"],
        [f"groups with {_mono('val_loader_order == val_index_order[val_perm]')}",
         f"{int(obs['perm_consistent'].sum())}"],
        [r"sharing: one permutation for all arms (\texttt{all\_identical})",
         f"{int(share.get('all_identical', 0))}"],
        [r"sharing: some arms share (\texttt{partial\_shared})",
         f"{int(share.get('partial_shared', 0))}"],
        [r"sharing: every arm distinct (\texttt{all\_distinct})",
         f"{int(share.get('all_distinct', 0))}"],
        ["identical arm pairs per group: "
         + ", ".join(f"{int(k)}" for k in ident.index) + " pairs",
         " / ".join(f"{int(v)}" for v in ident.to_numpy())],
        ["mean Spearman between arm permutations, median (min, max)",
         f"{obs['spearman_perm_mean'].median():.4f} "
         f"({obs['spearman_perm_mean'].min():.4f}, {obs['spearman_perm_mean'].max():.4f})"],
        ["CAT certifies the observed loader order",
         f"{int(obs['cat_certified'].sum())} / {ng} "
         f"({100 * obs['cat_certified'].mean():.2f}\\%)"],
        [r"CAT median $\bar{\rho}$, observed loader order", f"{obs['cat_rho_bar'].median():.4f}"],
        ["CAT certifies the intact index order",
         f"{int(obs['cat_certified_intact'].sum())} / {ng} "
         f"({100 * obs['cat_certified_intact'].mean():.2f}\\%)"],
        [r"CAT median $\bar{\rho}$, intact index order",
         f"{obs['cat_rho_bar_intact'].median():.4f}"],
        ["AOT flags the observed loader order",
         f"{arms - miss} / {arms} ({rnd(100 * (arms - miss) / arms, 2):.2f}\\%)"],
        ["arms AOT fails to flag", f"{miss}"],
        ["groups with at least one unflagged arm",
         f"{int((obs['n_arms_aot_rejects'] > 0).sum())} / {ng}"],
        ["smallest AOT $p$ over all arms", f"{obs['aot_p_min'].min():.3g}"],
    ]
    (out / "observed_perm.tex").write_text(
        "\\small\n" + _tabular(["quantity", "value"], rows, "lr")
    )

    # ---- CAT under six controlled sharing conditions ---- #
    rows = []
    for cond, _tag, label in CAT_CONDITIONS:
        g = cc[cc["condition"] == cond]
        lab = g["sharing_labels"].dropna().unique()
        rows.append([
            _mono(cond),
            label,
            _mono(lab[0]) if len(lab) else "---",
            f"{int(g['n_distinct_perms_injected'].iloc[0])}",
            f"{int(g['certified'].sum())} / {len(g)}",
            _num(float(g["rho_bar"].median()), 4),
            f"{g['p_mc'].median():.4f}",
        ])
    (out / "cat_conditions.tex").write_text(
        "\\small\n"
        + _tabular(
            ["condition", "sharing pattern", "arm labels", "distinct perms.",
             "certified", r"median $\bar{\rho}$", "median $p$ (MC)"],
            rows,
            "lllrrrr",
        )
    )

    # ---- analytic null versus Monte Carlo, independent permutations only ---- #
    ind = ca[ca["condition"] == "independent"]
    m = int(ind["m_pairs"].iloc[0])
    rows = [
        ["groups evaluated", f"{len(ind)}", f"{len(ind)}"],
        [r"median $|p_{\mathrm{analytic}} - p_{\mathrm{MC}}|$",
         f"{ind['abs_err_corrected'].median():.4f}", f"{ind['abs_err_naive'].median():.4f}"],
        [r"max $|p_{\mathrm{analytic}} - p_{\mathrm{MC}}|$",
         f"{ind['abs_err_corrected'].max():.4f}", f"{ind['abs_err_naive'].max():.4f}"],
        ["median null SD ratio (analytic / Monte Carlo)",
         f"{ind['sd_ratio_corrected'].median():.4f}", f"{ind['sd_ratio_naive'].median():.4f}"],
        [f"SD factor the formula assumes ($m={m}$ pairs)",
         "$1$", f"$1/\\sqrt{{{m}}} = {1 / np.sqrt(m):.4f}$"],
    ]
    (out / "cat_analytic.tex").write_text(
        "\\small\n"
        + _tabular(["quantity", "corrected $z$", "naive $z$"], rows, "lrr")
    )

    # ---- dependence-aware association, with cluster intervals ---- #
    di = dep[dep["condition"] == "index"]

    def cl(stat: str, cdef: str, scale: float, dec: int) -> str:
        r = clus.loc[(cdef, stat)]
        return _ci(scale * float(r["ci_lo"]), scale * float(r["ci_hi"]), dec)

    rows = [
        [r"median $\rho$(volatility, per-window error)",
         f"{di['assoc_rho_fstd'].median():.4f}",
         cl("index_median_rho", "sample_set", 1.0, 4),
         cl("index_median_rho", "dataset", 1.0, 4)],
        [r"significant at $\alpha=0.05$, iid approximation (\%)",
         f"{100 * (di['assoc_p_iid_naive'] < ALPHA).mean():.2f}",
         cl("index_frac_p_iid_lt_05", "sample_set", 100.0, 2),
         cl("index_frac_p_iid_lt_05", "dataset", 100.0, 2)],
        [r"significant at $\alpha=0.05$, block permutation (\%)",
         f"{100 * (di['assoc_p_block'] < ALPHA).mean():.2f}",
         cl("index_frac_p_block_lt_05", "sample_set", 100.0, 2),
         cl("index_frac_p_block_lt_05", "dataset", 100.0, 2)],
        [r"moving-block bootstrap interval excludes $0$ (\%)",
         f"{100 * di['assoc_rho_ci_excludes_zero'].mean():.2f}",
         cl("index_frac_ci_excludes_zero", "sample_set", 100.0, 2),
         cl("index_frac_ci_excludes_zero", "dataset", 100.0, 2)],
        [r"median $\rho$, defect $-$ control",
         _num(float(clus.loc[("sample_set", "defect_minus_control_median"), "point"]), 6),
         cl("defect_minus_control_median", "sample_set", 1.0, 6),
         cl("defect_minus_control_median", "dataset", 1.0, 6)],
        ["clusters resampled",
         f"{di['cell_id'].nunique()} cells",
         f"{int(clus.loc[('sample_set', 'index_median_rho'), 'n_clusters'])} sample sets",
         f"{int(clus.loc[('dataset', 'index_median_rho'), 'n_clusters'])} datasets"],
    ]
    (out / "depaware.tex").write_text(
        "\\small\n"
        + _tabular(
            ["statistic over cells", "point", r"95\% CI, sample-set clusters",
             r"95\% CI, dataset clusters"],
            rows,
            "lrrr",
        )
    )

    # ---- purge audit at the boundary, worst bucket L=96, H=720 ---- #
    b = pur[(pur["seq_len"] == 96) & (pur["pred_len"] == 720)
            & (pur["labels_holdout_eligible"] == 1)]
    rows = []
    for mode, label in (("legacy", "legacy, fixed purge"),
                        ("labels", r"\texttt{labels} (targets disjoint)"),
                        ("disjoint", r"\texttt{disjoint} (inputs and targets disjoint)")):
        req = (f"{int(b[mode + '_required_origin_gap'].median())}"
               if mode + "_required_origin_gap" in b else "---")
        elig = (f"{int(pur[mode + '_holdout_eligible'].sum())}"
                if mode + "_holdout_eligible" in pur else "---")
        rows.append([
            label,
            f"{int(b[mode + '_purge_origins'].median())}",
            req,
            f"{int(b[mode + '_min_origin_gap_actual'].median())}",
            f"{int(b[mode + '_shared_timepoints_at_boundary'].median())}",
            f"{int(b[mode + '_shared_target_points_at_boundary'].median())}",
            f"{int((pur[mode + '_shared_target_points_at_boundary'] > 0).sum())}",
            elig,
        ])
    (out / "purge_audit.tex").write_text(
        "\\small\n"
        + _tabular(
            ["purge rule", "purged origins", "required gap", "actual gap",
             "shared time pts.", "shared target pts.", "leaking cells", "eligible cells"],
            rows,
            "lrrrrrrr",
        )
    )

    # ---- provenance contract: what it catches and what it cannot ---- #
    hit = cov[cov["detected"] == 1]
    lines = [
        "\\footnotesize",
        "\\begin{tabular}{@{}p{0.44\\linewidth}p{0.26\\linewidth}p{0.20\\linewidth}@{}}",
        "\\toprule",
        "injected fault & clause that fires & outcome \\\\",
        "\\midrule",
    ]
    for _, r in hit.iterrows():
        lines.append(
            f"{_mono(r['fault'], True)} & {_mono(r['detected_by_field'], True)} & "
            f"{_mono(r['exception'], True)} \\\\"
        )
    lines += [
        "\\midrule",
        "\\multicolumn{3}{@{}l@{}}{\\emph{faults the contract accepts without error "
        "(blind spots)}} \\\\",
    ]
    for _, r in cov[cov["detected"] == 0].iterrows():
        lines.append(f"{_mono(r['fault'], True)} & (none) & {_mono(r['exception'])} \\\\")
        lines.append(
            "\\multicolumn{3}{@{}p{0.92\\linewidth}@{}}{\\hspace{1em}\\emph{"
            + _tex(r["note"])
            + "}} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "contract_coverage.tex").write_text("\n".join(lines))
    return 6


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

    # tail calibration of the normal reference
    tail = _read(cfg, "aot_tail.csv")
    if len(tail):
        rows = []
        for (cell, nn), g in tail.groupby(["cell_id", "n"], sort=False):
            g = g.sort_values("alpha", ascending=False)
            cells = [f"{nn}", f"{g['kurtosis'].iloc[0]:.1f}"]
            for _, r in g.iterrows():
                cells.append(f"{r['ratio_to_nominal']:.2f}")
            cells.append(f"{g['max_z_observed'].iloc[0]:.2f}")
            rows.append(cells)
        rows.sort(key=lambda r: int(r[0]))
        alph = sorted(tail["alpha"].unique(), reverse=True)
        header = ["$n$", "kurtosis"] + [
            f"$10^{{{int(round(np.log10(a)))}}}$" for a in alph
        ] + [r"$\max z$"]
        (out / "tail.tex").write_text(
            _tabular(header, rows, "rr" + "r" * len(alph) + "r")
        )

    # T2: sensitivity to milder corruptions, in the flagged direction
    sens = _read(cfg, "aot_sensitivity.csv")
    rows = []
    for kind, label in (("block", "block-order shuffle, block size $b$"),
                        ("partial", "partial shuffle, fraction $f$ moved")):
        g = sens[sens["kind"] == kind]
        for p, gg in g.groupby("param"):
            ptxt = f"{int(p)}" if kind == "block" else f"{p:.2f}"
            # order structure left behind: pairs inside a block, or entries not moved
            kept = 100.0 * (1.0 - 1.0 / p) if kind == "block" else 100.0 * (1.0 - p)
            rows.append(
                [label, ptxt, f"{kept:.1f}", f"{len(gg)}",
                 f"{100 * (1 - gg['reject'].mean()):.2f}"]
            )
    (out / "sensitivity.tex").write_text(
        _tabular(
            ["corruption", "parameter", r"order structure kept (\%)", "sequences",
             r"AOT flagged (\%)"],
            rows,
            "llrrr",
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
    print(f"[tables] {7 + revision_tables(cfg, out)} tables -> {out}")


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
