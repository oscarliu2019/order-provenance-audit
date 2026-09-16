"""Per-window features, computed in index order by construction.

These features play the role of the "left table" of the positional join under
audit. They are derived from the *input* window only, so nothing here leaks the
forecast target, and they are computed on the same standardised array the model
consumes (``data_set.data_x``), so a feature row and an error entry with the
same position genuinely describe the same window -- provided the error vector is
in index order.

Everything is vectorised over windows with stride tricks: a 34k-window split
with L=96 and 21 channels is a 34k x 96 x 21 view, not a copy.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Cell, GridConfig  # noqa: E402

FEATURE_COLUMNS = [
    "f_std",
    "f_range",
    "f_absdiff",
    "f_slope",
    "f_acf1",
    "f_drift",
    "f_spec_ent",
    "f_cross_corr",
    "f_last_dev",
    "f_kurtosis",
]


def _windows(x: np.ndarray, length: int) -> np.ndarray:
    """Sliding windows over axis 0 as a view: (n_windows, length, n_channels)."""
    n = x.shape[0] - length + 1
    if n <= 0:
        raise ValueError(f"series of length {x.shape[0]} is shorter than window {length}")
    s0, s1 = x.strides
    return np.lib.stride_tricks.as_strided(x, shape=(n, length, x.shape[1]), strides=(s0, s0, s1))


def window_features(data_x: np.ndarray, seq_len: int, pred_len: int) -> pd.DataFrame:
    """Features for every window that the TSLib sliding-window dataset yields.

    A window ``j`` needs ``seq_len + pred_len`` points, so the number of windows
    is ``len(data_x) - seq_len - pred_len + 1``; the features use the first
    ``seq_len`` points of each window.
    """
    x = np.ascontiguousarray(np.asarray(data_x, dtype=np.float64))
    n_windows = x.shape[0] - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("split too short for this (seq_len, pred_len)")
    w = _windows(x, seq_len)[:n_windows]  # (n, L, C)
    eps = 1e-12

    std = w.std(axis=1)                                    # (n, C)
    rng = w.max(axis=1) - w.min(axis=1)
    d1 = np.abs(np.diff(w, axis=1)).mean(axis=1)
    t = np.arange(seq_len, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + eps)
    slope = np.tensordot(w - w.mean(axis=1, keepdims=True), t, axes=([1], [0])) / seq_len

    centred = w - w.mean(axis=1, keepdims=True)
    num = (centred[:, :-1, :] * centred[:, 1:, :]).sum(axis=1)
    den = (centred**2).sum(axis=1) + eps
    acf1 = num / den

    half = seq_len // 2
    drift = np.abs(w[:, half:, :].mean(axis=1) - w[:, :half, :].mean(axis=1)) / (std + eps)
    last_dev = np.abs(w[:, -1, :] - w.mean(axis=1)) / (std + eps)
    m2 = (centred**2).mean(axis=1)
    m4 = (centred**4).mean(axis=1)
    kurt = m4 / (m2**2 + eps)

    # spectral entropy of the channel-mean series (one FFT per window)
    cm = w.mean(axis=2)
    cm = cm - cm.mean(axis=1, keepdims=True)
    p = np.abs(np.fft.rfft(cm, axis=1)) ** 2
    p = p[:, 1:]
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    spec_ent = -(p * np.log(p + eps)).sum(axis=1) / np.log(p.shape[1])

    # mean pairwise |correlation| across channels, computed from the window's
    # own covariance so it is a per-window quantity
    n_ch = w.shape[2]
    if n_ch > 1:
        cov = np.einsum("nlc,nld->ncd", centred, centred) / seq_len
        sd = np.sqrt(np.maximum(np.einsum("ncc->nc", cov), eps))
        corr = np.abs(cov / (sd[:, :, None] * sd[:, None, :] + eps))
        iu = np.triu_indices(n_ch, k=1)
        cross = corr[:, iu[0], iu[1]].mean(axis=1)
    else:
        cross = np.zeros(w.shape[0])

    df = pd.DataFrame(
        {
            "window_index": np.arange(n_windows),
            "f_std": std.mean(axis=1),
            "f_range": rng.mean(axis=1),
            "f_absdiff": d1.mean(axis=1),
            "f_slope": slope.mean(axis=1),
            "f_acf1": acf1.mean(axis=1),
            "f_drift": drift.mean(axis=1),
            "f_spec_ent": spec_ent,
            "f_cross_corr": cross,
            "f_last_dev": last_dev.mean(axis=1),
            "f_kurtosis": kurt.mean(axis=1),
        }
    )
    return df


def split_features(cfg: GridConfig, dataset: str, pred_len: int, split: str) -> pd.DataFrame:
    """Build features for one (dataset, horizon, split) using TSLib's own split."""
    from src.train import build_configs, ensure_tslib

    ensure_tslib(cfg)
    from data_provider.data_factory import data_provider  # type: ignore

    cell = Cell(dataset=dataset, backbone="DLinear", plugin="none", pred_len=pred_len, seed=2021)
    ns: Namespace = build_configs(cfg, cell, _cpu_run())
    ds, _ = data_provider(ns, split)
    df = window_features(ds.data_x, ns.seq_len, ns.pred_len)
    if len(df) != len(ds):
        raise AssertionError(
            f"{dataset}/{split}/h{pred_len}: {len(df)} feature rows vs {len(ds)} dataset windows"
        )
    df.insert(0, "split", split)
    df.insert(0, "pred_len", int(pred_len))
    df.insert(0, "dataset", dataset)
    return df


def _cpu_run():
    from src.train import RunOptions

    return RunOptions(device="cpu", num_workers=0)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="build per-window feature tables")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = GridConfig(a.config)
    out = Path(a.out) if a.out else cfg.path("artifacts_dir") / "window_features"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in cfg.raw["datasets"]:
        for h in ds["horizons"]:
            for split in ("val", "test"):
                df = split_features(cfg, ds["name"], int(h), split)
                p = out / f"{ds['name']}_h{h}_{split}.csv"
                df.to_csv(p, index=False)
                rows.append({"dataset": ds["name"], "pred_len": h, "split": split, "n": len(df)})
                print(f"[features] {p.name}: {len(df)} windows", flush=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
