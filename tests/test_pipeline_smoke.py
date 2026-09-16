"""End-to-end smoke tests that exercise the real upstream data pipeline.

These are the tests that establish the *premise* of the paper: that the upstream
validation loader really is shuffled, and that a per-window vector recorded
through it really is a permutation of the index-order vector. They need the
datasets, so they skip when the data directory is empty.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import REPO_ROOT, Cell, GridConfig

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    """A config whose *output* paths are redirected into a scratch directory.

    Without this, a one-epoch smoke run would overwrite the real per-window
    artefacts of the corresponding cell -- the same class of silent corruption
    the paper is about, so the test suite had better not commit it.
    """
    c = GridConfig()
    tag = tmp_path_factory.mktemp("scratch").name
    for key in ("runs_dir", "cells_dir", "winerr_dir", "artifacts_dir"):
        c.raw["paths"][key] = f".scratch/{tag}/{key}"
    return c


def _have(cfg, name: str) -> bool:
    ds = cfg.dataset(name)
    return (REPO_ROOT / ds["root_path"] / ds["data_path"]).exists()


def test_upstream_val_loader_is_shuffled_but_test_loader_is_not(cfg):
    """The defect's origin, asserted against the vendored upstream source."""
    pytest.importorskip("torch")
    if not _have(cfg, "ILI"):
        pytest.skip("ILI not downloaded")
    from torch.utils.data import RandomSampler, SequentialSampler

    from src.train import RunOptions, build_configs, ensure_tslib

    ensure_tslib(cfg)
    from data_provider.data_factory import data_provider  # type: ignore

    cell = Cell("ILI", "DLinear", "none", 24, 2021)
    ns = build_configs(cfg, cell, RunOptions(device="cpu", num_workers=0))
    _, val_loader = data_provider(ns, "val")
    _, test_loader = data_provider(ns, "test")
    assert isinstance(val_loader.sampler, RandomSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    # and drop_last is off for evaluation, so a length check cannot help
    assert val_loader.drop_last is False


def test_recorded_val_vector_is_a_permutation_of_the_index_order_vector(cfg, tmp_path):
    pytest.importorskip("torch")
    if not _have(cfg, "ILI"):
        pytest.skip("ILI not downloaded")
    from src.train import RunOptions, run_cell
    from src.winerr import load_series

    cell = Cell("ILI", "DLinear", "none", 24, 2021, "shuffled")
    run = RunOptions(device="cpu", max_epochs=1, num_workers=0)
    res = run_cell(cfg, cell, run)
    d = cfg.path("winerr_dir") / cell.cell_id
    rec = load_series(d, "val_loader_order", require_order="loader")
    idx = load_series(d, "val_index_order", require_order="index")
    perm = load_series(d, "val_perm", require_order="loader").astype(np.int64)

    assert rec.size == idx.size == res["n_val_windows"]
    assert np.array_equal(np.sort(perm), np.arange(perm.size))
    assert not np.array_equal(perm, np.arange(perm.size)), "loader was not actually shuffled"
    # same multiset, hence identical aggregate validation MSE
    assert np.array_equal(np.sort(rec), np.sort(idx))
    assert abs(rec.mean() - idx.mean()) < 1e-12 * abs(idx.mean())
    # but the serial structure is gone, which is what the audit detects
    from src.detect import aot

    assert aot(idx)["p"] < 1e-6
    assert aot(rec)["p"] > 1e-3


def test_deterministic_val_order_reproduces_index_order(cfg):
    pytest.importorskip("torch")
    if not _have(cfg, "ILI"):
        pytest.skip("ILI not downloaded")
    from src.train import RunOptions, run_cell
    from src.winerr import load_series

    cell = Cell("ILI", "DLinear", "none", 24, 2021, "deterministic")
    run_cell(cfg, cell, RunOptions(device="cpu", max_epochs=1, num_workers=0))
    d = cfg.path("winerr_dir") / cell.cell_id
    rec = load_series(d, "val_loader_order", require_order="loader")
    idx = load_series(d, "val_index_order", require_order="index")
    perm = load_series(d, "val_perm", require_order="loader").astype(np.int64)
    assert np.array_equal(perm, np.arange(perm.size))
    assert np.array_equal(rec, idx)


def test_truncated_runs_do_not_write_window_vectors(cfg):
    """A quick timing probe must never overwrite a full per-window artefact."""
    pytest.importorskip("torch")
    if not _have(cfg, "ILI"):
        pytest.skip("ILI not downloaded")
    from src.train import RunOptions, run_cell

    cell = Cell("ILI", "DLinear", "revin", 24, 2022, "deterministic")
    res = run_cell(
        cfg, cell, RunOptions(device="cpu", max_epochs=1, max_eval_steps=1, num_workers=0)
    )
    assert res["truncated"] is True
    assert "winerr_dir" not in res
    assert not (cfg.path("winerr_dir") / cell.cell_id).exists()
