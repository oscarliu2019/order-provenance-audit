import numpy as np
import pytest

from src.config import VAL_ORDERS, Cell, GridConfig
from src.features import FEATURE_COLUMNS, window_features


@pytest.fixture(scope="module")
def cfg():
    return GridConfig()


def test_val_order_is_part_of_the_cell_identity():
    a = Cell("ETTh1", "DLinear", "none", 96, 2021, "deterministic")
    b = Cell("ETTh1", "DLinear", "none", 96, 2021, "shuffled")
    assert a.cell_id != b.cell_id
    assert a.cell_id.endswith("_det") and b.cell_id.endswith("_shuf")


def test_invalid_val_order_is_rejected():
    with pytest.raises(ValueError):
        Cell("ETTh1", "DLinear", "none", 96, 2021, "sorted")
    assert set(VAL_ORDERS) == {"deterministic", "shuffled"}


def test_stage_expansion_is_deduplicated_and_stable(cfg):
    e1 = cfg.stage_cells("e1")
    e2 = cfg.stage_cells("e2")
    assert len({c.cell_id for c in e1}) == len(e1)
    assert len({c.cell_id for c in e2}) == len(e2)
    assert all(c.val_order == "deterministic" for c in e1)
    assert {c.val_order for c in e2} == {"deterministic", "shuffled"}
    # every shuffled cell in E2 has a deterministic twin, which is what makes
    # the RNG-coupling comparison paired
    det = {(c.dataset, c.backbone, c.plugin, c.pred_len, c.seed)
           for c in e2 if c.val_order == "deterministic"}
    shuf = {(c.dataset, c.backbone, c.plugin, c.pred_len, c.seed)
            for c in e2 if c.val_order == "shuffled"}
    assert det == shuf


def test_ili_overrides_are_honoured(cfg):
    assert cfg.seq_len("ILI") == 36
    assert cfg.label_len("ILI") == 18
    assert cfg.seq_len("ETTh1") == 96
    assert cfg.dataset("ILI")["horizons"] == [24, 36, 48, 60]


def test_window_features_count_matches_the_sliding_window_dataset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, 4))
    for seq_len, pred_len in ((96, 96), (36, 24), (96, 720)):
        df = window_features(x, seq_len, pred_len)
        assert len(df) == 1000 - seq_len - pred_len + 1
        assert list(df.columns) == ["window_index"] + FEATURE_COLUMNS


def test_window_features_are_computed_from_the_input_window_only():
    """Changing points that only ever appear in a target must not move features."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 2))
    seq_len, pred_len = 96, 96
    a = window_features(x, seq_len, pred_len)
    y = x.copy()
    y[-pred_len:] += 100.0  # only ever a target, never an input
    b = window_features(y, seq_len, pred_len)
    assert np.allclose(a[FEATURE_COLUMNS].to_numpy(), b[FEATURE_COLUMNS].to_numpy())


def test_feature_values_on_an_analytic_signal():
    t = np.arange(600)
    x = np.sin(2 * np.pi * t / 24).reshape(-1, 1)
    df = window_features(x, 96, 96)
    # four whole periods per window: std of a unit sine is 1/sqrt(2)
    assert abs(df["f_std"].mean() - 1 / np.sqrt(2)) < 1e-3
    assert abs(df["f_range"].mean() - 2.0) < 1e-2
    # lag-1 autocorrelation of a 24-step sine, up to the finite-window edge
    # effect of the non-circular estimator
    assert abs(df["f_acf1"].mean() - np.cos(2 * np.pi / 24)) < 2e-2
    # a pure tone concentrates its spectrum, so normalised entropy is small
    assert df["f_spec_ent"].mean() < 0.2


def test_volatility_feature_orders_two_series_correctly():
    rng = np.random.default_rng(2)
    calm = window_features(0.1 * rng.normal(size=(400, 1)), 96, 96)["f_std"].mean()
    wild = window_features(3.0 * rng.normal(size=(400, 1)), 96, 96)["f_std"].mean()
    assert wild > 10 * calm


def test_window_features_reject_a_too_short_series():
    with pytest.raises(ValueError):
        window_features(np.zeros((50, 2)), 96, 96)


def test_declared_dataset_lengths_match_the_csv_files(cfg):
    """Guards against silently running on a different release of a dataset."""
    import pandas as pd

    from src.config import REPO_ROOT

    for ds in cfg.raw["datasets"]:
        p = REPO_ROOT / ds["root_path"] / ds["data_path"]
        if not p.exists():
            pytest.skip(f"{p} not downloaded")
        df = pd.read_csv(p)
        assert len(df) == ds["total_len"], f"{ds['name']}: {len(df)} rows"
        assert df.shape[1] - 1 == ds["enc_in"], f"{ds['name']}: {df.shape[1] - 1} channels"
