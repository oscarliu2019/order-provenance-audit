import numpy as np
import pytest

from src.winerr import (
    ORDER_INDEX,
    ORDER_LOADER,
    PROVENANCE_SUFFIX,
    ProvenanceError,
    load_series,
    read_provenance,
    save_series,
    unpermute,
)


def test_unpermute_inverts_loader_order():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=500)
    perm = rng.permutation(500)
    recorded = truth[perm]
    assert np.allclose(unpermute(recorded, perm), truth)


def test_unpermute_rejects_non_permutation():
    with pytest.raises(ValueError):
        unpermute(np.zeros(4), np.array([0, 0, 1, 2]))
    with pytest.raises(ValueError):
        unpermute(np.zeros(4), np.array([0, 1, 2]))


def test_mean_is_invariant_to_loader_order():
    """The masking mechanism: aggregate validation MSE cannot see a permutation."""
    rng = np.random.default_rng(1)
    truth = rng.gamma(2.0, 1.0, size=4096)
    perm = rng.permutation(truth.size)
    assert abs(truth.mean() - truth[perm].mean()) < 1e-12 * abs(truth.mean())
    assert np.array_equal(np.sort(truth), np.sort(truth[perm]))


def test_length_check_is_blind_when_drop_last_is_false():
    """With drop_last=False both orders have the full window count."""
    rng = np.random.default_rng(2)
    truth = rng.normal(size=997)
    perm = rng.permutation(truth.size)
    assert truth.size == truth[perm].size


def test_save_and_load_round_trip(tmp_path):
    v = np.arange(10, dtype=np.float64)
    save_series(tmp_path, "val_index_order", v, ORDER_INDEX, cell_id="x")
    assert (tmp_path / f"val_index_order{PROVENANCE_SUFFIX}").exists()
    assert np.array_equal(load_series(tmp_path, "val_index_order"), v)
    prov = read_provenance(tmp_path, "val_index_order")
    assert prov["order"] == ORDER_INDEX
    assert prov["n_windows"] == 10
    assert prov["cell_id"] == "x"


def test_missing_sidecar_is_refused(tmp_path):
    v = np.arange(5, dtype=np.float64)
    save_series(tmp_path, "s", v, ORDER_INDEX)
    (tmp_path / f"s{PROVENANCE_SUFFIX}").unlink()
    with pytest.raises(ProvenanceError):
        load_series(tmp_path, "s")
    # order-invariant consumers may still opt out explicitly
    assert np.array_equal(load_series(tmp_path, "s", require_order=None), v)


def test_loader_order_cannot_be_used_for_positional_join(tmp_path):
    save_series(tmp_path, "s", np.arange(5, dtype=np.float64), ORDER_LOADER)
    with pytest.raises(ProvenanceError):
        load_series(tmp_path, "s", require_order=ORDER_INDEX)
    assert load_series(tmp_path, "s", require_order=ORDER_LOADER).size == 5


def test_sidecar_length_disagreement_is_caught(tmp_path):
    import json

    save_series(tmp_path, "s", np.arange(5, dtype=np.float64), ORDER_INDEX)
    p = tmp_path / f"s{PROVENANCE_SUFFIX}"
    rec = json.loads(p.read_text())
    rec["n_windows"] = 4
    p.write_text(json.dumps(rec))
    with pytest.raises(ProvenanceError):
        load_series(tmp_path, "s")


def test_expect_n_guard(tmp_path):
    save_series(tmp_path, "s", np.arange(5, dtype=np.float64), ORDER_INDEX)
    with pytest.raises(ProvenanceError):
        load_series(tmp_path, "s", expect_n=6)


def test_unknown_order_label_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_series(tmp_path, "s", np.arange(3, dtype=np.float64), "whatever")


def test_recorder_recovers_index_order():
    torch = pytest.importorskip("torch")
    from src.winerr import WindowRecorder

    rng = np.random.default_rng(3)
    n, h, c = 64, 8, 3
    true_err = np.zeros(n)
    pred = torch.zeros(n, h, c)
    tgt = torch.tensor(rng.normal(size=(n, h, c)).astype(np.float32))
    true_err = (tgt.numpy() ** 2).mean(axis=(1, 2))

    perm = rng.permutation(n)
    rec = WindowRecorder()
    for i in range(0, n, 7):
        sl = perm[i : i + 7]
        rec.update(pred[sl], tgt[sl], index=sl)
    assert not np.allclose(rec.errors(), true_err)
    assert np.allclose(rec.errors_in_index_order(), true_err, atol=1e-6)
    assert abs(rec.mse - true_err.mean()) < 1e-6


def test_recorder_without_indices_cannot_rekey():
    torch = pytest.importorskip("torch")
    from src.winerr import WindowRecorder

    rec = WindowRecorder()
    rec.update(torch.zeros(4, 2, 1), torch.ones(4, 2, 1))
    with pytest.raises(ProvenanceError):
        rec.errors_in_index_order()
