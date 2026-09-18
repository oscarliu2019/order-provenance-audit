"""Adversarial tests for the sample-level provenance contract.

Every test here is an attack that the pre-revision implementation waved through:
it only compared an ``order`` label and a window count, so any equal-length array
-- rotated, permuted, from another sample set, or left over from a previous run --
passed silently.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.provenance import (
    ORDER_DATASET_INDEX,
    SCHEMA_VERSION,
    DigestMismatchError,
    IdentityMismatchError,
    MissingFieldError,
    OrderSemanticsError,
    ProvenanceError,
    ProvenanceWarning,
    SchemaVersionError,
    assert_positional_join_allowed,
    build_sidecar,
    canonical_sample_ids,
    compute_ordered_sample_id_digest,
    compute_value_digest,
    join_by_sample_id,
    validate_sidecar,
)
from src.seeds import stable_rng
from src.winerr import (
    ORDER_INDEX,
    ORDER_LOADER,
    PROVENANCE_SUFFIX,
    load_series,
    load_series_with_ids,
    read_provenance,
    save_series,
    write_provenance,
)

IDENT = dict(dataset="ETTh1", backbone="DLinear", plugin="none", seq_len=96, pred_len=336,
             seed=2021)
N = 128


def _values(tag: str, n: int = N) -> np.ndarray:
    return stable_rng("test-provenance", tag).gamma(2.0, 1.0, size=n)


def _write_index(d, name="val_index_order", split="val", values=None, **over):
    ident = {**IDENT, **over}
    v = _values(f"{name}{split}{ident['pred_len']}{ident['dataset']}") if values is None else values
    save_series(d, name, v, ORDER_INDEX, split=split, window_origins=np.arange(v.size),
                sample_id_spec={"kind": "identity_range", "start": 0}, **ident)
    return v


def _write_loader(d, perm, values=None, split="val"):
    v = _values("loader") if values is None else values
    save_series(d, "val_loader_order", v[perm], ORDER_LOADER, split=split, window_origins=perm,
                sample_id_spec={"kind": "sibling_array", "name": "val_perm"}, **IDENT)
    save_series(d, "val_perm", perm.astype(np.float64), ORDER_LOADER, split=split,
                window_origins=perm,
                sample_id_spec={"kind": "sibling_array", "name": "val_perm"}, **IDENT)
    return v


def _patch(d, name, **changes):
    rec = read_provenance(d, name)
    rec.update(changes)
    write_provenance(d, name, rec)


# --------------------------------------------------------------------------- #
# round trip and digest primitives
# --------------------------------------------------------------------------- #
def test_round_trip_passes_and_records_every_contract_field(tmp_path):
    v = _write_index(tmp_path)
    got = load_series(tmp_path, "val_index_order", expect={"split": "val", **IDENT})
    assert np.array_equal(got, v)
    sc = read_provenance(tmp_path, "val_index_order")
    for field in ("schema_version", "split", "n_samples", "ordered_sample_id_digest",
                  "value_digest", "order_semantics", "sample_id_scheme", "dataset", "backbone",
                  "plugin", "seq_len", "pred_len", "seed", "producer_commit"):
        assert field in sc, field
    assert sc["schema_version"] == SCHEMA_VERSION
    assert sc["order_semantics"] == ORDER_DATASET_INDEX
    assert sc["n_samples"] == N


def test_value_digest_is_stable_across_dtype_and_endianness():
    a = _values("dtype")
    assert compute_value_digest(a) == compute_value_digest(a.astype(">f8"))
    assert compute_value_digest(a) == compute_value_digest(np.ascontiguousarray(a[::1]))
    assert compute_value_digest(a) != compute_value_digest(a.astype(np.float32))
    z = np.array([0.0, 1.0])
    assert compute_value_digest(z) == compute_value_digest(np.array([-0.0, 1.0]))


def test_sample_id_digest_is_order_sensitive():
    ids = canonical_sample_ids("val", 96, 336, range(10), dataset="ETTh1")
    rot = ids[3:] + ids[:3]
    assert compute_ordered_sample_id_digest(ids) != compute_ordered_sample_id_digest(rot)
    assert compute_ordered_sample_id_digest(ids) == compute_ordered_sample_id_digest(list(ids))
    other = canonical_sample_ids("test", 96, 336, range(10), dataset="ETTh1")
    assert compute_ordered_sample_id_digest(ids) != compute_ordered_sample_id_digest(other)


# --------------------------------------------------------------------------- #
# content attacks
# --------------------------------------------------------------------------- #
def test_single_element_edit_is_caught(tmp_path):
    v = _write_index(tmp_path)
    v2 = v.copy()
    v2[17] *= 1.000001
    np.save(tmp_path / "val_index_order.npy", v2)
    with pytest.raises(DigestMismatchError):
        load_series(tmp_path, "val_index_order")


def test_cyclic_rotation_is_caught_even_though_every_summary_is_invariant(tmp_path):
    v = _write_index(tmp_path)
    rolled = np.roll(v, 31)
    assert rolled.size == v.size
    assert abs(rolled.mean() - v.mean()) < 1e-15 * abs(v.mean())
    assert abs(rolled.sum() - v.sum()) < 1e-12
    assert np.array_equal(np.sort(rolled), np.sort(v))
    np.save(tmp_path / "val_index_order.npy", rolled)
    with pytest.raises(DigestMismatchError):
        load_series(tmp_path, "val_index_order")


def test_random_permutation_is_caught_even_though_the_multiset_is_identical(tmp_path):
    v = _write_index(tmp_path)
    perm = stable_rng("attack", "perm").permutation(v.size)
    shuffled = v[perm]
    assert np.array_equal(np.sort(shuffled), np.sort(v))
    assert shuffled.size == v.size
    np.save(tmp_path / "val_index_order.npy", shuffled)
    with pytest.raises(DigestMismatchError):
        load_series(tmp_path, "val_index_order")


def test_stale_sidecar_describing_a_previous_array_is_caught(tmp_path):
    _write_index(tmp_path)
    np.save(tmp_path / "val_index_order.npy", _values("rerun"))
    with pytest.raises(DigestMismatchError):
        load_series(tmp_path, "val_index_order")


def test_truncated_array_is_caught_by_n_samples(tmp_path):
    v = _write_index(tmp_path)
    np.save(tmp_path / "val_index_order.npy", v[:-1])
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_index_order")


# --------------------------------------------------------------------------- #
# sample-set identity attacks
# --------------------------------------------------------------------------- #
def test_equal_length_array_from_another_cell_is_caught(tmp_path):
    a, b = tmp_path / "h336", tmp_path / "h192"
    _write_index(a, pred_len=336)
    other = _write_index(b, pred_len=192)
    np.save(a / "val_index_order.npy", other)
    with pytest.raises(DigestMismatchError):
        load_series(a, "val_index_order")


def test_same_length_different_horizon_cannot_be_joined(tmp_path):
    a, b = tmp_path / "h336", tmp_path / "h192"
    va = _write_index(a, pred_len=336)
    vb = _write_index(b, pred_len=192)
    assert va.size == vb.size
    sa, sb = read_provenance(a, "val_index_order"), read_provenance(b, "val_index_order")
    assert sa["n_samples"] == sb["n_samples"]
    with pytest.raises(IdentityMismatchError):
        assert_positional_join_allowed({"h336": sa, "h192": sb})
    ga = load_series_with_ids(a, "val_index_order")
    gb = load_series_with_ids(b, "val_index_order")
    with pytest.raises(IdentityMismatchError):
        join_by_sample_id({"h336": ga, "h192": gb})


def test_same_length_different_split_cannot_be_joined(tmp_path):
    v = _write_index(tmp_path, name="val_index_order", split="val")
    _write_index(tmp_path, name="test_index_order", split="test", values=v)
    with pytest.raises(IdentityMismatchError):
        assert_positional_join_allowed({"v": read_provenance(tmp_path, "val_index_order"),
                                        "t": read_provenance(tmp_path, "test_index_order")})
    with pytest.raises(IdentityMismatchError):
        join_by_sample_id({"v": load_series_with_ids(tmp_path, "val_index_order"),
                           "t": load_series_with_ids(tmp_path, "test_index_order")})


def test_split_expectation_is_enforced(tmp_path):
    _write_index(tmp_path)
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_index_order", expect={"split": "test"})
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_index_order", expect={"pred_len": 192})


def test_same_length_different_dataset_cannot_be_joined(tmp_path):
    a, b = tmp_path / "etth1", tmp_path / "etth2"
    _write_index(a, dataset="ETTh1")
    _write_index(b, dataset="ETTh2")
    with pytest.raises(IdentityMismatchError):
        assert_positional_join_allowed({"a": read_provenance(a, "val_index_order"),
                                        "b": read_provenance(b, "val_index_order")})


def test_loader_order_relabelled_as_index_order_is_caught(tmp_path):
    perm = stable_rng("loader", "perm").permutation(N)
    _write_loader(tmp_path, perm)
    _patch(tmp_path, "val_loader_order", order=ORDER_INDEX, order_semantics=ORDER_DATASET_INDEX)
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_loader_order", require_order=ORDER_INDEX)


def test_loader_and_index_products_do_not_share_a_sample_id_digest(tmp_path):
    perm = stable_rng("loader", "perm2").permutation(N)
    v = _write_loader(tmp_path, perm)
    _write_index(tmp_path, values=v)
    sl = read_provenance(tmp_path, "val_loader_order")
    si = read_provenance(tmp_path, "val_index_order")
    assert sl["ordered_sample_id_digest"] != si["ordered_sample_id_digest"]
    assert sl["value_digest"] != si["value_digest"]
    with pytest.raises(IdentityMismatchError):
        assert_positional_join_allowed({"loader": sl, "index": si})


def test_replacing_the_sample_id_source_is_caught(tmp_path):
    perm = stable_rng("loader", "perm3").permutation(N)
    _write_loader(tmp_path, perm)
    np.save(tmp_path / "val_perm.npy",
            stable_rng("loader", "perm4").permutation(N).astype(np.float64))
    with pytest.raises(ProvenanceError):
        load_series(tmp_path, "val_loader_order", require_order=ORDER_LOADER)


def test_n_samples_edited_in_the_sidecar_is_caught(tmp_path):
    _write_index(tmp_path)
    _patch(tmp_path, "val_index_order", n_samples=N - 1)
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_index_order")


def test_duplicate_sample_ids_are_refused_at_write_time(tmp_path):
    with pytest.raises(IdentityMismatchError):
        build_sidecar(values=_values("dup", 8), split="val",
                      window_origins=np.zeros(8, dtype=np.int64),
                      order_semantics=ORDER_DATASET_INDEX, **IDENT)


# --------------------------------------------------------------------------- #
# schema attacks
# --------------------------------------------------------------------------- #
def test_legacy_sidecar_is_refused_by_default_and_warns_under_legacy_ok(tmp_path):
    v = _write_index(tmp_path)
    write_provenance(tmp_path, "val_index_order",
                     {"order": "index", "n_windows": int(v.size), "split": "val", "cell_id": "c"})
    with pytest.raises(SchemaVersionError):
        load_series(tmp_path, "val_index_order")
    with pytest.warns(ProvenanceWarning):
        got = load_series(tmp_path, "val_index_order", legacy_ok=True)
    assert np.array_equal(got, v)


def test_legacy_mode_still_rejects_the_pre_contract_length_lie(tmp_path):
    v = _write_index(tmp_path)
    write_provenance(tmp_path, "val_index_order",
                     {"order": "index", "n_windows": int(v.size) - 1, "split": "val"})
    with pytest.raises(IdentityMismatchError):
        load_series(tmp_path, "val_index_order", legacy_ok=True)


def test_unknown_schema_version_is_refused_even_with_legacy_ok(tmp_path):
    _write_index(tmp_path)
    _patch(tmp_path, "val_index_order", schema_version="order-audit/provenance/999")
    with pytest.raises(SchemaVersionError):
        load_series(tmp_path, "val_index_order")
    with pytest.raises(SchemaVersionError):
        load_series(tmp_path, "val_index_order", legacy_ok=True)


def test_missing_required_field_is_refused(tmp_path):
    _write_index(tmp_path)
    rec = read_provenance(tmp_path, "val_index_order")
    rec.pop("value_digest")
    write_provenance(tmp_path, "val_index_order", rec)
    with pytest.raises(MissingFieldError):
        load_series(tmp_path, "val_index_order")


def test_order_semantics_outside_the_enumeration_is_refused(tmp_path):
    _write_index(tmp_path)
    _patch(tmp_path, "val_index_order", order_semantics="index-ish")
    with pytest.raises(OrderSemanticsError):
        load_series(tmp_path, "val_index_order")


def test_sidecar_is_json_and_survives_a_reread(tmp_path):
    _write_index(tmp_path)
    raw = json.loads((tmp_path / f"val_index_order{PROVENANCE_SUFFIX}").read_text())
    assert raw["producer_commit"]
    validate_sidecar(raw, np.load(tmp_path / "val_index_order.npy"),
                     sample_ids=canonical_sample_ids("val", 96, 336, range(N), dataset="ETTh1"))


# --------------------------------------------------------------------------- #
# join semantics
# --------------------------------------------------------------------------- #
def test_join_by_sample_id_aligns_by_id_not_by_position():
    ids_a = canonical_sample_ids("val", 96, 336, range(0, 10), dataset="ETTh1")
    ids_b = canonical_sample_ids("val", 96, 336, range(3, 13), dataset="ETTh1")
    a = np.arange(10, dtype=np.float64)
    b = np.arange(3, 13, dtype=np.float64) * 100
    out, ids = join_by_sample_id({"a": (a, ids_a), "b": (b, ids_b)})
    assert ids == ids_a[3:]
    assert np.array_equal(out["a"], np.arange(3, 10, dtype=np.float64))
    assert np.array_equal(out["b"], np.arange(3, 10, dtype=np.float64) * 100)
    assert not np.array_equal(out["a"], a[: len(ids)])  # the positional prefix would be wrong


def test_join_by_sample_id_refuses_a_product_without_ids():
    with pytest.raises(MissingFieldError):
        join_by_sample_id({"a": (np.zeros(4), None), "b": (np.zeros(4), range(4))})


def test_join_by_sample_id_refuses_disjoint_sample_sets():
    with pytest.raises(IdentityMismatchError):
        join_by_sample_id({"a": (np.zeros(4), ["x0", "x1", "x2", "x3"]),
                           "b": (np.zeros(4), ["y0", "y1", "y2", "y3"])})


def test_join_by_sample_id_refuses_duplicate_ids():
    with pytest.raises(IdentityMismatchError):
        join_by_sample_id({"a": (np.zeros(3), ["x0", "x0", "x1"]),
                           "b": (np.zeros(3), ["x0", "x1", "x2"])})


def test_positional_join_guard_accepts_identical_sample_sets(tmp_path):
    a, b = tmp_path / "arm1", tmp_path / "arm2"
    _write_index(a, plugin="none")
    _write_index(b, plugin="revin")
    assert_positional_join_allowed({"a": read_provenance(a, "val_index_order"),
                                    "b": read_provenance(b, "val_index_order")})


def test_positional_join_guard_refuses_legacy_sidecars(tmp_path):
    a, b = tmp_path / "arm1", tmp_path / "arm2"
    _write_index(a)
    _write_index(b)
    write_provenance(b, "val_index_order", {"order": "index", "n_windows": N})
    with pytest.raises(SchemaVersionError):
        assert_positional_join_allowed({"a": read_provenance(a, "val_index_order"),
                                        "b": read_provenance(b, "val_index_order")})


# --------------------------------------------------------------------------- #
# the backfilled archive
# --------------------------------------------------------------------------- #
def _archive_dirs(tag: str, k: int = 3):
    from src.config import REPO_ROOT

    root = REPO_ROOT / "results" / "winerr"
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.endswith(tag)][:k]


def test_backfilled_archive_reads_under_the_strict_contract():
    dirs = _archive_dirs("_shuf") + _archive_dirs("_det")
    if not dirs:
        pytest.skip("no results/winerr archive in this checkout")
    for d in dirs:
        for name, order in (("val_index_order", ORDER_INDEX), ("test_index_order", ORDER_INDEX),
                            ("val_loader_order", ORDER_LOADER), ("val_perm", ORDER_LOADER)):
            if not (d / f"{name}.npy").exists():
                continue
            sc = read_provenance(d, name)
            assert sc["schema_version"] == SCHEMA_VERSION, d.name
            load_series(d, name, require_order=order, legacy_ok=False)


def test_backfilled_loader_order_joins_back_by_sample_id():
    dirs = _archive_dirs("_shuf", k=1)
    if not dirs:
        pytest.skip("no results/winerr archive in this checkout")
    d = dirs[0]
    yi, ids_i = load_series_with_ids(d, "val_index_order")
    yl, ids_l = load_series_with_ids(d, "val_loader_order", require_order=ORDER_LOADER)
    assert ids_i != ids_l, "a shuffled cell must not claim the same ordered sample ids"
    assert not np.array_equal(yi, yl)
    out, ids = join_by_sample_id({"index": (yi, ids_i), "loader": (yl, ids_l)})
    assert len(ids) == yi.size
    assert np.array_equal(out["index"], out["loader"])  # id join repairs what position broke


# --------------------------------------------------------------------------- #
# the coverage table must stay honest
# --------------------------------------------------------------------------- #
def test_contract_coverage_table_is_self_consistent_and_admits_blind_spots():
    from tools.make_contract_coverage import run_scenarios

    df = run_scenarios()
    assert set(["fault", "detected_by_field", "detected", "note"]) <= set(df.columns)
    assert len(df) >= 20
    assert df["fault"].is_unique
    assert (df.loc[df["detected"] == 1, "exception"] != "accepted").all()
    assert (df.loc[df["detected"] == 0, "exception"] == "accepted").all()
    assert int((df["detected"] == 0).sum()) >= 1
    assert "two_producers_share_one_wrong_loader_order" in set(df[df["detected"] == 0]["fault"])
