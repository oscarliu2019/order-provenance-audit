"""Machine-generated coverage table: which faults the provenance contract catches, and which it cannot."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.provenance import (  # noqa: E402
    ORDER_DATASET_INDEX,
    ProvenanceError,
    ProvenanceWarning,
    assert_positional_join_allowed,
    build_sidecar,
    join_by_sample_id,
)
from src.seeds import stable_rng  # noqa: E402
from src.winerr import (  # noqa: E402
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
N = 64


def _rng(tag: str) -> np.random.Generator:
    return stable_rng("contract-coverage", tag)


def write_cell(d: Path, *, n: int = N, split: str = "val", pred_len: int = 336,
               dataset: str = "ETTh1", shuffled: bool = True) -> dict:
    """One cell directory with the four products a real run writes."""
    d.mkdir(parents=True, exist_ok=True)
    ident = {**IDENT, "dataset": dataset, "pred_len": pred_len}
    y = _rng(f"{dataset}{split}{pred_len}{n}").gamma(2.0, 1.0, size=n)
    perm = _rng("perm").permutation(n) if shuffled else np.arange(n)
    save_series(d, f"{split}_index_order", y, ORDER_INDEX, split=split,
                window_origins=np.arange(n), sample_id_spec={"kind": "identity_range", "start": 0},
                **ident)
    save_series(d, f"{split}_loader_order", y[perm], ORDER_LOADER, split=split,
                window_origins=perm, sample_id_spec={"kind": "sibling_array", "name": "val_perm"},
                **ident)
    save_series(d, "val_perm", perm.astype(np.float64), ORDER_LOADER, split=split,
                window_origins=perm, sample_id_spec={"kind": "sibling_array", "name": "val_perm"},
                **ident)
    return {"values": y, "perm": perm, "ident": ident, "n": n, "split": split}


def _edit_sidecar(d: Path, name: str, **changes) -> None:
    rec = read_provenance(d, name)
    rec.update(changes)
    write_provenance(d, name, rec)


def _drop_field(d: Path, name: str, field: str) -> None:
    rec = read_provenance(d, name)
    rec.pop(field, None)
    write_provenance(d, name, rec)


def _overwrite_npy(d: Path, name: str, values: np.ndarray) -> None:
    np.save(d / f"{name}.npy", np.asarray(values, dtype=np.float64))


def _probe(fn: Callable[[], object]) -> tuple[int, str]:
    """Run a consumer action; detected=1 iff the contract raises a ProvenanceError."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProvenanceWarning)
        try:
            fn()
        except ProvenanceError as exc:
            return 1, f"{type(exc).__name__}"
    return 0, "accepted"


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def scenarios(root: Path) -> list[dict]:
    rows: list[dict] = []

    def add(fault: str, field: str, note: str, fn: Callable[[], object]) -> None:
        detected, exc = _probe(fn)
        rows.append({"fault": fault, "detected_by_field": field, "detected": detected,
                     "exception": exc, "note": note})

    def cell(tag: str, **kw) -> tuple[Path, dict]:
        d = root / tag
        if d.exists():
            shutil.rmtree(d)
        return d, write_cell(d, **kw)

    # ---- content faults ----
    d, c = cell("edit_one")
    _overwrite_npy(d, "val_index_order", np.concatenate([[c["values"][0] * 1.01], c["values"][1:]]))
    add("single_element_edited", "value_digest",
        "one entry rewritten; length, order label and sample ids all still agree",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("tiny")
    y = c["values"].copy()
    y[7] += 1e-12
    _overwrite_npy(d, "val_index_order", y)
    add("float_perturbation_1e-12", "value_digest",
        "sha256 over normalised float64 bytes is bit-exact, so sub-tolerance edits are caught",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("rot")
    _overwrite_npy(d, "val_index_order", np.roll(c["values"], 13))
    add("cyclic_rotation_of_the_whole_array", "value_digest",
        "n, mean, sum and multiset are all invariant under rotation; the byte digest is not",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("shuf")
    _overwrite_npy(d, "val_index_order", c["values"][_rng("attack").permutation(c["n"])])
    add("random_permutation_of_the_array", "value_digest",
        "the multiset is unchanged, so aggregate MSE and length checks stay silent",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("trunc")
    _overwrite_npy(d, "val_index_order", c["values"][:-1])
    add("array_truncated_by_one_sample", "n_samples",
        "length clause fires before the digest clause",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("stale")
    _overwrite_npy(d, "val_index_order", _rng("rerun").gamma(2.0, 1.0, size=c["n"]))
    add("stale_sidecar_array_rerun_in_place", "value_digest",
        "sidecar describes the previous epoch's array; same shape, same identity",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("swap_other", pred_len=336)
    other = root / "swap_other_src"
    if other.exists():
        shutil.rmtree(other)
    c2 = write_cell(other, pred_len=336, dataset="ETTh2")
    _overwrite_npy(d, "val_index_order", c2["values"])
    add("array_replaced_by_another_cell_same_length", "value_digest",
        "equal-length vector from a different dataset dropped in place of the right one",
        lambda: load_series(d, "val_index_order"))

    # ---- identity faults ----
    d, c = cell("split_swap")
    add("val_product_consumed_as_test", "split",
        "consumer states expect={'split': 'test'}; the sidecar says val",
        lambda: load_series(d, "val_index_order", expect={"split": "test"}))

    d, c = cell("horizon")
    add("same_length_different_horizon_joined", "ordered_sample_id_digest",
        "H=192 and H=336 products of one dataset can coincide in length; the sample ids cannot",
        lambda: _join_two_horizons(root))

    d, c = cell("dataset_mix")
    add("same_length_different_dataset_joined", "ordered_sample_id_digest",
        "ETTh1 and ETTh2 have identical split geometry, so only the id namespace separates them",
        lambda: _join_two_datasets(root))

    d, c = cell("relabel")
    _edit_sidecar(d, "val_loader_order", order=ORDER_INDEX, order_semantics=ORDER_DATASET_INDEX)
    add("loader_order_relabelled_as_index_order", "order_semantics+ordered_sample_id_digest",
        "dataset_index order is only accepted when position i really is window origin i",
        lambda: load_series(d, "val_loader_order", require_order=ORDER_INDEX))

    d, c = cell("stale_perm")
    _overwrite_npy(d, "val_perm", _rng("newperm").permutation(c["n"]).astype(np.float64))
    add("sample_id_source_replaced_val_perm", "ordered_sample_id_digest",
        "loader-order ids are resolved from val_perm.npy, so a stale perm breaks the digest",
        lambda: load_series(d, "val_loader_order", require_order=ORDER_LOADER))

    d, c = cell("pos_join")
    other2 = root / "pos_join_other"
    if other2.exists():
        shutil.rmtree(other2)
    write_cell(other2, pred_len=192)
    add("positional_join_across_different_sample_sets", "ordered_sample_id_digest",
        "the guard that a performance-motivated positional read must pass first",
        lambda: assert_positional_join_allowed(
            {"a": read_provenance(d, "val_index_order"),
             "b": read_provenance(other2, "val_index_order")}))

    d, c = cell("join_missing")
    add("join_of_a_product_with_no_sample_ids", "sample_id_spec",
        "join_by_sample_id refuses to fall back to positional alignment",
        lambda: join_by_sample_id({"a": (c["values"], None), "b": (c["values"], range(c["n"]))}))

    # ---- schema faults ----
    d, c = cell("legacy")
    write_provenance(d, "val_index_order",
                     {"order": "index", "n_windows": c["n"], "split": "val", "cell_id": "x"})
    add("legacy_sidecar_without_schema_version", "schema_version",
        "the 356 archived directories looked like this; legacy_ok=True downgrades to a warning",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("future")
    _edit_sidecar(d, "val_index_order", schema_version="order-audit/provenance/999")
    add("unknown_future_schema_version", "schema_version",
        "an unrecognised version is refused even with legacy_ok=True",
        lambda: load_series(d, "val_index_order", legacy_ok=True))

    d, c = cell("nofield")
    _drop_field(d, "val_index_order", "ordered_sample_id_digest")
    add("required_field_removed_from_sidecar", "ordered_sample_id_digest",
        "every field of the contract is checked for presence before use",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("badsem")
    _edit_sidecar(d, "val_index_order", order_semantics="whatever_the_loader_did")
    add("order_semantics_outside_the_enumeration", "order_semantics",
        "free-text order labels are how the original defect hid",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("nosidecar")
    (d / f"val_index_order{PROVENANCE_SUFFIX}").unlink()
    add("sidecar_file_deleted", "schema_version",
        "a missing sidecar is refused for any positional consumer",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("dupids")
    add("duplicate_sample_ids_in_one_product", "ordered_sample_id_digest",
        "build_sidecar refuses a non-injective sample id list",
        lambda: build_sidecar(values=c["values"], split="val",
                              window_origins=np.zeros(c["n"], dtype=np.int64),
                              order_semantics=ORDER_DATASET_INDEX, **c["ident"]))

    # ---- honest blind spots ----
    d, c = cell("shared_wrong")
    add("two_producers_share_one_wrong_loader_order", "(none)",
        "each product is self-consistent and both declare the same ordered ids, so the "
        "contract sees a legitimate loader-order join; only the AOT/CAT detectors or the "
        "raw dataset indices can reveal that the order is semantically wrong",
        lambda: _shared_wrong_order(root))

    d, c = cell("optout")
    (d / f"val_index_order{PROVENANCE_SUFFIX}").unlink()
    add("no_sidecar_but_consumer_passes_require_order_None", "(none)",
        "kept for order-invariant consumers of pre-contract arrays; warns loudly but does "
        "not raise, so it is an explicit opt-out hole rather than a checked path",
        lambda: load_series(d, "val_index_order", require_order=None))

    d, c = cell("truncjoin")
    other3 = root / "truncjoin_other"
    if other3.exists():
        shutil.rmtree(other3)
    write_cell(other3, n=N - 4)
    add("positional_join_truncated_to_n_min", "(none)",
        "when products have different lengths the ordered digests are not comparable, so "
        "only dataset/split/seq_len/pred_len are verified and a prefix misalignment survives",
        lambda: assert_positional_join_allowed(
            {"a": read_provenance(d, "val_index_order"),
             "b": read_provenance(other3, "val_index_order")}, n=N - 4))

    d, c = cell("upstream")
    y2 = c["values"] * 1.03
    _overwrite_npy(d, "val_index_order", y2)
    rec = read_provenance(d, "val_index_order")
    write_provenance(d, "val_index_order",
                     build_sidecar(values=y2, split="val", window_origins=np.arange(c["n"]),
                                   order_semantics=ORDER_DATASET_INDEX,
                                   sample_id_spec=rec["sample_id_spec"], **c["ident"]))
    add("upstream_data_or_preprocessing_changed", "(none)",
        "sample ids are (dataset, split, seq_len, pred_len, window_origin) and carry no hash "
        "of the raw series, so a re-derived product from changed input data validates and "
        "joins cleanly with features computed from the other version",
        lambda: load_series(d, "val_index_order"))

    d, c = cell("wrongslice")
    y3 = c["values"] + 0.5
    _overwrite_npy(d, "val_index_order", y3)
    write_provenance(d, "val_index_order",
                     build_sidecar(values=y3, split="val", window_origins=np.arange(c["n"]),
                                   order_semantics=ORDER_DATASET_INDEX,
                                   sample_id_spec={"kind": "identity_range", "start": 0},
                                   **c["ident"]))
    add("per_sample_value_computed_wrongly_but_consistently", "(none)",
        "the contract certifies who each number belongs to, never that the number is the "
        "right statistic; a wrong reduction with a fresh digest is indistinguishable",
        lambda: load_series(d, "val_index_order"))

    return rows


def _join_two_horizons(root: Path) -> None:
    a, b = root / "h336", root / "h192"
    for p in (a, b):
        if p.exists():
            shutil.rmtree(p)
    ca = write_cell(a, pred_len=336)
    cb = write_cell(b, pred_len=192)
    va, ia = load_series_with_ids(a, "val_index_order")
    vb, ib = load_series_with_ids(b, "val_index_order")
    assert va.size == vb.size and ca["n"] == cb["n"]
    join_by_sample_id({"h336": (va, ia), "h192": (vb, ib)})


def _join_two_datasets(root: Path) -> None:
    a, b = root / "etth1", root / "etth2"
    for p in (a, b):
        if p.exists():
            shutil.rmtree(p)
    write_cell(a, dataset="ETTh1")
    write_cell(b, dataset="ETTh2")
    va, ia = load_series_with_ids(a, "val_index_order")
    vb, ib = load_series_with_ids(b, "val_index_order")
    join_by_sample_id({"ETTh1": (va, ia), "ETTh2": (vb, ib)})


def _shared_wrong_order(root: Path) -> None:
    """Both arms record in the same loader order and both digests agree with it."""
    a, b = root / "wrong_a", root / "wrong_b"
    for p in (a, b):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)
    perm = _rng("shared").permutation(N)
    out = {}
    for tag, p in (("a", a), ("b", b)):
        y = _rng(f"vals{tag}").gamma(2.0, 1.0, size=N)
        save_series(p, "val_loader_order", y[perm], ORDER_LOADER, split="val",
                    window_origins=perm,
                    sample_id_spec={"kind": "inline", "window_origins": [int(i) for i in perm]},
                    **IDENT)
        v, ids = load_series_with_ids(p, "val_loader_order", require_order=ORDER_LOADER)
        out[tag] = (v, ids)
    assert_positional_join_allowed(
        {"a": read_provenance(a, "val_loader_order"), "b": read_provenance(b, "val_loader_order")}
    )
    join_by_sample_id(out)


def run_scenarios(root: str | Path | None = None) -> pd.DataFrame:
    """Execute every fault scenario and return the coverage table."""
    if root is None:
        tmp = tempfile.mkdtemp(prefix="contract-coverage-")
        try:
            return pd.DataFrame(scenarios(Path(tmp)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(scenarios(Path(root)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = run_scenarios()
    default_out = Path(__file__).resolve().parent.parent / "artifacts" / "contract_coverage.csv"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_det = int(df["detected"].sum())
    print(df[["fault", "detected_by_field", "detected", "exception"]].to_string(index=False))
    print(f"\n[coverage] {len(df)} faults: detected={n_det} undetected={len(df) - n_det}")
    print(json.dumps(df[df["detected"] == 0]["fault"].tolist(), indent=2))
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
