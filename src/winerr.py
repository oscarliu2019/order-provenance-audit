"""Per-window error recording and the order-provenance contract.

Vocabulary used throughout the code and the paper
-------------------------------------------------
*index order*   the order induced by the sliding-window index ``j = 0..n-1``.
                Window ``j`` covers input ``[j, j+L)`` and target
                ``[j+L, j+L+H)``. Every per-window *feature* table is written in
                index order, because features are computed from the raw series.
*loader order*  the order in which a ``DataLoader`` happens to yield windows. It
                equals index order iff the loader is not shuffled.

A per-window error vector is only joinable with a per-window feature table by
position when it is in index order. The defect studied in this paper is a vector
recorded in loader order and joined as if it were in index order.

Two mechanisms are provided here:

1. ``WindowRecorder`` optionally records, alongside each window error, the
   dataset index that produced it. This makes the permutation *observable*
   inside one pass, so a single evaluation yields both the vector a defective
   pipeline would have written and the ground-truth index-order vector.
2. ``save_series`` writes a sidecar provenance record and ``load_series``
   refuses to hand out a vector whose order cannot be established. The point is
   that the two files are byte-indistinguishable without the sidecar: the fix
   must be a *contract*, not a code comment. The contract itself -- schema
   version, split, sample-set identity, ordered sample-id digest, value digest,
   order semantics, cell identity, producer commit -- lives in ``src.provenance``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.provenance import (  # noqa: F401
    DECLARED_POSITION_SCHEME,
    KNOWN_ORDER_SEMANTICS,
    ORDER_DATASET_INDEX,
    ORDER_LOADER_EMISSION,
    SAMPLE_ID_SCHEME,
    SCHEMA_VERSION,
    DigestMismatchError,
    IdentityMismatchError,
    MissingFieldError,
    OrderSemanticsError,
    ProvenanceError,
    ProvenanceWarning,
    SchemaVersionError,
    build_sidecar,
    canonical_sample_ids,
    compute_ordered_sample_id_digest,
    compute_value_digest,
    is_legacy_sidecar,
    identity_complete,
    join_by_sample_id,
    validate_sidecar,
)

PROVENANCE_SUFFIX = ".provenance.json"
ORDER_INDEX = "index"
ORDER_LOADER = "loader"
KNOWN_ORDERS = (ORDER_INDEX, ORDER_LOADER)

ORDER_SEMANTICS_OF = {ORDER_INDEX: ORDER_DATASET_INDEX, ORDER_LOADER: ORDER_LOADER_EMISSION}


# --------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------- #
class WindowRecorder:
    """Streaming accumulator for per-window MSE.

    Memory is O(n_windows) for the error vector only; predictions are never
    materialised, which is what makes it affordable to record the validation
    split at every epoch.
    """

    def __init__(self, per_window: bool = True) -> None:
        self.per_window = bool(per_window)
        self.se = 0.0
        self.n = 0
        self._err: list[np.ndarray] = []
        self._idx: list[np.ndarray] = []

    def update(self, pred, true, index=None) -> None:
        import torch

        with torch.no_grad():
            d = (pred - true).float()
            self.se += float((d**2).sum())
            self.n += int(d.numel())
            if self.per_window:
                self._err.append(
                    (d**2).mean(dim=tuple(range(1, d.dim()))).detach().cpu().numpy().astype(np.float64)
                )
                if index is not None:
                    self._idx.append(np.asarray(index, dtype=np.int64).reshape(-1))

    @property
    def mse(self) -> float:
        return self.se / self.n if self.n else float("nan")

    def errors(self) -> np.ndarray:
        """Errors in *loader* order (the order the loader happened to yield)."""
        if not self._err:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self._err)

    def indices(self) -> np.ndarray:
        if not self._idx:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(self._idx)

    def errors_in_index_order(self) -> np.ndarray:
        """Errors re-keyed to index order using the captured dataset indices."""
        err = self.errors()
        perm = self.indices()
        if err.size == 0:
            return err
        if perm.size != err.size:
            raise ProvenanceError(
                "cannot re-key without one index per window "
                f"(got {perm.size} indices for {err.size} errors)"
            )
        return unpermute(err, perm)


def unpermute(recorded: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Invert a loader permutation.

    ``recorded[i]`` is the error of window ``perm[i]``; the returned vector
    ``out`` satisfies ``out[j] = error of window j``.
    """
    recorded = np.asarray(recorded)
    perm = np.asarray(perm, dtype=np.int64)
    if recorded.shape[0] != perm.shape[0]:
        raise ValueError("recorded and perm must have the same length")
    if not np.array_equal(np.sort(perm), np.arange(perm.size)):
        raise ValueError("perm must be a permutation of 0..n-1")
    out = np.empty_like(recorded)
    out[perm] = recorded
    return out


# --------------------------------------------------------------------------- #
# provenance contract
# --------------------------------------------------------------------------- #
def _npy(win_dir: Path, name: str) -> Path:
    return Path(win_dir) / f"{name}.npy"


def _split_from_name(name: str) -> str | None:
    for split in ("val", "test", "train"):
        if str(name).startswith(f"{split}_"):
            return split
    return None


def save_series(
    win_dir: str | Path,
    name: str,
    values: np.ndarray,
    order: str,
    *,
    split: str | None = None,
    window_origins: Iterable[Any] | None = None,
    sample_id_spec: Mapping[str, Any] | None = None,
    dataset: Any = None,
    backbone: Any = None,
    plugin: Any = None,
    seq_len: Any = None,
    pred_len: Any = None,
    seed: Any = None,
    **extra: Any,
) -> Path:
    """Write ``values`` plus a contract sidecar that pins order, length, content and identity.

    ``np.save`` appends ``.npy`` to any path that lacks it, so the temporary
    file has to be handed over as an already-open file object; otherwise the
    rename cannot find its source.
    """
    if order not in KNOWN_ORDERS:
        raise ValueError(f"order must be one of {KNOWN_ORDERS}, got {order!r}")
    win_dir = Path(win_dir)
    win_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float64)
    final = _npy(win_dir, name)
    tmp = win_dir / f".{name}.npy.tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, values)
    tmp.replace(final)

    split = split if split is not None else _split_from_name(name)
    if window_origins is None:
        # no origins supplied: the stored positions are all the producer can attest to
        origins: Iterable[Any] = np.arange(int(values.size), dtype=np.int64)
        scheme = SAMPLE_ID_SCHEME if order == ORDER_INDEX else DECLARED_POSITION_SCHEME
        spec = dict(sample_id_spec or {"kind": "identity_range", "start": 0})
    else:
        origins = np.asarray(list(window_origins), dtype=np.int64)
        scheme = SAMPLE_ID_SCHEME
        spec = dict(sample_id_spec or {"kind": "inline", "window_origins": [int(o) for o in origins]})
    rec = build_sidecar(
        values=values,
        split=split,
        window_origins=origins,
        order_semantics=ORDER_SEMANTICS_OF[order],
        dataset=dataset,
        backbone=backbone,
        plugin=plugin,
        seq_len=seq_len,
        pred_len=pred_len,
        seed=seed,
        sample_id_scheme=scheme,
        sample_id_spec=spec,
        extra={"order": order, "n_windows": int(values.size), **extra},
    )
    if not identity_complete(rec):
        warnings.warn(
            f"{final}: provenance identity is incomplete "
            f"(split/dataset/backbone/plugin/seq_len/pred_len/seed partly 'unknown'), so "
            f"cross-cell sample-set checks will be weak",
            ProvenanceWarning,
            stacklevel=2,
        )
    side = win_dir / f"{name}{PROVENANCE_SUFFIX}"
    tmp_side = win_dir / f".{name}{PROVENANCE_SUFFIX}.tmp"
    tmp_side.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    tmp_side.replace(side)
    return final


def read_provenance(win_dir: str | Path, name: str) -> dict[str, Any] | None:
    p = Path(win_dir) / f"{name}{PROVENANCE_SUFFIX}"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_provenance(win_dir: str | Path, name: str, rec: Mapping[str, Any]) -> Path:
    """Atomically replace one sidecar (used by the backfill tool)."""
    win_dir = Path(win_dir)
    side = win_dir / f"{name}{PROVENANCE_SUFFIX}"
    tmp = win_dir / f".{name}{PROVENANCE_SUFFIX}.tmp"
    tmp.write_text(json.dumps(dict(rec), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(side)
    return side


def resolve_sample_ids(
    win_dir: str | Path, sidecar: Mapping[str, Any]
) -> list[str] | None:
    """Rebuild the ordered sample id list a sidecar points at, or None if it cannot."""
    spec = sidecar.get("sample_id_spec") or {}
    kind = str(spec.get("kind", ""))
    scheme = str(sidecar.get("sample_id_scheme", SAMPLE_ID_SCHEME))
    n = int(sidecar.get("n_samples", -1))
    if kind == "inline" and "sample_ids" in spec:
        return [str(s) for s in spec["sample_ids"]]
    if kind == "inline" and "window_origins" in spec:
        origins: Any = spec["window_origins"]
    elif kind == "identity_range":
        origins = np.arange(int(spec.get("start", 0)), int(spec.get("start", 0)) + n)
    elif kind == "sibling_array":
        sib = _npy(Path(win_dir), str(spec["name"]))
        if not sib.exists():
            raise MissingFieldError(f"{sib}: sample_id_spec points at a missing sibling array")
        origins = np.load(sib).astype(np.int64)
    else:
        return None
    return canonical_sample_ids(
        sidecar.get("split"),
        sidecar.get("seq_len"),
        sidecar.get("pred_len"),
        origins,
        dataset=sidecar.get("dataset"),
        scheme=scheme,
    )


def _legacy_checks(path: Path, prov: Mapping[str, Any], require_order: str | None,
                   values: np.ndarray, *, require_order_field: bool = True) -> None:
    if require_order is not None and (require_order_field or "order" in prov):
        if prov.get("order") != require_order:
            raise IdentityMismatchError(
                f"{path} was recorded in {prov.get('order')!r} order but {require_order!r} order "
                "is required for a positional join."
            )
    if "n_windows" in prov and int(prov.get("n_windows", -1)) != int(values.size):
        raise IdentityMismatchError(
            f"{path} declares {prov.get('n_windows')} windows but holds {values.size}"
        )


def load_series(
    win_dir: str | Path,
    name: str,
    require_order: str | None = ORDER_INDEX,
    expect_n: int | None = None,
    *,
    legacy_ok: bool = False,
    expect: Mapping[str, Any] | None = None,
    return_sample_ids: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str] | None]:
    """Load a per-sample vector and enforce the provenance contract, fail closed.

    ``legacy_ok=True`` is the only way to read a pre-contract sidecar, and it warns.
    ``require_order=None`` opts out of the order check only; every other clause of the
    contract is still enforced whenever a sidecar exists.
    """
    path = _npy(win_dir, name)
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.load(path)
    sample_ids: list[str] | None = None
    prov = read_provenance(win_dir, name)
    if prov is None:
        if require_order is not None:
            raise MissingFieldError(
                f"{path} has no {PROVENANCE_SUFFIX} sidecar: its sample set and order are "
                "unknown, so a positional join with per-window features is not justified. "
                "Re-record it with src.train, backfill it with tools/backfill_provenance.py, "
                "or load it with require_order=None if you only need order-invariant statistics."
            )
        warnings.warn(
            f"{path}: no provenance sidecar; accepted only because require_order=None and no "
            f"sample-level claim is being made",
            ProvenanceWarning,
            stacklevel=2,
        )
    elif is_legacy_sidecar(prov):
        if not legacy_ok:
            raise SchemaVersionError(
                f"{path}: sidecar predates {SCHEMA_VERSION} (no schema_version field). "
                "Run tools/backfill_provenance.py --apply, or pass legacy_ok=True to accept "
                "the weaker order+length checks with a warning."
            )
        warnings.warn(
            f"{path}: legacy sidecar accepted with legacy_ok=True; only 'order' and "
            f"'n_windows' are verified, not sample-set identity or content",
            ProvenanceWarning,
            stacklevel=2,
        )
        _legacy_checks(path, prov, require_order, values)
    else:
        sample_ids = resolve_sample_ids(win_dir, prov)
        want = dict(expect or {})
        if require_order is not None:
            want.setdefault("order_semantics", ORDER_SEMANTICS_OF[require_order])
        validate_sidecar(prov, values, expect=want, sample_ids=sample_ids, label=str(path))
        _legacy_checks(path, prov, require_order, values, require_order_field=False)
    if expect_n is not None and int(values.size) != int(expect_n):
        raise IdentityMismatchError(
            f"{path}: expected {expect_n} windows, found {values.size}"
        )
    if return_sample_ids:
        return values, sample_ids
    return values


def load_series_with_ids(
    win_dir: str | Path, name: str, require_order: str | None = ORDER_INDEX, **kw: Any
) -> tuple[np.ndarray, list[str] | None]:
    """load_series plus the ordered sample ids, for join_by_sample_id."""
    out = load_series(win_dir, name, require_order, return_sample_ids=True, **kw)
    return out  # type: ignore[return-value]
