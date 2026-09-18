"""Sample-level provenance contract for per-window evaluation artefacts."""

from __future__ import annotations

import hashlib
import subprocess
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "order-audit/provenance/2"
KNOWN_SCHEMA_VERSIONS = (SCHEMA_VERSION,)

UNKNOWN = "unknown"

SAMPLE_ID_SCHEME = "window_origin/v1"
DECLARED_POSITION_SCHEME = "declared_position/v1"
KNOWN_SAMPLE_ID_SCHEMES = (SAMPLE_ID_SCHEME, DECLARED_POSITION_SCHEME)

ORDER_DATASET_INDEX = "dataset_index"
ORDER_LOADER_EMISSION = "loader_emission"
KNOWN_ORDER_SEMANTICS = (ORDER_DATASET_INDEX, ORDER_LOADER_EMISSION)

IDENTITY_FIELDS = ("dataset", "backbone", "plugin", "seq_len", "pred_len", "seed")
REQUIRED_FIELDS = (
    "schema_version",
    "split",
    "n_samples",
    "ordered_sample_id_digest",
    "value_digest",
    "order_semantics",
    "sample_id_scheme",
    *IDENTITY_FIELDS,
)
RECOMMENDED_FIELDS = ("producer_commit",)

SAMPLE_ID_DIGEST_DOMAIN = b"order-audit/ordered-sample-ids/v1\n"
VALUE_DIGEST_DOMAIN = b"order-audit/value-digest/v1\n"


class ProvenanceError(RuntimeError):
    """Base class for every fail-closed provenance rejection."""


class MissingFieldError(ProvenanceError):
    """A required sidecar field is absent."""


class SchemaVersionError(ProvenanceError):
    """The sidecar declares a schema this reader does not understand."""


class DigestMismatchError(ProvenanceError):
    """The array bytes do not match the digest the sidecar declares."""


class IdentityMismatchError(ProvenanceError):
    """The sample set, its order, or the cell identity is not the expected one."""


class OrderSemanticsError(ProvenanceError):
    """The declared order semantics is outside the contract enumeration."""


class ProvenanceWarning(UserWarning):
    """Emitted when a legacy or identity-incomplete artefact is accepted."""


# --------------------------------------------------------------------------- #
# canonical sample ids
# --------------------------------------------------------------------------- #
def _tok(value: Any) -> str:
    if value is None:
        return UNKNOWN
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def canonical_sample_id(
    split: Any,
    seq_len: Any,
    pred_len: Any,
    window_origin: Any,
    dataset: Any = UNKNOWN,
    scheme: str = SAMPLE_ID_SCHEME,
) -> str:
    """Canonical sample id: the window origin inside (dataset, split, seq_len, pred_len)."""
    if scheme not in KNOWN_SAMPLE_ID_SCHEMES:
        raise SchemaVersionError(f"unknown sample_id_scheme {scheme!r}")
    return (
        f"{scheme}|{_tok(dataset)}|{_tok(split)}|L{_tok(seq_len)}|H{_tok(pred_len)}|"
        f"w{_tok(window_origin)}"
    )


def canonical_sample_ids(
    split: Any,
    seq_len: Any,
    pred_len: Any,
    window_origins: Iterable[Any],
    dataset: Any = UNKNOWN,
    scheme: str = SAMPLE_ID_SCHEME,
) -> list[str]:
    """Canonical ids for an ordered iterable of window origins."""
    return [
        canonical_sample_id(split, seq_len, pred_len, o, dataset=dataset, scheme=scheme)
        for o in np.asarray(list(window_origins)).reshape(-1).tolist()
    ]


def compute_ordered_sample_id_digest(sample_ids: Sequence[Any]) -> str:
    """sha256 over the length-framed, ordered sample id list."""
    ids = list(sample_ids)
    h = hashlib.sha256()
    h.update(SAMPLE_ID_DIGEST_DOMAIN)
    h.update(f"count={len(ids)}\n".encode("utf-8"))
    for sid in ids:
        raw = _sample_id_bytes(sid)
        h.update(f"{len(raw)}:".encode("ascii"))
        h.update(raw)
        h.update(b"\n")
    return f"sha256:{h.hexdigest()}"


def _sample_id_bytes(sid: Any) -> bytes:
    if isinstance(sid, (tuple, list)):
        return "|".join(_tok(x) for x in sid).encode("utf-8")
    return _tok(sid).encode("utf-8")


def _canonical_bytes(arr: np.ndarray) -> tuple[np.ndarray, bytes]:
    a = np.asarray(arr)
    if a.dtype.kind == "f":
        a = a.astype("<f8", copy=True)
        a[a == 0] = 0.0  # collapse -0.0 so the digest tracks values, not sign bits
        a[np.isnan(a)] = np.nan
    elif a.dtype.kind in "iu":
        a = a.astype("<i8", copy=True)
    elif a.dtype.kind == "b":
        a = a.astype(np.uint8, copy=True)
    else:
        raise TypeError(f"cannot digest dtype {a.dtype!r}")
    return a, np.ascontiguousarray(a).tobytes()


def compute_value_digest(arr: np.ndarray) -> str:
    """sha256 over a dtype- and endianness-normalised view of the array bytes."""
    a, raw = _canonical_bytes(arr)
    h = hashlib.sha256()
    h.update(VALUE_DIGEST_DOMAIN)
    h.update(f"dtype={a.dtype.str}|shape={tuple(int(s) for s in a.shape)}\n".encode("utf-8"))
    h.update(raw)
    return f"sha256:{h.hexdigest()}"


# --------------------------------------------------------------------------- #
# producer commit
# --------------------------------------------------------------------------- #
_COMMIT_CACHE: dict[str, str] = {}


def producer_commit(repo_root: str | Path | None = None) -> str:
    """Current git commit of the producing code, or the explicit string 'unknown'."""
    root = str(Path(repo_root or Path(__file__).resolve().parent.parent))
    if root not in _COMMIT_CACHE:
        try:
            out = subprocess.run(
                ["git", "-C", root, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            _COMMIT_CACHE[root] = out.stdout.strip() or UNKNOWN
        except Exception:
            _COMMIT_CACHE[root] = UNKNOWN
    return _COMMIT_CACHE[root]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_sidecar(
    *,
    values: np.ndarray,
    split: Any,
    sample_ids: Sequence[Any] | None = None,
    window_origins: Iterable[Any] | None = None,
    order_semantics: str,
    dataset: Any = None,
    backbone: Any = None,
    plugin: Any = None,
    seq_len: Any = None,
    pred_len: Any = None,
    seed: Any = None,
    sample_id_scheme: str = SAMPLE_ID_SCHEME,
    sample_id_spec: Mapping[str, Any] | None = None,
    commit: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-complete sidecar for one per-sample artefact."""
    if order_semantics not in KNOWN_ORDER_SEMANTICS:
        raise OrderSemanticsError(
            f"order_semantics must be one of {KNOWN_ORDER_SEMANTICS}, got {order_semantics!r}"
        )
    if sample_id_scheme not in KNOWN_SAMPLE_ID_SCHEMES:
        raise SchemaVersionError(f"unknown sample_id_scheme {sample_id_scheme!r}")
    values = np.asarray(values)
    n = int(values.shape[0]) if values.ndim else 1
    origins: list[int] | None = None
    if sample_ids is None:
        if window_origins is None:
            raise MissingFieldError("build_sidecar needs sample_ids or window_origins")
        origins = [int(o) for o in np.asarray(list(window_origins)).reshape(-1).tolist()]
        sample_ids = canonical_sample_ids(
            split, seq_len, pred_len, origins, dataset=dataset or UNKNOWN,
            scheme=sample_id_scheme,
        )
    sample_ids = list(sample_ids)
    default_spec: dict[str, Any] = (
        {"kind": "inline", "window_origins": origins}
        if origins is not None
        else {"kind": "inline", "sample_ids": [_tok(s) for s in sample_ids]}
    )
    if len(sample_ids) != n:
        raise IdentityMismatchError(
            f"{len(sample_ids)} sample ids for {n} values"
        )
    if len(set(_tok(s) for s in sample_ids)) != n:
        raise IdentityMismatchError("sample ids are not unique")
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split": _tok(split),
        "n_samples": n,
        "ordered_sample_id_digest": compute_ordered_sample_id_digest(sample_ids),
        "value_digest": compute_value_digest(values),
        "order_semantics": order_semantics,
        "sample_id_scheme": sample_id_scheme,
        "sample_id_spec": dict(sample_id_spec or default_spec),
        "dataset": _tok(dataset),
        "backbone": _tok(backbone),
        "plugin": _tok(plugin),
        "seq_len": int(seq_len) if isinstance(seq_len, (int, np.integer)) else _tok(seq_len),
        "pred_len": int(pred_len) if isinstance(pred_len, (int, np.integer)) else _tok(pred_len),
        "seed": int(seed) if isinstance(seed, (int, np.integer)) else _tok(seed),
        "producer_commit": commit if commit is not None else producer_commit(),
    }
    if extra:
        for k, v in extra.items():
            if k not in rec:
                rec[k] = v
    return rec


def identity_complete(sidecar: Mapping[str, Any]) -> bool:
    """True when no contract field is the explicit 'unknown' sentinel."""
    keys = ("split", *IDENTITY_FIELDS)
    return all(_tok(sidecar.get(k)) != UNKNOWN for k in keys)


def is_legacy_sidecar(sidecar: Mapping[str, Any] | None) -> bool:
    """True for a pre-contract sidecar, i.e. one with no schema_version at all."""
    return sidecar is not None and "schema_version" not in sidecar


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def validate_sidecar(
    sidecar: Mapping[str, Any] | None,
    arr: np.ndarray,
    expect: Mapping[str, Any] | None = None,
    sample_ids: Sequence[Any] | None = None,
    label: str = "artefact",
) -> None:
    """Fail closed: raise unless the sidecar, the array and the expectations all agree."""
    if sidecar is None:
        raise MissingFieldError(f"{label}: no provenance sidecar")
    if "schema_version" not in sidecar:
        raise SchemaVersionError(
            f"{label}: sidecar has no schema_version (legacy pre-contract artefact); "
            f"backfill it with tools/backfill_provenance.py or pass legacy_ok=True"
        )
    if sidecar["schema_version"] not in KNOWN_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"{label}: unknown schema_version {sidecar['schema_version']!r}, "
            f"this reader speaks {KNOWN_SCHEMA_VERSIONS}"
        )
    missing = [f for f in REQUIRED_FIELDS if f not in sidecar]
    if missing:
        raise MissingFieldError(f"{label}: sidecar is missing {missing}")
    for f in RECOMMENDED_FIELDS:
        if f not in sidecar:
            warnings.warn(f"{label}: sidecar has no {f}", ProvenanceWarning, stacklevel=2)
    if sidecar["order_semantics"] not in KNOWN_ORDER_SEMANTICS:
        raise OrderSemanticsError(
            f"{label}: order_semantics {sidecar['order_semantics']!r} is not in "
            f"{KNOWN_ORDER_SEMANTICS}"
        )
    if sidecar["sample_id_scheme"] not in KNOWN_SAMPLE_ID_SCHEMES:
        raise SchemaVersionError(
            f"{label}: sample_id_scheme {sidecar['sample_id_scheme']!r} is not in "
            f"{KNOWN_SAMPLE_ID_SCHEMES}"
        )
    arr = np.asarray(arr)
    n = int(arr.shape[0]) if arr.ndim else 1
    if int(sidecar["n_samples"]) != n:
        raise IdentityMismatchError(
            f"{label}: sidecar declares n_samples={sidecar['n_samples']} but the array holds {n}"
        )
    got = compute_value_digest(arr)
    if got != sidecar["value_digest"]:
        raise DigestMismatchError(
            f"{label}: value_digest mismatch, sidecar says {sidecar['value_digest']} "
            f"but the array hashes to {got} (array edited, replaced, reordered, or the "
            f"sidecar is stale)"
        )
    if sample_ids is not None:
        ids = [_tok(s) for s in sample_ids]
        if len(ids) != n:
            raise IdentityMismatchError(
                f"{label}: {len(ids)} resolved sample ids for {n} values"
            )
        dig = compute_ordered_sample_id_digest(ids)
        if dig != sidecar["ordered_sample_id_digest"]:
            raise IdentityMismatchError(
                f"{label}: ordered_sample_id_digest mismatch, sidecar says "
                f"{sidecar['ordered_sample_id_digest']} but the resolved sample ids hash to {dig}"
            )
        if (
            sidecar["order_semantics"] == ORDER_DATASET_INDEX
            and sidecar["sample_id_scheme"] == SAMPLE_ID_SCHEME
        ):
            # dataset_index order means position i must be window origin i, no exceptions
            want_ids = canonical_sample_ids(
                sidecar["split"], sidecar["seq_len"], sidecar["pred_len"], range(n),
                dataset=sidecar["dataset"], scheme=SAMPLE_ID_SCHEME,
            )
            if ids != want_ids:
                first = next(i for i, (a, b) in enumerate(zip(ids, want_ids)) if a != b)
                raise IdentityMismatchError(
                    f"{label}: order_semantics={ORDER_DATASET_INDEX} requires position i to hold "
                    f"window origin i, but position {first} holds {ids[first]!r}"
                )
    if expect:
        for key, want in expect.items():
            if want is None:
                continue
            if key not in sidecar:
                raise MissingFieldError(f"{label}: sidecar has no {key} to check against")
            if _tok(sidecar[key]) != _tok(want):
                raise IdentityMismatchError(
                    f"{label}: expected {key}={want!r} but the sidecar declares "
                    f"{sidecar[key]!r}"
                )


def assert_positional_join_allowed(
    sidecars: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    n: int | None = None,
) -> None:
    """Refuse a positional join unless every product covers the same ordered sample set."""
    items = (
        list(sidecars.items())
        if isinstance(sidecars, Mapping)
        else [(f"[{i}]", s) for i, s in enumerate(sidecars)]
    )
    if len(items) < 2:
        return
    for label, sc in items:
        if sc is None or is_legacy_sidecar(sc):
            raise SchemaVersionError(
                f"{label}: positional join needs a contract sidecar, got a legacy one"
            )
    space_keys = ("dataset", "split", "seq_len", "pred_len", "sample_id_scheme", "order_semantics")
    ref_label, ref = items[0]
    for label, sc in items[1:]:
        for k in space_keys:
            if _tok(sc.get(k)) != _tok(ref.get(k)):
                raise IdentityMismatchError(
                    f"positional join refused: {label} has {k}={sc.get(k)!r} but {ref_label} "
                    f"has {k}={ref.get(k)!r}"
                )
    sizes = {int(sc["n_samples"]) for _, sc in items}
    truncated = n is not None and (len(sizes) > 1 or int(n) != next(iter(sizes)))
    if truncated:
        warnings.warn(
            f"positional join truncated to n={n} over n_samples={sorted(sizes)}: only the "
            f"sample space is verified, not the ordered digest",
            ProvenanceWarning,
            stacklevel=2,
        )
        return
    digests = {sc["ordered_sample_id_digest"] for _, sc in items}
    if len(digests) != 1:
        detail = ", ".join(f"{lab}={sc['ordered_sample_id_digest'][:23]}" for lab, sc in items)
        raise IdentityMismatchError(
            f"positional join refused: ordered_sample_id_digest differs ({detail})"
        )


# --------------------------------------------------------------------------- #
# join
# --------------------------------------------------------------------------- #
def _as_pair(label: str, obj: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(obj, Mapping):
        values = obj.get("values")
        ids = obj.get("sample_ids")
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        values, ids = obj
    else:
        raise MissingFieldError(f"{label}: expected (values, sample_ids), got {type(obj)!r}")
    if values is None:
        raise MissingFieldError(f"{label}: no values")
    if ids is None:
        raise MissingFieldError(
            f"{label}: no sample_ids, so it can only be joined positionally, which this "
            f"contract refuses"
        )
    values = np.asarray(values)
    ids = [_tok(s) for s in list(ids)]
    if len(ids) != (int(values.shape[0]) if values.ndim else 1):
        raise IdentityMismatchError(f"{label}: {len(ids)} sample ids for {values.shape} values")
    if len(set(ids)) != len(ids):
        raise IdentityMismatchError(f"{label}: duplicate sample ids")
    return values, ids


def join_by_sample_id(
    series_dict: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Align several per-sample products on the intersection of their sample ids."""
    if not series_dict:
        raise MissingFieldError("join_by_sample_id needs at least one series")
    pairs = {k: _as_pair(k, v) for k, v in series_dict.items()}
    labels = list(pairs)
    keep = set(pairs[labels[0]][1])
    for lab in labels[1:]:
        keep &= set(pairs[lab][1])
    ordered = [s for s in pairs[labels[0]][1] if s in keep]
    if not ordered:
        raise IdentityMismatchError(
            f"join_by_sample_id: the sample id sets of {labels} do not intersect"
        )
    out: dict[str, np.ndarray] = {}
    for lab in labels:
        values, ids = pairs[lab]
        pos = {s: i for i, s in enumerate(ids)}
        out[lab] = values[np.asarray([pos[s] for s in ordered], dtype=np.int64)]
    return out, ordered
