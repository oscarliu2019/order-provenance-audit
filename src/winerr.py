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
   must be a *contract*, not a code comment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROVENANCE_SUFFIX = ".provenance.json"
ORDER_INDEX = "index"
ORDER_LOADER = "loader"
KNOWN_ORDERS = (ORDER_INDEX, ORDER_LOADER)


class ProvenanceError(RuntimeError):
    """Raised when a per-window vector's order cannot be established."""


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


def save_series(
    win_dir: str | Path,
    name: str,
    values: np.ndarray,
    order: str,
    **extra: Any,
) -> Path:
    """Write ``values`` plus a sidecar declaring its order.

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
    rec = {"order": order, "n_windows": int(values.size), **extra}
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


def load_series(
    win_dir: str | Path,
    name: str,
    require_order: str | None = ORDER_INDEX,
    expect_n: int | None = None,
) -> np.ndarray:
    """Load a per-window vector, enforcing the order contract.

    ``require_order=None`` opts out, which is what the pre-fix pipeline did
    implicitly. Every positional join in this repository passes
    ``require_order="index"``.
    """
    path = _npy(win_dir, name)
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.load(path)
    if require_order is not None:
        prov = read_provenance(win_dir, name)
        if prov is None:
            raise ProvenanceError(
                f"{path} has no {PROVENANCE_SUFFIX} sidecar: its window order is unknown, so a "
                "positional join with per-window features is not justified. Re-record it with "
                "src.train, or load it with require_order=None if you only need order-invariant "
                "statistics."
            )
        if prov.get("order") != require_order:
            raise ProvenanceError(
                f"{path} was recorded in {prov.get('order')!r} order but {require_order!r} order "
                "is required for a positional join."
            )
        if int(prov.get("n_windows", -1)) != int(values.size):
            raise ProvenanceError(
                f"{path} declares {prov.get('n_windows')} windows but holds {values.size}"
            )
    if expect_n is not None and int(values.size) != int(expect_n):
        raise ProvenanceError(f"{path}: expected {expect_n} windows, found {values.size}")
    return values
