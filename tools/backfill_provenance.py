"""Backfill the sample-level provenance contract onto per-window artefacts written before it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402
from src.provenance import (  # noqa: E402
    UNKNOWN,
    ProvenanceError,
    build_sidecar,
    is_legacy_sidecar,
    producer_commit,
    validate_sidecar,
)
from src.winerr import (  # noqa: E402
    ORDER_INDEX,
    ORDER_LOADER,
    ORDER_SEMANTICS_OF,
    load_series,
    read_provenance,
    resolve_sample_ids,
    write_provenance,
)

IDENT_KEYS = ("dataset", "backbone", "plugin", "pred_len", "seed", "seq_len", "val_order")
LEGACY_KEEP = ("order", "n_windows", "cell_id", "val_order", "note", "split")


def _order_of(name: str, old: dict | None) -> str:
    if old and old.get("order") in (ORDER_INDEX, ORDER_LOADER):
        return str(old["order"])
    return ORDER_LOADER if ("loader" in name or name == "val_perm") else ORDER_INDEX


def _split_of(name: str, old: dict | None) -> str:
    if old and old.get("split"):
        return str(old["split"])
    return "test" if name.startswith("test") else "val"


def plan_dir(d: Path, ident: dict, commit: str) -> dict:
    """Compute the new sidecar for every array in one cell directory."""
    names = sorted(p.stem for p in d.glob("*.npy"))
    if not names:
        return {"status": "skipped", "reason": "no .npy arrays", "sidecars": {}, "names": []}
    perm_path = d / "val_perm.npy"
    perm = np.load(perm_path).astype(np.int64) if perm_path.exists() else None
    if perm is not None and not np.array_equal(np.sort(perm), np.arange(perm.size)):
        return {"status": "failed", "reason": "val_perm is not a permutation of 0..n-1",
                "sidecars": {}, "names": names}
    sidecars: dict[str, dict] = {}
    n_already = 0
    for name in names:
        arr = np.load(d / f"{name}.npy")
        old = read_provenance(d, name)
        if old is not None and not is_legacy_sidecar(old):
            n_already += 1
            continue
        order = _order_of(name, old)
        split = _split_of(name, old)
        if order == ORDER_LOADER:
            if perm is None or perm.size != arr.size:
                return {"status": "failed",
                        "reason": f"{name}: loader-order array needs a matching val_perm",
                        "sidecars": {}, "names": names}
            origins = perm
            spec = {"kind": "sibling_array", "name": "val_perm"}
        else:
            origins = np.arange(arr.size, dtype=np.int64)
            spec = {"kind": "identity_range", "start": 0}
        extra = {k: old[k] for k in LEGACY_KEEP if old and k in old}
        extra.update({
            "order": order,
            "n_windows": int(arr.size),
            "split": split,
            "backfilled": True,
            "backfill_commit": commit,
            "backfill_tool": "tools/backfill_provenance.py",
        })
        rec = build_sidecar(
            values=arr,
            split=split,
            window_origins=origins,
            order_semantics=ORDER_SEMANTICS_OF[order],
            dataset=ident.get("dataset"),
            backbone=ident.get("backbone"),
            plugin=ident.get("plugin"),
            seq_len=ident.get("seq_len"),
            pred_len=ident.get("pred_len"),
            seed=ident.get("seed"),
            sample_id_spec=spec,
            commit=UNKNOWN,  # the producing commit of these arrays was never recorded
            extra=extra,
        )
        sidecars[name] = rec
    ok, why = _self_check(d, names, sidecars)
    if not ok:
        return {"status": "failed", "reason": why, "sidecars": {}, "names": names}
    if not sidecars:
        return {"status": "skipped", "reason": f"{n_already} sidecars already on contract",
                "sidecars": {}, "names": names}
    return {"status": "ok", "reason": f"{len(sidecars)} sidecars", "sidecars": sidecars,
            "names": names}


def _self_check(d: Path, names: list[str], sidecars: dict[str, dict]) -> tuple[bool, str]:
    """Validate every planned sidecar in memory, plus the loader/index/perm consistency."""
    for name, rec in sidecars.items():
        arr = np.load(d / f"{name}.npy")
        ids = resolve_sample_ids(d, rec)
        try:
            validate_sidecar(rec, arr, sample_ids=ids, label=f"{d.name}/{name}")
        except ProvenanceError as exc:
            return False, f"planned sidecar rejected: {exc}"
    need = {"val_index_order", "val_loader_order", "val_perm"}
    if need <= set(names):
        idx = np.load(d / "val_index_order.npy")
        ld = np.load(d / "val_loader_order.npy")
        perm = np.load(d / "val_perm.npy").astype(np.int64)
        if idx.size == ld.size == perm.size and not np.allclose(idx[perm], ld):
            return False, "val_loader_order != val_index_order[val_perm]"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--apply", action="store_true", help="actually write sidecars (default: dry run)")
    ap.add_argument("--verify", action="store_true",
                    help="after the pass, re-read every array with legacy_ok=False")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--winerr-dir", default=None, help="migrate a copy of the archive instead")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    cfg = GridConfig(args.config)
    root = Path(args.winerr_dir) if args.winerr_dir else cfg.path("winerr_dir")
    cells_dir = cfg.path("cells_dir")
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]
    commit = producer_commit()
    mode = "apply" if args.apply else "dry-run"
    print(f"[setup] {len(dirs)} directories under {root} | mode={mode} | tool commit {commit}")

    rows: list[dict] = []
    n_ok = n_failed = n_skipped = n_files = 0
    for d in dirs:
        cell_json = cells_dir / f"{d.name}.json"
        if cell_json.exists():
            rec = json.loads(cell_json.read_text(encoding="utf-8"))
            ident = {k: rec.get(k) for k in IDENT_KEYS}
        else:
            ident = {k: None for k in IDENT_KEYS}
        plan = plan_dir(d, ident, commit)
        if plan["status"] == "ok" and not cell_json.exists():
            plan = {**plan, "status": "failed", "reason": "no results/cells record for identity",
                    "sidecars": {}}
        if plan["status"] == "ok":
            if args.apply:
                for name, sc in plan["sidecars"].items():
                    write_provenance(d, name, sc)
            n_ok += 1
            n_files += len(plan["sidecars"])
        elif plan["status"] == "failed":
            n_failed += 1
            print(f"[fail] {d.name}: {plan['reason']}")
        else:
            n_skipped += 1
        rows.append({
            "cell_id": d.name,
            "status": plan["status"],
            "reason": plan["reason"],
            "n_arrays": len(plan["names"]),
            "n_sidecars_planned": len(plan["sidecars"]),
            "applied": int(bool(args.apply and plan["status"] == "ok")),
        })

    print(f"\n[backfill] directories: ok={n_ok} failed={n_failed} skipped={n_skipped} "
          f"(total {len(dirs)}); sidecars {'written' if args.apply else 'planned'}={n_files}")
    if not args.apply:
        print("[backfill] dry run: nothing was written, re-run with --apply")

    out = Path(args.report) if args.report else cfg.path("artifacts_dir") / "backfill_provenance_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[out] {out}")

    if args.verify:
        v_ok = v_bad = 0
        bad: list[str] = []
        for d in dirs:
            for p in sorted(d.glob("*.npy")):
                name = p.stem
                sc = read_provenance(d, name)
                order = _order_of(name, sc)
                try:
                    load_series(d, name, require_order=order, legacy_ok=False)
                    v_ok += 1
                except Exception as exc:
                    v_bad += 1
                    if len(bad) < 5:
                        bad.append(f"{d.name}/{name}: {type(exc).__name__}: {exc}")
        print(f"[verify] strict reads (legacy_ok=False): ok={v_ok} failed={v_bad}")
        for b in bad:
            print(f"  {b}")
        return 1 if v_bad else 0
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
