"""Declarative grid config plus the on-disk layout shared by every stage."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

VAL_ORDERS = ("deterministic", "shuffled")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass(frozen=True)
class Cell:
    """One training run. ``val_order`` is part of the identity on purpose.

    Two cells that differ only in ``val_order`` are the paired unit of stage E2:
    the model, the data and the seed are identical, so any difference in the
    reported test error is attributable to the evaluation loader alone.
    """

    dataset: str
    backbone: str
    plugin: str
    pred_len: int
    seed: int
    val_order: str = "deterministic"

    def __post_init__(self) -> None:
        if self.val_order not in VAL_ORDERS:
            raise ValueError(f"val_order must be one of {VAL_ORDERS}, got {self.val_order!r}")

    @property
    def cell_id(self) -> str:
        tag = "det" if self.val_order == "deterministic" else "shuf"
        return f"{self.backbone}_{self.dataset}_h{self.pred_len}_{self.plugin}_s{self.seed}_{tag}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "backbone": self.backbone,
            "plugin": self.plugin,
            "pred_len": int(self.pred_len),
            "seed": int(self.seed),
            "val_order": self.val_order,
        }


class GridConfig:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path_yaml = Path(path or REPO_ROOT / "configs" / "grid.yaml")
        with open(self.path_yaml, encoding="utf-8") as fh:
            self.raw: dict[str, Any] = yaml.safe_load(fh)
        self.meta = self.raw["meta"]
        self.protocol = self.meta["protocol"]
        self._ds = {d["name"]: d for d in self.raw["datasets"]}
        self._bk = {b["name"]: b for b in self.raw["backbones"]}
        self._pl = {p["name"]: p for p in self.raw["plugins"]}

    # ---------------- lookups ---------------- #
    def dataset(self, name: str) -> dict[str, Any]:
        return self._ds[name]

    def backbone(self, name: str) -> dict[str, Any]:
        return self._bk[name]

    def plugin(self, name: str) -> dict[str, Any]:
        return self._pl[name]

    def plugin_impl(self, name: str) -> str:
        return str(self._pl[name]["impl"])

    def seq_len(self, dataset: str) -> int:
        return int(self._ds[dataset].get("seq_len", self.protocol["seq_len"]))

    def label_len(self, dataset: str) -> int:
        return int(self._ds[dataset].get("label_len", self.protocol["label_len"]))

    def path(self, key: str) -> Path:
        return REPO_ROOT / self.raw["paths"][key]

    # ---------------- grid expansion ---------------- #
    def _horizons(self, dataset: str, spec: Any) -> list[int]:
        hs = list(self._ds[dataset]["horizons"])
        if spec == "all":
            return hs
        out: list[int] = []
        alias = {"first": 0, "second": 1, "third": 2, "fourth": 3}
        for s in spec:
            if isinstance(s, str):
                out.append(hs[alias[s]])
            else:
                out.append(int(s))
        return out

    def stage_cells(self, stage: str) -> list[Cell]:
        spec = self.raw[f"stage_{stage}"]
        orders = spec.get("val_orders") or [spec.get("val_order", "deterministic")]
        cells: list[Cell] = []
        for group in spec["cells"]:
            for ds in group["datasets"]:
                for h in self._horizons(ds, group["horizons"]):
                    for pl in group["plugins"]:
                        for seed in spec["seeds"]:
                            for order in orders:
                                cells.append(
                                    Cell(
                                        dataset=ds,
                                        backbone=group["backbone"],
                                        plugin=pl,
                                        pred_len=int(h),
                                        seed=int(seed),
                                        val_order=order,
                                    )
                                )
        # stable de-duplication: E1 and E2 overlap on (DLinear, ETT*, seed 2021)
        seen: set[str] = set()
        uniq: list[Cell] = []
        for c in cells:
            if c.cell_id not in seen:
                seen.add(c.cell_id)
                uniq.append(c)
        return uniq
