"""Single-cell trainer.

Only three things are imported from the upstream Time-Series-Library: the data
provider, the backbone definitions and the learning-rate schedule. The training
loop is local because the upstream ``test()`` materialises every prediction,
which is unaffordable at ``pred_len=720`` on a 32 GB device, and because the
validation loader has to be swappable — that switch is the independent variable
of this study.

The one upstream line under audit is
``third_party/tslib/data_provider/data_factory.py``::

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True

so ``data_provider(args, "val")`` yields a shuffled loader. This file reproduces
that behaviour verbatim when ``cell.val_order == "shuffled"`` and replaces it
with a deterministic loader when ``cell.val_order == "deterministic"``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, Cell, GridConfig, atomic_write_json  # noqa: E402
from src.fft_compat import apply_fft_fp32_patch  # noqa: E402
from src.plugins import kind_of, make_criterion, wrap  # noqa: E402
from src.winerr import ORDER_INDEX, ORDER_LOADER, WindowRecorder, save_series, unpermute  # noqa: E402

apply_fft_fp32_patch()


@dataclass
class RunOptions:
    device: str = "cuda"
    max_epochs: int | None = None
    max_train_steps: int | None = None
    max_eval_steps: int | None = None
    num_workers: int | None = None
    save_windows: bool = True
    overwrite: bool = False


# --------------------------------------------------------------------------- #
# upstream plumbing
# --------------------------------------------------------------------------- #
def ensure_tslib(cfg: GridConfig) -> Path:
    root = cfg.path("tslib_root")
    if not (root / "models").is_dir():
        raise FileNotFoundError(f"Time-Series-Library not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def build_configs(cfg: GridConfig, cell: Cell, run: RunOptions) -> Namespace:
    ds = cfg.dataset(cell.dataset)
    hp = dict(cfg.backbone(cell.backbone).get("hparams") or {})
    proto = cfg.protocol
    return Namespace(
        task_name=proto["task_name"],
        is_training=1,
        model_id=cell.cell_id,
        model=cell.backbone,
        data=ds["tslib_data"],
        root_path=str((REPO_ROOT / ds["root_path"]).resolve()) + os.sep,
        data_path=ds["data_path"],
        features=proto["features"],
        target="OT",
        freq=ds["freq"],
        checkpoints=str(cfg.path("runs_dir") / cell.cell_id) + os.sep,
        seq_len=cfg.seq_len(cell.dataset),
        label_len=cfg.label_len(cell.dataset),
        pred_len=int(cell.pred_len),
        seasonal_patterns="Monthly",
        inverse=bool(proto.get("inverse", False)),
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2, d_conv=4, tv_dt=0, tv_B=0, tv_C=0, use_D=0,
        top_k=int(hp.get("top_k", 5)), num_kernels=6,
        enc_in=int(ds["enc_in"]), dec_in=int(ds["enc_in"]), c_out=int(ds["enc_in"]),
        d_model=int(hp.get("d_model", 512)), n_heads=int(hp.get("n_heads", 8)),
        e_layers=int(hp.get("e_layers", 2)), d_layers=int(hp.get("d_layers", 1)),
        d_ff=int(hp.get("d_ff", 2048)), moving_avg=int(hp.get("moving_avg", 25)),
        factor=int(hp.get("factor", 3)), distil=True,
        dropout=float(hp.get("dropout", 0.1)), embed="timeF", activation="gelu",
        channel_independence=int(hp.get("channel_independence", 1)),
        decomp_method="moving_avg", use_norm=int(hp.get("use_norm", 1)),
        down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
        seg_len=96, patch_len=int(hp.get("patch_len", 16)),
        node_dim=10, gcn_depth=2, gcn_dropout=0.3, propalpha=0.3,
        conv_channel=32, skip_channel=32, individual=False,
        alpha=0.1, top_p=0.5, pos=1, num_class=0,
        num_workers=int(run.num_workers if run.num_workers is not None else proto["num_workers"]),
        itr=1,
        train_epochs=int(run.max_epochs or proto["train_epochs"]),
        batch_size=int(ds["batch_size"]),
        patience=int(proto["patience"]),
        learning_rate=float(ds.get("learning_rate", proto["learning_rate"])),
        des=cfg.meta["project"],
        loss=proto["loss"],
        lradj=proto["lradj"],
        use_amp=False,
        use_gpu=run.device == "cuda",
        gpu=0, gpu_type="cuda", use_multi_gpu=False, devices="0",
        p_hidden_dims=[128, 128], p_hidden_layers=2,
        use_dtw=False, augmentation_ratio=0, seed=int(cell.seed),
    )


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(cfg: GridConfig, cell: Cell, configs: Namespace):
    ensure_tslib(cfg)
    mod = importlib.import_module(f"models.{cell.backbone}")
    backbone = mod.Model(configs).float()
    params = dict(cfg.plugin(cell.plugin).get("params") or {})
    return wrap(
        backbone,
        cfg.plugin_impl(cell.plugin),
        n_channels=configs.enc_in,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        **params,
    )


class IndexedDataset:
    """Wrap a TSLib dataset so each item carries its window index.

    Without this the loader permutation is unobservable from inside the training
    process, which is precisely why the defect is hard to see: the recorded
    error vector looks exactly like a correct one.
    """

    def __init__(self, base) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        return (*item, i)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run_cell(cfg: GridConfig, cell: Cell, run: RunOptions) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    ensure_tslib(cfg)
    from data_provider.data_factory import data_provider  # type: ignore
    from utils.tools import adjust_learning_rate  # type: ignore

    configs = build_configs(cfg, cell, run)
    set_seed(cell.seed)
    device = torch.device(
        "cuda:0" if (run.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    run_dir = cfg.path("runs_dir") / cell.cell_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "configs.json", vars(configs))
    epoch_log = run_dir / "epochs.jsonl"
    if epoch_log.exists():
        epoch_log.unlink()

    train_set, train_loader = data_provider(configs, "train")
    val_set, _ = data_provider(configs, "val")
    test_set, _ = data_provider(configs, "test")

    # The independent variable. `shuffled` reproduces upstream data_factory
    # exactly (shuffle=True on the val split); `deterministic` is the repair.
    shuffled = cell.val_order == "shuffled"
    val_loader = torch.utils.data.DataLoader(
        IndexedDataset(val_set),
        batch_size=configs.batch_size,
        shuffle=shuffled,
        num_workers=configs.num_workers,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        IndexedDataset(test_set),
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.num_workers,
        drop_last=False,
    )

    model = build_model(cfg, cell, configs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = make_criterion(
        cfg.plugin_impl(cell.plugin),
        default=nn.MSELoss(),
        **dict(cfg.plugin(cell.plugin).get("params") or {}),
    )

    best = {"val_mse": float("inf"), "epoch": -1}
    best_arrays: dict[str, np.ndarray] = {}
    bad_epochs = 0
    t0 = time.time()
    epoch = -1
    for epoch in range(configs.train_epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        ep_t = time.time()
        for i, batch in enumerate(train_loader):
            if run.max_train_steps is not None and i >= run.max_train_steps:
                break
            optim.zero_grad(set_to_none=True)
            loss = _forward_loss(model, criterion, configs, batch[:4], device)
            loss.backward()
            optim.step()
            loss_sum += float(loss.detach())
            n_batches += 1

        val = _evaluate(model, configs, val_loader, device, run.max_eval_steps)
        test = _evaluate(model, configs, test_loader, device, run.max_eval_steps)
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_mse": val["mse"],
            "test_mse": test["mse"],
            "epoch_seconds": round(time.time() - ep_t, 2),
        }
        with open(epoch_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"[{cell.cell_id}] ep{epoch} train={rec['train_loss']:.5f} "
            f"val={val['mse']:.5f} test={test['mse']:.5f} ({rec['epoch_seconds']}s)",
            flush=True,
        )
        if val["mse"] < best["val_mse"] - 1e-8:
            best = {"val_mse": val["mse"], "epoch": epoch, "test_mse": test["mse"]}
            bad_epochs = 0
            best_arrays = {
                "val_loader": val["errors"],
                "val_perm": val["indices"],
                "test_index": test["errors_index_order"],
            }
        else:
            bad_epochs += 1
            if bad_epochs >= configs.patience:
                print(f"[{cell.cell_id}] early stop at epoch {epoch}", flush=True)
                break
        adjust_learning_rate(optim, epoch + 1, configs)

    truncated = run.max_eval_steps is not None or run.max_train_steps is not None
    written: dict[str, Any] = {}
    if run.save_windows and best_arrays and not truncated:
        win_dir = cfg.path("winerr_dir") / cell.cell_id
        rec_val = best_arrays["val_loader"]
        perm = best_arrays["val_perm"]
        val_index = unpermute(rec_val, perm)
        # What a defective pipeline would have persisted: the loader-order vector
        # with no record of the permutation.
        save_series(win_dir, "val_loader_order", rec_val, ORDER_LOADER,
                    cell_id=cell.cell_id, val_order=cell.val_order, split="val")
        # Ground truth, recoverable only because the indices were captured.
        save_series(win_dir, "val_index_order", val_index, ORDER_INDEX,
                    cell_id=cell.cell_id, val_order=cell.val_order, split="val")
        save_series(win_dir, "val_perm", perm.astype(np.float64), ORDER_LOADER,
                    cell_id=cell.cell_id, val_order=cell.val_order, split="val",
                    note="dataset index of the window at each loader position")
        save_series(win_dir, "test_index_order", best_arrays["test_index"], ORDER_INDEX,
                    cell_id=cell.cell_id, val_order=cell.val_order, split="test")
        written = {"winerr_dir": str(win_dir)}
    result = {
        **cell.to_dict(),
        "cell_id": cell.cell_id,
        "mse": best.get("test_mse", float("nan")),
        "val_mse": best["val_mse"],
        "best_epoch": best["epoch"],
        "epochs_run": epoch + 1,
        "n_val_windows": int(best_arrays.get("val_loader", np.empty(0)).size),
        "n_test_windows": int(best_arrays.get("test_index", np.empty(0)).size),
        "n_train_windows": len(train_set),
        "n_params": n_params,
        "loader_shuffled_val": bool(shuffled),
        "seq_len": configs.seq_len,
        "batch_size": configs.batch_size,
        "plugin_kind": kind_of(cfg.plugin_impl(cell.plugin)),
        "train_seconds": round(time.time() - t0, 1),
        "device": str(device),
        "truncated": bool(truncated),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **written,
    }
    return result


def _forward_loss(model, criterion, configs, batch, device):
    import torch

    bx, by, bxm, bym = (t.float().to(device) for t in batch)
    dec_inp = torch.cat(
        [by[:, : configs.label_len, :], torch.zeros_like(by[:, -configs.pred_len :, :])], dim=1
    ).float()
    f_dim = -1 if configs.features == "MS" else 0
    out = model(bx, bxm, dec_inp, bym)[:, -configs.pred_len :, f_dim:]
    loss = criterion(out, by[:, -configs.pred_len :, f_dim:])
    aux = model.pop_aux_loss() if hasattr(model, "pop_aux_loss") else 0.0
    return loss + aux if torch.is_tensor(aux) else loss


def _evaluate(model, configs, loader, device, max_steps: int | None) -> dict[str, Any]:
    import torch

    model.eval()
    m = WindowRecorder(per_window=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            bx, by, bxm, bym = (t.float().to(device) for t in batch[:4])
            idx = batch[4]
            dec_inp = torch.cat(
                [by[:, : configs.label_len, :], torch.zeros_like(by[:, -configs.pred_len :, :])],
                dim=1,
            ).float()
            f_dim = -1 if configs.features == "MS" else 0
            out = model(bx, bxm, dec_inp, bym)[:, -configs.pred_len :, f_dim:]
            m.update(out, by[:, -configs.pred_len :, f_dim:], index=idx)
    model.train()
    return {
        "mse": m.mse,
        "errors": m.errors(),
        "indices": m.indices(),
        "errors_index_order": m.errors_in_index_order(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="train one cell")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--val-order", choices=["deterministic", "shuffled"], default="deterministic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-eval-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    cfg = GridConfig(a.config)
    cell = Cell(
        dataset=a.dataset, backbone=a.backbone, plugin=a.plugin,
        pred_len=a.pred_len, seed=a.seed, val_order=a.val_order,
    )
    out = cfg.path("cells_dir") / f"{cell.cell_id}.json"
    if out.exists() and not a.overwrite:
        print(f"[skip] {cell.cell_id} already done")
        return 0
    run = RunOptions(
        device=a.device, max_epochs=a.max_epochs, max_train_steps=a.max_train_steps,
        max_eval_steps=a.max_eval_steps, num_workers=a.num_workers, overwrite=a.overwrite,
    )
    res = run_cell(cfg, cell, run)
    atomic_write_json(out, res)
    print(f"[done] {cell.cell_id} test_mse={res['mse']:.6f} ({res['train_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
