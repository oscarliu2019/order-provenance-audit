"""Grid launcher: expands a stage into cells and runs them as subprocesses.

Each cell is a separate process so that one crash (OOM, upstream assertion)
cannot take the sweep down, and so that global RNG state can never leak between
cells — which matters here because RNG consumption is one of the quantities
under study.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GridConfig  # noqa: E402


def build_cmd(cell, args) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.train",
        "--dataset", cell.dataset,
        "--backbone", cell.backbone,
        "--plugin", cell.plugin,
        "--pred-len", str(cell.pred_len),
        "--seed", str(cell.seed),
        "--val-order", cell.val_order,
        "--device", args.device,
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.max_epochs:
        cmd += ["--max-epochs", str(args.max_epochs)]
    if args.max_train_steps:
        cmd += ["--max-train-steps", str(args.max_train_steps)]
    if args.max_eval_steps:
        cmd += ["--max-eval-steps", str(args.max_eval_steps)]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["e1", "e2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-eval-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir", default="runs/_launcher")
    a = ap.parse_args(argv)

    cfg = GridConfig(a.config)
    cells = cfg.stage_cells(a.stage)
    done_dir = cfg.path("cells_dir")
    pending = [c for c in cells if not (done_dir / f"{c.cell_id}.json").exists()]
    print(f"stage {a.stage}: {len(cells)} cells, {len(pending)} pending, jobs={a.jobs}")
    if a.dry_run:
        for c in pending[:20]:
            print("  ", c.cell_id)
        if len(pending) > 20:
            print(f"   ... and {len(pending) - 20} more")
        return 0

    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    q: Queue = Queue()
    for c in pending:
        q.put(c)
    failures: list[str] = []
    lock = threading.Lock()
    t_start = time.time()
    counter = {"n": 0}

    def worker(wid: int) -> None:
        while True:
            try:
                cell = q.get_nowait()
            except Exception:
                return
            log = log_dir / f"{cell.cell_id}.log"
            with open(log, "w", encoding="utf-8") as fh:
                p = subprocess.run(build_cmd(cell, a), stdout=fh, stderr=subprocess.STDOUT)
            with lock:
                counter["n"] += 1
                el = time.time() - t_start
                status = "ok" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(
                    f"[{counter['n']}/{len(pending)}] w{wid} {cell.cell_id} {status} "
                    f"elapsed={el / 60:.1f}m",
                    flush=True,
                )
                if p.returncode != 0:
                    failures.append(cell.cell_id)
            q.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"finished in {(time.time() - t_start) / 60:.1f} min; {len(failures)} failures")
    if failures:
        (log_dir / f"failures_{a.stage}.json").write_text(json.dumps(failures, indent=2))
        for f in failures[:20]:
            print("  FAIL", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
