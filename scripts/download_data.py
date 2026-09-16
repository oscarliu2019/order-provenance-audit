"""Fetch the benchmark CSVs and verify them against the checksums we used.

The datasets are public and we do not redistribute them. This script downloads
the copies hosted with the Time-Series-Library dataset mirror on the Hugging Face
Hub and refuses to proceed if a file's SHA-256 differs from the one this study
was run against, so a silent upstream change cannot be mistaken for a result.

    python scripts/download_data.py            # download and verify
    python scripts/download_data.py --verify   # verify what is already there
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

# relative path -> (url path, sha256 of the file this study used)
FILES: dict[str, tuple[str, str]] = {
    "data/ETT-small/ETTh1.csv": (
        "ETT-small/ETTh1.csv",
        "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
    ),
    "data/ETT-small/ETTh2.csv": (
        "ETT-small/ETTh2.csv",
        "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
    ),
    "data/ETT-small/ETTm1.csv": (
        "ETT-small/ETTm1.csv",
        "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
    ),
    "data/ETT-small/ETTm2.csv": (
        "ETT-small/ETTm2.csv",
        "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
    ),
    "data/weather/weather.csv": (
        "weather/weather.csv",
        "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
    ),
    "data/exchange_rate/exchange_rate.csv": (
        "exchange_rate/exchange_rate.csv",
        "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
    ),
    "data/illness/national_illness.csv": (
        "illness/national_illness.csv",
        "93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="do not download, only verify")
    a = ap.parse_args(argv)

    bad = 0
    for rel, (url_path, want) in FILES.items():
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            if a.verify:
                print(f"MISSING  {rel}")
                bad += 1
                continue
            url = f"{BASE}/{url_path}"
            print(f"download {rel} <- {url}")
            urllib.request.urlretrieve(url, dest)
        got = sha256(dest)
        if got == want:
            print(f"ok       {rel}")
        else:
            print(f"MISMATCH {rel}\n  expected {want}\n  got      {got}")
            bad += 1
    if bad:
        print(f"\n{bad} file(s) missing or changed upstream; results are not comparable.")
        return 1
    print(f"\nall {len(FILES)} files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
