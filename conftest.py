import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "third_party" / "tslib")):
    if p not in sys.path:
        sys.path.insert(0, p)
