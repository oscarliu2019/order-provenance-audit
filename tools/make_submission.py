#!/usr/bin/env python3
"""Build the submission packages: the LaTeX source archive and a reviewer bundle.

    python tools/make_submission.py

Writes into submission/:
  OrderProvenance_EM_latex_<date>.zip   LaTeX sources Elsevier's system compiles
  OrderProvenance_submission_<date>.zip the same plus the compiled PDF, the
                                        highlights, the cover letter and the
                                        declarations, for a single upload
It refuses to run when the number checker fails, so a package can never contain
a manuscript whose numbers disagree with the artefacts.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "submission"

LATEX_FILES = [
    "main.tex",
    "refs.bib",
    "numbers.tex",
    "cas-sc.cls",
    "cas-common.sty",
    "cas-model2-names.bst",
]
LATEX_GLOBS = ["tables/*.tex", "figs/*.pdf"]


def _check() -> None:
    r = subprocess.run([sys.executable, "tools/verify_paper_numbers.py"], cwd=ROOT,
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    print("verify_paper_numbers:", *tail)
    if r.returncode != 0:
        raise SystemExit("refusing to package: the number checker failed")
    if not (PAPER / "main.pdf").exists():
        raise SystemExit("refusing to package: paper/main.pdf is missing")


def _latex_members() -> list[tuple[Path, str]]:
    members = []
    for name in LATEX_FILES:
        p = PAPER / name
        if not p.exists():
            raise SystemExit(f"missing LaTeX source: {p}")
        members.append((p, name))
    for pattern in LATEX_GLOBS:
        found = sorted(PAPER.glob(pattern))
        if not found:
            raise SystemExit(f"nothing matched {pattern}")
        members += [(p, str(p.relative_to(PAPER))) for p in found]
    return members


def main() -> int:
    _check()
    OUT.mkdir(exist_ok=True)
    stamp = _dt.date.today().strftime("%Y%m%d")

    # EM rejects subdirectories, so the LaTeX archive is built flat and compiled
    # once before packaging; see tools/make_em_latex_flat.py.
    r = subprocess.run([sys.executable, "tools/make_em_latex_flat.py"], cwd=ROOT,
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit("refusing to package: the flat LaTeX build failed")
    latex_zip = OUT / f"OrderProvenance_EM_latex_{stamp}.zip"

    full_zip = OUT / f"OrderProvenance_submission_{stamp}.zip"
    with zipfile.ZipFile(full_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in _latex_members():
            z.write(src, f"latex/{arc}")
        z.write(PAPER / "main.pdf", "manuscript.pdf")
        for name in ("highlights.md", "highlights.docx", "cover_letter.md",
                     "declarations.md", "declaration_of_interest_elsevier.docx",
                     "credit_author_statement.md", "data_availability.md"):
            p = OUT / name
            if p.exists():
                z.write(p, name)
    print(f"{full_zip.relative_to(ROOT)}  ({full_zip.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
