#!/usr/bin/env python3
"""Build the flat LaTeX package that Editorial Manager can actually compile.

EM's PDF builder does not accept subdirectories: every .tex/.cls/.sty/.bst/.bbl
and every graphic has to sit in one flat level, so a zip that keeps `tables/`
and `figs/` builds a PDF with missing figures and `??` cross-references. This
script derives a flat copy of paper/ instead of hand-maintaining one, rewrites
the `\\input{}` and `\\includegraphics{}` paths, compiles it, and refuses to
package unless the flat build has the same page count as paper/main.pdf and no
undefined references.

    python tools/make_em_latex_flat.py            # flatten, compile, zip
    python tools/make_em_latex_flat.py --no-build  # flatten and zip only
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "submission" / "em_latex"
STYLE_FILES = ["cas-sc.cls", "cas-common.sty", "cas-model2-names.bst", "refs.bib"]
TOP_LEVEL_TEX = ["numbers.tex"]
SUBDIRS = ["tables", "figs"]


def flatten() -> list[Path]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    files: list[Path] = []

    for name in STYLE_FILES + TOP_LEVEL_TEX:
        src = PAPER / name
        if not src.exists():
            sys.exit(f"[fail] missing source: {src}")
        shutil.copy2(src, OUT / name)
        files.append(OUT / name)

    bbl = PAPER / "main.bbl"
    if not bbl.exists():
        sys.exit("[fail] paper/main.bbl is missing: run a full build in paper/ first")
    shutil.copy2(bbl, OUT / "main.bbl")
    files.append(OUT / "main.bbl")

    for sub in SUBDIRS:
        for src in sorted((PAPER / sub).iterdir()):
            if not src.is_file():
                continue
            dst = OUT / src.name
            if dst.exists():
                sys.exit(f"[fail] name collision after flattening: {src.name}")
            shutil.copy2(src, dst)
            files.append(dst)

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    tex, n_in = re.subn(r"\\input\{tables/", r"\\input{", tex)
    tex, n_fig = re.subn(r"(\\includegraphics(?:\[[^\]]*\])?\{)figs/", r"\1", tex)
    if n_in == 0 or n_fig == 0:
        sys.exit(f"[fail] path rewrite matched nothing (input={n_in}, fig={n_fig})")
    code = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    leftover = sorted(set(re.findall(r"\{(?:tables|figs)/", code)))
    if leftover:
        sys.exit(f"[fail] subdirectory paths remain in main.tex: {leftover}")
    (OUT / "main.tex").write_text(tex, encoding="utf-8")
    files.append(OUT / "main.tex")

    for f in files:
        if f.name.count(".") != 1 or re.search(r"[^A-Za-z0-9._-]", f.name):
            sys.exit(f"[fail] EM will not accept this file name: {f.name}")
    print(f"[ok  ] flattened {len(files)} files; rewrote {n_in} inputs, {n_fig} figures")
    return files


def build() -> int:
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ):
        p = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, timeout=900)
        if p.returncode != 0 and cmd[0] == "pdflatex":
            sys.exit("[fail] " + " ".join(cmd) + ":\n" + "\n".join(p.stdout.splitlines()[-30:]))

    log = (OUT / "main.log").read_text(encoding="utf-8", errors="ignore")
    undefined = [ln for ln in log.splitlines()
                 if "LaTeX Warning: Citation" in ln or "LaTeX Warning: Reference" in ln]
    if undefined:
        sys.exit("[fail] undefined references in the flat build:\n  " + "\n  ".join(undefined[:10]))
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    ref = re.search(r"Output written on main\.pdf \((\d+) pages",
                    (PAPER / "main.log").read_text(encoding="utf-8", errors="ignore"))
    if not pages:
        sys.exit("[fail] could not read the page count of the flat build")
    if ref and pages.group(1) != ref.group(1):
        sys.exit(f"[fail] flat build has {pages.group(1)} pages, paper/ has {ref.group(1)}")
    print(f"[ok  ] flat build compiles: {pages.group(1)} pages, 0 undefined references")
    return int(pages.group(1))


def pack(files: list[Path]) -> Path:
    stamp = dt.date.today().strftime("%Y%m%d")
    zip_path = ROOT / "submission" / f"OrderProvenance_EM_latex_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, arcname=f.name)
    print(f"[done] {zip_path.relative_to(ROOT)}"
          f"  ({len(files)} files, no subdirectories, {zip_path.stat().st_size/1024:.0f} KB)")
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    files = flatten()
    if not a.no_build:
        build()
    pack(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
