#!/usr/bin/env python3
"""Build the LaTeX package that Editorial Manager can actually compile.

Editorial Manager compiles the manuscript item on its own: only that one file
is placed in the working directory, and everything uploaded as "LaTeX Source
Files" is merely attached to the built PDF. A submission that keeps
`\\input{numbers}` therefore dies with

    ! LaTeX Error: File `numbers.tex' not found.

so this script derives a *single-file* manuscript from paper/main.tex: every
`\\input{}` is inlined, the BibTeX pass is replaced by the generated
`thebibliography`, and figure paths are flattened because EM stores figure
items without their directory. The class/style/bib sources are still shipped
alongside so the editorial office can rebuild the paper from what it received.
The package is only written if the derived build has the same page count as
paper/main.pdf and no undefined references.

    python tools/make_em_latex_flat.py            # derive, compile, zip
    python tools/make_em_latex_flat.py --no-build  # derive and zip only
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
# Shipped for the editorial office; EM resolves the Elsevier class itself.
SUPPORT_FILES = ["cas-sc.cls", "cas-common.sty", "cas-model2-names.bst", "refs.bib"]


def inline_inputs(tex: str) -> tuple[str, int]:
    """Replace every \\input{...} with the contents of the file it names."""
    pattern = re.compile(r"\\input\{([^}]+)\}")
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        name = m.group(1)
        src = PAPER / (name if name.endswith(".tex") else name + ".tex")
        if not src.exists():
            sys.exit(f"[fail] \\input{{{name}}} does not resolve to a file: {src}")
        count += 1
        body = src.read_text(encoding="utf-8").rstrip("\n")
        # The trailing `%` swallows the newline so an \input inside a macro
        # argument keeps expanding to exactly what the file contains, and the
        # closing brace of that argument never lands on a comment line.
        return (f"% >>> inlined from paper/{src.relative_to(PAPER)}\n"
                f"{body}%\n% <<< inlined\n")

    out = pattern.sub(repl, tex)
    if pattern.search(out):
        sys.exit("[fail] nested \\input{} survived inlining")
    return out, count


def inline_bibliography(tex: str) -> str:
    bbl = PAPER / "main.bbl"
    if not bbl.exists():
        sys.exit("[fail] paper/main.bbl is missing: run a full build in paper/ first")
    body = bbl.read_text(encoding="utf-8").strip()
    if "\\begin{thebibliography}" not in body:
        sys.exit("[fail] paper/main.bbl does not contain a thebibliography environment")
    block = ("% >>> bibliography inlined from the BibTeX run over refs.bib\n"
             "% (refs.bib is attached to this submission for the editorial office)\n"
             + body + "\n% <<< bibliography")
    tex, n = re.subn(r"\\bibliographystyle\{[^}]*\}\s*\n\\bibliography\{[^}]*\}",
                     lambda _m: block, tex)
    if n != 1:
        sys.exit(f"[fail] expected exactly one \\bibliography block, replaced {n}")
    return tex


def derive() -> list[Path]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    files: list[Path] = []

    for name in SUPPORT_FILES:
        src = PAPER / name
        if not src.exists():
            sys.exit(f"[fail] missing source: {src}")
        shutil.copy2(src, OUT / name)
        files.append(OUT / name)

    for src in sorted((PAPER / "figs").iterdir()):
        if not src.is_file():
            continue
        dst = OUT / src.name
        if dst.exists():
            sys.exit(f"[fail] name collision after flattening: {src.name}")
        shutil.copy2(src, dst)
        files.append(dst)

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    tex, n_in = inline_inputs(tex)
    tex = inline_bibliography(tex)
    tex, n_fig = re.subn(r"(\\includegraphics(?:\[[^\]]*\])?\{)figs/", r"\1", tex)
    if n_in == 0 or n_fig == 0:
        sys.exit(f"[fail] nothing was rewritten (inlined={n_in}, figures={n_fig})")
    code = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    leftover = sorted(set(re.findall(r"\{(?:tables|figs)/", code)))
    if leftover:
        sys.exit(f"[fail] subdirectory paths remain in main.tex: {leftover}")
    (OUT / "main.tex").write_text(tex, encoding="utf-8")
    files.append(OUT / "main.tex")

    for f in files:
        if f.name.count(".") != 1 or re.search(r"[^A-Za-z0-9._-]", f.name):
            sys.exit(f"[fail] EM will not accept this file name: {f.name}")
    print(f"[ok  ] single-file manuscript: inlined {n_in} \\input files and the "
          f"bibliography, flattened {n_fig} figure paths")
    return files


def build() -> int:
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ):
        p = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            sys.exit("[fail] " + " ".join(cmd) + ":\n" + "\n".join(p.stdout.splitlines()[-30:]))

    log = (OUT / "main.log").read_text(encoding="utf-8", errors="ignore")
    undefined = [ln for ln in log.splitlines()
                 if "LaTeX Warning: Citation" in ln or "LaTeX Warning: Reference" in ln]
    if undefined:
        sys.exit("[fail] undefined references in the derived build:\n  " + "\n  ".join(undefined[:10]))
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    ref = re.search(r"Output written on main\.pdf \((\d+) pages",
                    (PAPER / "main.log").read_text(encoding="utf-8", errors="ignore"))
    if not pages:
        sys.exit("[fail] could not read the page count of the derived build")
    if ref and pages.group(1) != ref.group(1):
        sys.exit(f"[fail] derived build has {pages.group(1)} pages, paper/ has {ref.group(1)}")
    print(f"[ok  ] derived build compiles standalone: {pages.group(1)} pages, "
          "0 undefined references")
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
    files = derive()
    if not a.no_build:
        build()
    pack(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
