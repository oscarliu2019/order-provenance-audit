# Silent order corruption in per-sample evaluation artefacts

Reproduction package for the manuscript

> **Silent order corruption in per-sample evaluation artefacts: why the standard
> negative control cannot detect it, and two tests that can**
> Mianhan Liu, Chen Chen. Shanghai, China.

[`paper/main.pdf`](paper/main.pdf) is the manuscript. Everything it prints is
generated from the artefacts in this repository, and
[`tools/verify_paper_numbers.py`](tools/verify_paper_numbers.py) re-derives every
number independently and fails on any mismatch.

---

## The one-paragraph version

A per-sample analysis joins two artefacts: a per-sample error vector written by
the evaluation loop, and a per-sample feature table written by something else. The
join is positional. If the evaluation loader shuffles — which is the default for
the **validation** split in the Time-Series-Library family of forecasting
codebases, while the test split is protected — the stored vector is in *loader
order* and the feature table is in *index order*, so entry `i` of one describes a
different window than row `i` of the other.

Nothing catches it:

* the validation mean, early stopping and every aggregate table are **exactly**
  invariant, because a mean is a function of a multiset;
* the shape check passes, because `drop_last=False` keeps the length equal to `n`;
* and the canonical **shuffled-feature negative control is equal in distribution
  to the defect**, so the control that is supposed to protect the analysis
  provably cannot fail (Proposition 5 in the paper);
* and when the analysis derives its label from several independently written
  vectors — per-sample arm selection is the common case — the control is worse than
  powerless: it keeps the label law intact and therefore reports a *milder* failure
  than the defect causes (Proposition 7).

What survives is structure the aggregate throws away. Consecutive sliding windows
overlap, so per-window errors are serially dependent (median lag-1 correlation
`0.9975` over 712 real sequences and 4.27 M windows) and a permutation destroys it
— that is **AOT**. Several arms
evaluated on the same windows agree on which windows are hard — that is **CAT**.
Both read only the stored vectors: no model, no data, no rerun.

The write-side fix is a provenance sidecar plus a reader that refuses to load an
array for a positional join unless the sidecar says `index` order.

---

## Repository layout

```
src/            the library
  winerr.py       order-provenance contract: sidecars, WindowRecorder, load_series
  detect.py       AOT (exact Wald-Wolfowitz moments) and CAT (Monte-Carlo null)
  downstream.py   T1 difficulty association, T2 per-window arm selection, TOST
  features.py     per-window covariates, computed from the look-back only
  train.py        one training cell; records per-window MSE under the contract
  config.py       the grid, cell identity (val_order is part of the identity)
  plugins/        the four arms: none, RevIN, FreDF, SAN-lite
scripts/
  download_data.py  fetch the public CSVs and verify SHA-256
  run_grid.py       expand a stage into cells and run them as subprocesses
tools/
  make_artifacts.py       artefacts -> artifacts/*.csv
  paper_assets.py         artifacts -> paper/numbers.tex, tables, figures
  verify_paper_numbers.py independent re-derivation of every claim
tests/          57 tests; the propositions as executable statements
artifacts/      every CSV/JSON the paper cites (incl. label_law.csv, the
                mechanism behind Proposition 7)
results/
  winerr/         per-window error vectors + provenance sidecars (auditable as-is)
  cells/, runs/   per-run metadata
paper/          main.tex, refs.bib, generated numbers/tables/figures, main.pdf
third_party/tslib  vendored Time-Series-Library subset (upstream licence applies)
```

## Reproduce without a GPU

The interesting part of the paper — the detectors, their calibration, the
downstream collapse, the equivalence tests — needs only the stored vectors, which
are in this repository.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

pytest -q                                  # 57 tests, ~15 s
python tools/make_artifacts.py aot          # AOT over every stored vector
python tools/make_artifacts.py cat          # cross-arm agreement
python tools/verify_paper_numbers.py        # re-derive every number in the paper
```

`verify_paper_numbers.py` prints `N/N claims re-derived from artefacts` and exits
non-zero on the first disagreement, naming the macro, the printed value and the
recomputed value.

To rebuild the manuscript:

```bash
python tools/paper_assets.py all            # numbers.tex, tables/, figs/
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Reproduce the runs (GPU)

```bash
python scripts/download_data.py             # ~32 MB, verified by SHA-256
python scripts/run_grid.py --stage e1 --jobs 4    # detector bank
python scripts/run_grid.py --stage e2 --jobs 4    # paired deterministic/shuffled
python tools/make_artifacts.py all
```

One NVIDIA V100 (32 GB) is enough: the full grid in the paper is 356 runs and
21.3 GPU hours, reported in Table 1. Cells are independent subprocesses and completed cells are skipped, so the
sweep is restartable.

## Using the contract in your own code

```python
from src.winerr import WindowRecorder, save_series, load_series

rec = WindowRecorder()                       # streaming: one float per window
for batch_x, batch_y, index in loader:       # index = dataset indices of the batch
    rec.update(model(batch_x), batch_y, index=index)

save_series(out_dir, "val_index_order", rec.errors_in_index_order(), order="index",
            dataset="ETTh1", split="val", seq_len=96, pred_len=96)
save_series(out_dir, "val_loader_order", rec.errors(), order="loader")

err  = load_series(out_dir, "val_index_order")           # index order required
mean = load_series(out_dir, "val_loader_order",
                   require_order=None).mean()            # order-symmetric: explicit
```

`load_series` raises `ProvenanceError` when the sidecar is missing, when the order
is not the one the caller requires, or when the length disagrees. Auditing an
artefact you did not produce:

```python
from src.detect import aot, cat
aot(err)["p"]      # < 1e-6 for every intact overlapping-window sequence we saw
cat(err_matrix)    # several arms, same windows
```

## What the tests do and do not certify

| | AOT | CAT |
|---|---|---|
| needs serial dependence | yes | no |
| needs ≥ 2 arms | no | yes |
| catches a shared permutation | yes | **no** |
| catches independent per-arm permutations | yes | yes |
| catches circular rotation / reversal | **no** | **no** |
| cost | `O(n)`, closed form | `O(B K² n log n)` |

Neither test proves alignment. The sidecar contract does not have this limitation,
which is why the paper presents it as the primary defence and the tests as an
audit for artefacts that already exist.

## Datasets

ETTh1, ETTh2, ETTm1, ETTm2, Weather, Exchange-Rate and ILI, all public standard
benchmarks, not redistributed here. `scripts/download_data.py` fetches them and
verifies the exact SHA-256 of every file this study used.

## Related

The companion study — *Per-window oracle headroom is not learnable: a
feedback-delay audit of plug-in selection for time series forecasting* — shares
this experimental infrastructure and asks a disjoint question (is per-window
plug-in selection learnable under legal feedback delay?). Its package is at
<https://github.com/oscarliu2019/plugin-selection-audit>.

## Licence

Code in `src/`, `scripts/`, `tools/` and `tests/`: MIT (see `LICENSE`).
`third_party/tslib` is a vendored subset of the Time-Series-Library and remains
under its upstream licence. The datasets are the property of their original
publishers.

## Citation

See `CITATION.cff`.
