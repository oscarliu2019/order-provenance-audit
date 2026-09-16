# Data availability statement

All datasets used in this study are public standard benchmarks (ETT-small,
Weather, Exchange-Rate, ILI) and are not redistributed. The release includes the
download script and the SHA-256 checksums of the exact files this study used.

Everything required to recompute every number, table and figure in the manuscript
is publicly released at

<https://github.com/oscarliu2019/order-provenance-audit>

and comprises: the experiment configuration; the per-run table; the per-sample
audit outputs (AOT sequence statistics, calibration, sensitivity, the
certification sweep, cross-arm agreement, the battery); the per-sample feature
tables; the T1 and T2 downstream outputs together with the equivalence tests and
the derived-label distribution table behind Proposition 7; the
paired repair-cost table; the reference implementation of the order-provenance
contract and of both tests; a test suite that states the propositions as
executable assertions; and the generators for `paper/numbers.tex`, every table and
every figure.

`tools/verify_paper_numbers.py` re-derives every quantitative claim in the
manuscript independently from those artefacts and exits non-zero on any mismatch,
naming the claim identifier, the printed value and the recomputed value.
