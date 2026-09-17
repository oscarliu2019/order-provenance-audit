# Cover letter — Journal of Systems and Software

Dear Editors,

We submit for your consideration the manuscript **“Silent order corruption in
per-sample evaluation artefacts: why the standard negative control cannot detect
it, and two tests that can.”**

Machine-learning papers increasingly report per-sample results — per-instance
difficulty analyses, sample-level ablations, per-instance routing — and every such
analysis joins two artefacts written by different parts of a pipeline. In practice
the join is positional: entry *i* of the error vector is assumed to describe row
*i* of the feature table. We study what happens when that assumption silently
fails, using a widely adopted forecasting library as the carrier: its data
provider disables shuffling for the test split and leaves it enabled for
validation, which is correct for a loader consumed through a mean and becomes a
hazard the moment a per-sample artefact is stored from that split.

We believe the manuscript fits JSS for three reasons.

**It characterises a fault class, not a single bug.** We prove that the fault is
invariant to every aggregate the pipeline checks, that the shape assertion which
array-oriented code habitually writes is blind to it by construction whenever the
last partial batch is kept, and that it attenuates every association towards the
null — so it cannot fabricate a positive finding, only a negative one.

**Its central result is about the limits of a standard validation device.** We
prove that the corrupted joined table is *equal in distribution* to the table
produced by the shuffled-feature negative control that the community uses to
validate exactly this kind of analysis. A per-sample null result accompanied by a
well-behaved negative control is therefore exactly as consistent with a broken
join as with a true null; the control cannot fail. We are not aware of a prior
statement of this degeneracy, and we verify it with equivalence tests rather than
with a failure to reject. We also isolate the single case in which the defect and
the control do differ — analyses whose label is derived from several independently
written vectors, per-sample arm selection being the common instance — and show
that the difference runs the wrong way: the control preserves the label law and
therefore reports a *milder* failure than the fault produces, which we confirm
both theoretically and on the label distributions of our grid.

**It delivers tooling with measured error rates.** We give two post-hoc tests that
read only the stored vectors — one exploiting the serial dependence that
overlapping windows induce, with exact randomisation moments and hence an O(n)
p-value, and one exploiting cross-arm agreement, which needs no serial structure —
together with a minimal write-side provenance contract that makes the positional
join fail closed. On 356 single-GPU training runs (21.3 GPU hours) across seven
public benchmarks, three backbones and four arms — 712 audited error vectors, 4.27 M
summed window-level evaluations over 56 distinct sample sets — the battery separates
intact from permuted artefacts by a large measured margin (smallest intact z 5.7
against largest permuted z 3.35) and detects corruptions far milder than a full
permutation. We also calibrate the normal tail of the test against 2.5 M Monte Carlo
draws and report where it is anti-conservative, rather than quoting a nominal
false-alarm rate we have not measured. We also quantify a maintenance
trap: the obvious one-line repair changes the random stream and therefore the
reported test error, so a maintainer must re-run rather than patch.

The manuscript is fully reproducible. It contains no typed number: every
quantitative claim expands a macro generated from the released artefacts, and a
verification script re-derives each one independently and fails on any mismatch.
The per-sample vectors themselves are released, so a reader can run the audit
without a GPU. The artefacts, the reference implementation, the test suite and the
manuscript sources are at
<https://github.com/oscarliu2019/order-provenance-audit>.

The archived release is at <https://doi.org/10.5281/zenodo.22809148>.

**Disclosure of related work by the same authors.** A companion manuscript, *“Per-window
oracle headroom is not learnable: a feedback-delay audit of plug-in selection for
time series forecasting,”* is under review elsewhere. It shares this experimental
infrastructure but addresses a disjoint question — whether per-window plug-in
selection is learnable under a legal feedback delay — and reports disjoint
results. Neither manuscript’s claims depend on the other’s. We cite it in the
present manuscript and state the relationship in a dedicated declaration so that
the overlap can be assessed directly. We are happy to supply the companion
manuscript to the editors on request.

The work is original, has not been published elsewhere, and is not under
consideration by another journal. All authors have approved the submission and
declare no competing interests. Our use of a large language model was limited to
copy-editing, LaTeX formatting and the drafting of analysis and plotting scripts,
and is disclosed in the manuscript; all experimental design, statistical decisions
and conclusions are the authors’ own.

Thank you for your consideration.

Sincerely,

Mianhan Liu (corresponding author) and Chen Chen
Shanghai, China
