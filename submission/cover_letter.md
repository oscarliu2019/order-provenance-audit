# Cover letter — Journal of Systems and Software

Dear Editors,

We submit for your consideration the manuscript **“Silent order corruption in
per-sample evaluation artefacts: control degeneration, diagnostic blind spots,
and a provenance contract.”**

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

**It characterises a fault class, not a single bug.** We prove that the fault
leaves every order-symmetric aggregate the pipeline checks numerically unchanged,
that the shape assertion which array-oriented code habitually writes is blind to
it by construction whenever the last partial batch is kept, and that under an
exchangeability assumption it attenuates association towards the null — so it
does not fabricate a positive finding, only a spurious negative one.

**Its central result is about the limits of a standard validation device.** We
prove that the corrupted joined table is *equal in distribution* to the table
produced by the shuffled-feature negative control that the community uses to
validate exactly this kind of analysis. A per-sample null result accompanied by a
well-behaved negative control is therefore exactly as consistent with a broken
join as with a true null; under this fault the control is uninformative by
construction. We are not aware of a prior statement of this degeneracy, and we
verify it with equivalence tests rather than with a failure to reject. We also
isolate the single case in which the defect and the control do differ — analyses
whose label is derived from several independently written vectors, per-sample arm
selection being the common instance — and show that the difference runs the wrong
way: the control preserves the label law and therefore reports a *milder* failure
than the fault produces, which we confirm both theoretically and on the label
distributions of our grid.

**It delivers tooling whose limits we measure rather than assert.** We give a
write-side provenance contract that binds each per-sample artefact to a digest of
its ordered sample identifiers, so that a positional join fails closed, and two
read-side diagnostics that need only the stored vectors — one exploiting the
serial dependence that overlapping windows induce, with exact randomisation
moments and hence an O(n) p-value, and one exploiting cross-arm agreement, which
needs no serial structure. The honest finding is that the two roles are not
symmetric. Against 25 injected identity faults the contract refuses 20 and we
enumerate the 5 it cannot see. The order test flags 90.63 % of arms in the real
defect, but flags 0.0 % of our injected block permutations and mild partial
permutations, so we present it as a targeted diagnostic for near-random
permutation rather than a general integrity test. The cross-arm test assumes
weakly correlated arm-wise permutations, and in the real defect it is broken by
its own assumption: 20 of 24 defective groups share order across at least one arm
pair, and the test certifies 91.67 % of genuinely corrupted groups as clean. We
therefore recommend the contract as the primary defence and the diagnostics as
narrow forensic instruments, and we state the blind spots in the abstract rather
than burying them.

The revision also strengthens the statistical treatment. Window-level statistics
are re-analysed with circular block permutation and moving-block bootstrap, and
group-level statistics with dataset and sample-set cluster bootstrap, so that no
verdict rests on an independence assumption that overlapping windows violate. We
additionally report a dynamic purge audit separating the horizon-only and
input-plus-horizon exclusion radii. We also corrected an analytic variance term
in the cross-arm statistic and quantify a maintenance trap: the obvious one-line
repair changes the random stream and therefore the reported test error, so a
maintainer must re-run rather than patch.

The manuscript is fully reproducible. It contains no typed number: every
quantitative claim expands a macro generated from the released artefacts, and a
verification script re-derives each one independently and fails on any mismatch.
The per-sample vectors, their provenance sidecars, the observed permutations, and
the controlled sharing conditions are all released, so a reader can run the audit
without a GPU. The artefacts, the reference implementation, the test suite and the
manuscript sources are at
<https://github.com/oscarliu2019/order-provenance-audit>.

The archived release is at <https://doi.org/10.5281/zenodo.22809148>.

**Disclosure of related work by the same authors.** A companion manuscript,
*“Large per-window oracle headroom that three selector families did not convert:
a feedback-delay audit of plug-in selection for time series forecasting,”* is
under review elsewhere. The two manuscripts share the forecasting grid, the
trained checkpoints, the dataset preparation, the same loader-order defect and
the same corrected cross-arm variance term; they address disjoint questions and
report disjoint results. The present manuscript contributes the control
degeneracy proposition, the two order diagnostics and their measured blind spots,
the observed-permutation audit, the provenance contract, the dependence-aware
re-analysis and the repair-cost measurement. The companion contributes selector
training, the feedback-delay model, oracle headroom and gating policy. Neither
manuscript’s claims depend on the other’s acceptance, and the shared material is
described identically in both. We cite it in the present manuscript and state the
relationship in a dedicated declaration so that the overlap can be assessed
directly. We are happy to supply the companion manuscript to the editors on
request.

**Relationship to our earlier JSS submission.** An earlier version of this work
was submitted to JSS as JSSOFTWARE-D-26-02395, “Silent order corruption in
per-sample evaluation artefacts: why the standard negative control cannot detect
it, and two tests that can.” On 18 September 2026 we asked the editorial office
to withdraw it, because reviewing that version would have been a poor use of
referee time: it framed the two diagnostics as detectors of the fault, whereas
the evidence in this version shows that the write-side provenance contract is the
primary defence and that both diagnostics have blind spots we can now measure and
enumerate. The present submission is that substantially revised manuscript,
with a new title, a rewritten abstract, the dependence-aware re-analysis, the
corrected cross-arm variance term and the measured contract coverage. It is
intended to replace JSSOFTWARE-D-26-02395 rather than to accompany it; if the
withdrawal has not yet been processed when this reaches you, please treat this
submission as its replacement.

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
