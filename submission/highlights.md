# Highlights (Elsevier: 3-5 bullets, each <= 85 characters including spaces)

- Shuffled evaluation loaders silently permute per-sample error artefacts
- Every aggregate check passes: a mean cannot see a permutation
- The shuffled-feature negative control is equal in law to the defect
- For labels derived from several arms the control is optimistic, not safe
- Two O(n) order tests, plus a sidecar contract that fails closed
