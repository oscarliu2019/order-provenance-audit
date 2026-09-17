"""Content-derived seeds that are stable across processes and machines.

Python's builtin ``hash()`` is salted per interpreter for ``str`` (and for any
tuple containing one) unless ``PYTHONHASHSEED`` is pinned, so seeding a
generator with ``hash(key)`` silently makes a study irreproducible: the same
command emits different permutations on every invocation. Every pseudo-random
draw in this repository is therefore seeded through :func:`stable_seed`, which
is a pure function of the key contents.
"""

from __future__ import annotations

from hashlib import blake2b

import numpy as np

_DIGEST_BYTES = 8


def stable_seed(*parts: object) -> int:
    """Return a deterministic 64-bit seed derived from ``parts``.

    The parts are rendered with ``repr`` and joined with a separator that
    cannot appear in a rendered scalar, so distinct keys stay distinct. The
    result depends only on the key, never on interpreter state.
    """
    key = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(blake2b(key, digest_size=_DIGEST_BYTES).digest(), "big")


def stable_rng(*parts: object) -> np.random.Generator:
    """A ``numpy`` generator seeded by :func:`stable_seed`."""
    return np.random.default_rng(stable_seed(*parts))
