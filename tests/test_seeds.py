"""Seeds must be reproducible across processes, or no artefact is reproducible.

The regression these tests guard is subtle and was present in this repository:
seeding a generator with Python's builtin ``hash()`` of a string key makes the
draw depend on the per-process hash salt, so re-running the analysis silently
produces different permutations -- and therefore different numbers -- than the
ones printed in the paper.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.seeds import stable_rng, stable_seed

ROOT = Path(__file__).resolve().parent.parent


def test_stable_seed_is_deterministic_within_process():
    assert stable_seed("t1", "cell-7") == stable_seed("t1", "cell-7")
    assert stable_seed("t1", "cell-7") != stable_seed("t1", "cell-8")
    assert stable_seed("t1", "cell-7") != stable_seed("t2", "cell-7")


def test_stable_seed_separates_keys_that_concatenate_alike():
    assert stable_seed("ab", "c") != stable_seed("a", "bc")


def test_stable_seed_survives_a_fresh_interpreter_with_a_different_hash_salt():
    """Two subprocesses with different PYTHONHASHSEED must agree."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from src.seeds import stable_seed;"
        "print(stable_seed('t1', 'ETTh1|iTransformer|none|96|2021'))" % str(ROOT)
    )
    outs = []
    for salt in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"}
        outs.append(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
            ).stdout.strip()
        )
    assert len(set(outs)) == 1, f"seed depends on the interpreter hash salt: {outs}"


def test_builtin_hash_is_not_stable_so_the_guard_above_is_meaningful():
    code = "print(hash(('t1', 'cell-7')))"
    outs = []
    for salt in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"}
        outs.append(
            subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
            ).stdout.strip()
        )
    assert len(set(outs)) > 1


def test_no_artefact_seed_uses_builtin_hash():
    """Guard the whole analysis layer against a relapse."""
    offenders = []
    for path in sorted((ROOT / "tools").glob("*.py")) + sorted((ROOT / "src").glob("*.py")):
        if path.name == "seeds.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "hash(" in stripped and "stable" not in stripped and "blake2b" not in stripped:
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {stripped}")
    assert not offenders, "builtin hash() used for seeding:\n" + "\n".join(offenders)


def test_stable_rng_reproduces_the_same_permutation():
    a = stable_rng("t1-control", "cell-1").permutation(50)
    b = stable_rng("t1-control", "cell-1").permutation(50)
    assert np.array_equal(a, b)
