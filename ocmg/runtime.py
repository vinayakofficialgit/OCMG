"""Process setup shared by the experiment scripts.

Two settings, both chosen because they change how long a run takes and not
what it produces.

Thread pinning. The fitted estimator is a calibrated ensemble, so scoring one
candidate walks several sub-models' trees. The admission loop scores candidates
one at a time, and at that size the thread pool costs more to coordinate than
the work it distributes - measured at 16.2 ms per single-row prediction with
threads left to default against 10.4 ms pinned to one, a third of the time
spent waiting on pool updates. Tree prediction sums the same leaf values in the
same order either way, so the numbers are unchanged.

Deterministic hashing. PYTHONHASHSEED is not set here because it must be set
before the interpreter starts; the scripts do not depend on it, because every
seeded path uses numpy's generator and the simulated reader derives its draw
from a blake2b digest rather than from hash().
"""

from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def threads_for_argv(argv) -> int:
    """How many threads this invocation wants.

    One thread suits the estimator-bound runs, where the work per call is a
    single row and the pool costs more to coordinate than it saves. A real
    reader inverts that completely: one generation step is a matmul over
    billions of parameters, which is exactly what multiple cores are for, so
    clamping to one thread there would cost roughly the physical core count in
    wall time. The choice has to be made before torch is imported, hence the
    argv inspection rather than a parsed flag.
    """
    if "hf" not in argv:
        return 1
    cores = os.cpu_count() or 2
    # Physical cores, approximately: generation is memory-bandwidth bound and
    # hyperthread siblings contend for the same ports rather than adding
    # throughput.
    return max(1, cores // 2)


def pin_threads(n: int = 1) -> None:
    """Limit BLAS/OpenMP threads. Must run before numpy or sklearn import.

    Only sets a variable the caller has not already chosen, so an explicit
    OMP_NUM_THREADS in the environment still wins.
    """
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))
