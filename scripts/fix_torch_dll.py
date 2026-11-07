#!/usr/bin/env python3
"""Make torch importable on a Windows box with an older VC++ runtime.

The symptom is an import error that names the wrong file:

    OSError: [WinError 126] The specified module could not be found.
    Error loading "...\\torch\\lib\\shm.dll" or one of its dependencies.

shm.dll is fine. Walking the import tables shows the real gap is one file
further in: torch_cpu.dll imports VCRUNTIME140_THREADS.dll, which ships only
with the Visual C++ 2015-2022 redistributable from about 14.30 onwards. A
machine carrying the VS2019 runtime (14.28) has vcruntime140.dll and
vcruntime140_1.dll but not the _threads one, so every other name in the chain
resolves and this one does not.

The obvious fix - install the current redistributable - needs administrator
rights. This does not: the same Microsoft-signed DLL is published in the
`msvc-runtime` wheel, and dropping it into torch's own lib directory is
enough, because torch adds that directory to the DLL search path itself. The
change is scoped to this virtual environment and is undone by deleting one
file.

    python scripts/fix_torch_dll.py            # check, and fix if needed
    python scripts/fix_torch_dll.py --check    # report only, change nothing

Exit status is 0 when torch can be imported afterwards, 1 when it cannot.

Note on the api-ms-win-crt-*.dll names: a dependency walker reports those as
missing on a healthy system too. They are API sets, resolved by the loader
through the apiset schema rather than by finding a file, so they are not
evidence of anything and chasing them wastes time.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

DLL = "vcruntime140_threads.dll"
WHEEL = "msvc-runtime"


def torch_lib_dir() -> str | None:
    """torch's lib directory, without importing torch (which is what fails)."""
    for base in sys.path:
        cand = os.path.join(base, "torch", "lib")
        if os.path.isdir(cand):
            return cand
    return None


def torch_imports() -> str | None:
    """None when torch imports, else the first line of the error.

    Run in a child process: a failed torch import leaves a partially
    initialised module behind, so checking twice in one process is unreliable.
    """
    proc = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return None
    err = (proc.stderr or "").strip().splitlines()
    return err[-1][:160] if err else f"exit {proc.returncode}"


def fetch_dll(dest_dir: str) -> str | None:
    """Download the wheel and pull the one DLL out of it."""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "download", WHEEL,
             "-d", tmp, "--no-deps", "-q"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  could not download {WHEEL}: "
                  f"{(proc.stderr or '').strip().splitlines()[-1:]}")
            return None
        wheels = glob.glob(os.path.join(tmp, "*.whl"))
        if not wheels:
            print(f"  {WHEEL} produced no wheel for this interpreter")
            return None

        with zipfile.ZipFile(wheels[0]) as z:
            member = next(
                (n for n in z.namelist()
                 if os.path.basename(n).lower() == DLL),
                None,
            )
            if member is None:
                print(f"  {os.path.basename(wheels[0])} does not contain {DLL}")
                return None
            out = os.path.join(dest_dir, DLL)
            with z.open(member) as src, open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="repair a Windows torch import without administrator rights")
    ap.add_argument("--check", action="store_true",
                    help="report the state and change nothing")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("Not Windows; nothing to do.")
        return 0 if torch_imports() is None else 1

    err = torch_imports()
    if err is None:
        print("torch imports fine; nothing to do.")
        return 0

    print("torch does not import:")
    print(f"  {err}")

    lib = torch_lib_dir()
    if lib is None:
        print("  torch is not installed in this environment "
              "(pip install -r requirements-full.txt)")
        return 1

    target = os.path.join(lib, DLL)
    if os.path.exists(target):
        # The DLL is already there, so the import is failing on something
        # else and guessing further would be noise.
        print(f"  {DLL} is already present in {lib}, so this is a different "
              "problem. Run the import-table check in the docstring above.")
        return 1

    print(f"  {DLL} is absent from {lib}")
    if args.check:
        print("  --check given, so nothing was changed. Re-run without it to fix.")
        return 1

    print(f"fetching {DLL} from the {WHEEL} wheel...")
    got = fetch_dll(lib)
    if got is None:
        print("could not obtain the DLL. The alternative is to install the "
              "Visual C++ 2015-2022 x64 redistributable, which needs "
              "administrator rights.")
        return 1
    print(f"  installed {got}")

    err = torch_imports()
    if err is None:
        print("torch now imports. Real-model runs are unblocked.")
        return 0
    print("torch still does not import:")
    print(f"  {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
