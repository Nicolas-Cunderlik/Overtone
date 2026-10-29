"""Shared helpers for the utility scripts: repo paths, the venv interpreter, and
whether a pipeline stage is running. Standard library only, so any Python
can run the scripts - each one hands real work to the repo's venv."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"  # one environment for the app and the research pipeline (uv sync)
RESEARCH = REPO / "research" / "stem_similarity"
RESEARCH_DATA = RESEARCH / "data"
LOCK_DIR = RESEARCH_DATA / "locks"
LOG_DIR = RESEARCH_DATA / "logs"


def venv_python():
    """The interpreter inside the repo's venv, or exit with a setup hint if it's missing."""
    for candidate in (VENV / "Scripts" / "python.exe", VENV / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    sys.exit(f"No virtualenv at {VENV} - create it first (uv sync)")


def run(args, cwd=REPO, env=None):
    """Runs a command in the foreground; returns its exit code."""
    return subprocess.call(args, cwd=cwd, env=env)  # noqa: S603 - fixed commands built by these scripts


def running_stages():
    """Names of pipeline stages currently holding their lock (see
    stemsim.registry.exclusive). A lock the OS still holds means the stage's
    process is alive; a leftover lock file from a finished run doesn't count."""
    held = []
    for path in sorted(LOCK_DIR.glob("*.lock")) if LOCK_DIR.exists() else []:
        try:
            with open(path, "a+") as handle:
                if _try_lock(handle):
                    _unlock(handle)
                else:
                    held.append(path.stem)
        except OSError:
            held.append(path.stem)
    return held


if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle):
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(handle):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle):
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(handle):
        fcntl.flock(handle, fcntl.LOCK_UN)
