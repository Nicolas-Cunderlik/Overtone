"""Versioned model artifacts: models/<model>/<version>/, one folder per build.

The layout follows TF Serving and Triton model repositories. A model is one
loader and file contract; each build of it is a numbered version (v1, v2, ...)
that is never changed once written. The app loads a pinned version (ml/paths.py),
so a new build changes nothing until the pin moves. A different design (say, a
neural network instead of fixed weights) is a new model name, not a version.

Every version has a manifest.json with a shared header:
  model, version, format     what this is; `format` is the header's own layout
  created_at                 UTC, ISO 8601
  git_commit, git_dirty      the code that built it
  files                      {name: {"sha256", "bytes"}} for every other file
  source                     where the inputs came from (runs, datasets, counts)
  params                     whatever the model's loader needs besides its files

Standard library only: the research builder and the app both use it.
"""

import hashlib
import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .paths import MODELS_DIR

MANIFEST_FILE = "manifest.json"
MANIFEST_FORMAT = 1
_VERSION = re.compile(r"v(\d+)")
_STAGING_PREFIX = ".staging-"


def version_dir(model, version, root=MODELS_DIR):
    return Path(root) / model / version


def versions(model, root=MODELS_DIR):
    """The model's finished versions, oldest first (v2 before v10)."""
    folder = Path(root) / model
    if not folder.is_dir():
        return []
    numbers = [int(m.group(1)) for p in folder.iterdir() if (m := _VERSION.fullmatch(p.name)) and p.is_dir()]
    return [f"v{n}" for n in sorted(numbers)]


def read_manifest(model, version, root=MODELS_DIR):
    return json.loads((version_dir(model, version, root) / MANIFEST_FILE).read_text(encoding="utf-8"))


def verify(folder):
    """What's wrong with a version folder's files against its manifest, as
    messages; empty if every listed file is there with its size and sha256."""
    folder = Path(folder)
    try:
        files = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError) as exc:
        return [f"unreadable {MANIFEST_FILE}: {exc}"]
    problems = []
    for name, expected in files.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif _describe(path) != expected:
            problems.append(f"{name} doesn't match its size or sha256")
    return problems


@contextmanager
def new_version(model, source=None, params=None, root=MODELS_DIR):
    """Builds the next version of `model`.

    Yields an empty staging folder to write the model's files into. On a clean
    exit the manifest is written and the folder is renamed to the next free
    version; on an exception the staging folder is removed. A half-written
    build never appears as a version, and an existing version is never touched.

        with model_store.new_version("stem-muq", source={...}, params={...}) as build:
            np.save(build.path / "mix.npy", mix)
        build.version  # "v3", set after the block
    """
    folder = Path(root) / model
    folder.mkdir(parents=True, exist_ok=True)
    staging = folder / f"{_STAGING_PREFIX}{datetime.now(UTC):%Y%m%dT%H%M%S%f}"
    staging.mkdir()
    build = _Build(staging)
    try:
        yield build
        existing = versions(model, root)
        version = f"v{int(existing[-1][1:]) + 1}" if existing else "v1"
        files = {p.name: _describe(p) for p in sorted(staging.iterdir()) if p.is_file()}
        status = _git("status", "--porcelain")
        manifest = {
            "model": model,
            "version": version,
            "format": MANIFEST_FORMAT,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "files": files,
            "source": source or {},
            "params": params or {},
        }
        (staging / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Fails if another build claimed this number first (Windows and POSIX both refuse a non-empty target)
        staging.rename(folder / version)
        build.version, build.path = version, folder / version
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@dataclass
class _Build:
    path: Path  # the staging folder while building, then the version folder
    version: str | None = None  # set once the build is renamed into place


def sha256(path):
    """Hex sha256 of a file, read in chunks (index files run to hundreds of MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(path):
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def _git(*args):
    """Git output for the repo this file lives in, or None (no git, or a frozen build)."""
    try:
        command = ["git", *args]  # noqa: S607 - git from PATH, fixed arguments
        cwd = Path(__file__).resolve().parent
        run = subprocess.run  # noqa: S603 - fixed git arguments, no user input
        return run(command, capture_output=True, text=True, cwd=cwd, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
