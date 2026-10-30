"""Builds the Windows installer, dist/Overtone-setup-<version>.exe.

    python scripts/build_installer.py

Collects into build/installer/ what installer/overtone.iss packs:
  - wheels of the app and overtone-core (uv build);
  - pylock.toml: every package they need, with exact versions and hashes,
    exported from uv.lock (no dev group, no research pipeline);
  - uv.exe, a pinned release checked against its sha256;
then compiles the installer with Inno Setup 6. Setup downloads the rest
(Python, torch and the other packages, the catalog index, the model weights).
Needs uv on PATH and Inno Setup 6 (winget install JRSoftware.InnoSetup).
"""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from _common import REPO

BUILD = REPO / "build" / "installer"
DIST = REPO / "dist"
ISS = REPO / "installer" / "overtone.iss"
PACKAGES = ("overtone", "overtone-core")  # the app and what it imports; stemsim is research only

# The uv the installer bundles; keep in step with CI's (.github/workflows/ci.yml)
UV_VERSION = "0.13.6"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"
UV_SHA256 = "a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f"
PYTHON_VERSION = "3.13.18"  # installed by uv during setup; the wheels in uv.lock are cp313
DOWNLOAD_TIMEOUT_S = 60

ISCC_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",  # winget --scope user
    Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6" / "ISCC.exe",
)


def main():
    iscc = find_iscc()
    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    # All of build/, not just ours: setuptools reuses build/lib, so a file dropped from the
    # package (e.g. the excluded font) would otherwise stay in the next wheel
    shutil.rmtree(REPO / "build", ignore_errors=True)
    (BUILD / "wheels").mkdir(parents=True)

    for package in PACKAGES:
        uv("build", "--package", package, "--wheel", "--out-dir", str(BUILD / "wheels"))
    export = ["export", "--locked", "--no-dev", "--package", "overtone", "--no-emit-workspace"]
    pylock = uv(*export, "--format", "pylock.toml", capture=True)
    (BUILD / "pylock.toml").write_text(pylock, encoding="utf-8")
    fetch_uv(BUILD / "uv.exe")

    print(f"\n=== Inno Setup ({iscc})", flush=True)
    defines = [f"/DAppVersion={version}", f"/DPythonVersion={PYTHON_VERSION}", f"/DBuildDir={BUILD}"]
    subprocess.run([str(iscc), *defines, f"/O{DIST}", str(ISS)], check=True)  # noqa: S603 - fixed compiler
    print(f"\nBuilt {DIST / f'Overtone-setup-{version}.exe'}")


def find_iscc():
    for candidate in (shutil.which("ISCC"), *ISCC_CANDIDATES):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    sys.exit("Inno Setup 6 not found - install it: winget install JRSoftware.InnoSetup")


def uv(*args, capture=False):
    print(f"\n=== uv {' '.join(args)}", flush=True)  # before uv's own output
    run = subprocess.run(["uv", *args], cwd=REPO, check=True, text=True, capture_output=capture)
    return run.stdout


def fetch_uv(path):
    """Writes the pinned uv.exe to `path`, after checking the release zip's sha256."""
    print(f"\n=== uv {UV_VERSION} for the installer")
    with urllib.request.urlopen(UV_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:  # noqa: S310 - fixed https URL
        data = response.read()
    if (digest := hashlib.sha256(data).hexdigest()) != UV_SHA256:
        sys.exit(f"{UV_URL} has sha256 {digest}, expected {UV_SHA256}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        path.write_bytes(archive.read("uv.exe"))


if __name__ == "__main__":
    main()
