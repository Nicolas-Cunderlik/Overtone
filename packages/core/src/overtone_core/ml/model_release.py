"""Released model versions: a version folder zipped, published as a GitHub
Release asset (paths.MODEL_RELEASE_URL) and fetched by the installer.

    python -m overtone_core.ml.model_release pack    # dist/models/stem-muq-v1.zip and its sha256, to publish
    python -m overtone_core.ml.model_release fetch   # downloads and installs the pinned version (no-op if installed)

A fetched zip is checked twice before its folder is renamed into place: the
zip against the sha256 pinned in paths.py, then every file against the
manifest. A broken or tampered download never shows up as a version.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import truststore

from ..clients.http import download
from ..config import PROJECT_ROOT
from . import model_store
from .paths import MODEL_RELEASE_URL, MODELS_DIR, STEM_MODEL, STEM_MODEL_SHA256, STEM_MODEL_VERSION

DIST_DIR = Path(PROJECT_ROOT) / "dist" / "models"
# Fixed timestamps and order: packing the same version twice gives the same zip, so its sha256 stays valid
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class ReleaseError(Exception):
    """A release that can't be packed or installed: files that don't match
    their manifest or pinned sha256, or no sha256 pinned."""


def release_name(model, version):
    return f"{model}-{version}"


def pack(model=STEM_MODEL, version=STEM_MODEL_VERSION, root=MODELS_DIR, out_dir=DIST_DIR):
    """Zips a finished version for publishing; returns (zip path, sha256).
    Raises ReleaseError if the version's files don't match its manifest."""
    folder = model_store.version_dir(model, version, root)
    if problems := model_store.verify(folder):
        raise ReleaseError(f"{folder}: {'; '.join(problems)}")
    out = Path(out_dir) / f"{release_name(model, version)}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w") as archive:
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            info = zipfile.ZipInfo(path.name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(path, "rb") as src, archive.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    partial.replace(out)
    return out, model_store.sha256(out)


def fetch(model=STEM_MODEL, version=STEM_MODEL_VERSION, sha256=STEM_MODEL_SHA256, url=None, root=MODELS_DIR):
    """Installs a released version as root/<model>/<version>, downloading it
    from `url` (default: its GitHub Release) unless it's already installed;
    returns the version folder.

    Raises ReleaseError when no sha256 is pinned, the download doesn't match
    it, the zip's files don't match its manifest, or an installed copy is
    damaged (it's left for the caller to inspect, never deleted); OSError or
    ConnectionError when the download fails."""
    name = release_name(model, version)
    folder = model_store.version_dir(model, version, root)
    if folder.is_dir():
        if problems := model_store.verify(folder):
            raise ReleaseError(f"{folder} is damaged ({'; '.join(problems)}): delete it and fetch again")
        return folder
    if not sha256:
        raise ReleaseError(f"No sha256 pinned for {name} (paths.py)")

    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / f".{name}.zip"
    download(url or MODEL_RELEASE_URL.format(model=model, version=version), archive)
    try:
        if (actual := model_store.sha256(archive)) != sha256:
            raise ReleaseError(f"Downloaded {name}.zip has sha256 {actual}, expected {sha256}")
        staging = folder.parent / f".fetch-{name}"
        shutil.rmtree(staging, ignore_errors=True)  # left by an interrupted fetch
        try:
            _extract(archive, staging)
            if problems := model_store.verify(staging):
                raise ReleaseError(f"{name}.zip: {'; '.join(problems)}")
            staging.rename(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        archive.unlink(missing_ok=True)  # a bad download is fetched afresh next time
    return folder


def _extract(archive, target):
    """Extracts a release zip, which holds plain file names only."""
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or Path(member.filename).name != member.filename:
                raise ReleaseError(f"Unexpected entry in {archive.name}: {member.filename!r}")
            with zf.open(member) as src, open(target / member.filename, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pack", "fetch"])
    args = parser.parse_args(argv)
    truststore.inject_into_ssl()
    name = release_name(STEM_MODEL, STEM_MODEL_VERSION)
    try:
        if args.command == "pack":
            path, digest = pack()
            print(f"{path}\nsha256 {digest}")
            print(f"Publish it on a GitHub Release tagged {name}, then set STEM_MODEL_SHA256 in paths.py to the sha256")
        else:
            print(f"{name} installed in {fetch()}")
    except (ReleaseError, OSError) as exc:  # OSError covers failed downloads (URLError, ConnectionError)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
