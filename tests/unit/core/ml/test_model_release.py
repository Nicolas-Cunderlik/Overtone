import os
import zipfile

import pytest
from overtone_core.ml import model_release, model_store
from overtone_core.ml.model_release import ReleaseError

# Downloads come from local file:// URLs: no network


def build_demo(root, payload=b"vectors"):
    """Builds v1 of model "demo" under `root`; returns its folder."""
    with model_store.new_version("demo", params={"k": 1}, root=root) as build:
        (build.path / "mix.npy").write_bytes(payload)
    return build.path


def forbid_downloads(monkeypatch):
    def download(url, path):
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(model_release, "download", download)


@pytest.fixture
def release(tmp_path):
    """demo v1 packed for publishing: (zip path, sha256)."""
    build_demo(tmp_path / "built")
    return model_release.pack("demo", "v1", root=tmp_path / "built", out_dir=tmp_path / "dist")


def test_fetch_installs_a_packed_version_once_and_never_replaces_it(tmp_path, monkeypatch, release):
    archive, digest = release
    root = tmp_path / "installed"

    folder = model_release.fetch("demo", "v1", sha256=digest, url=archive.as_uri(), root=root)

    assert folder == root / "demo" / "v1"
    assert (folder / "mix.npy").read_bytes() == b"vectors"
    assert model_store.read_manifest("demo", "v1", root=root)["params"] == {"k": 1}
    assert [p.name for p in (root / "demo").iterdir()] == ["v1"]  # no zip or staging folder left

    forbid_downloads(monkeypatch)
    assert model_release.fetch("demo", "v1", sha256=digest, url=archive.as_uri(), root=root) == folder

    (folder / "mix.npy").write_bytes(b"damaged")
    with pytest.raises(ReleaseError, match="damaged"):
        model_release.fetch("demo", "v1", sha256=digest, url=archive.as_uri(), root=root)
    assert (folder / "mix.npy").read_bytes() == b"damaged"  # left for the caller to inspect


def test_packing_the_same_version_twice_gives_the_same_zip(tmp_path, release):
    _, digest = release
    for path in (tmp_path / "built" / "demo" / "v1").iterdir():
        an_hour_earlier = path.stat().st_mtime - 3600
        os.utime(path, (an_hour_earlier, an_hour_earlier))  # as a copied or re-extracted folder would have

    _, again = model_release.pack("demo", "v1", root=tmp_path / "built", out_dir=tmp_path / "again")

    assert again == digest  # so the sha256 pinned in paths.py stays valid


def test_pack_refuses_a_version_that_doesnt_match_its_manifest(tmp_path):
    folder = build_demo(tmp_path)
    (folder / "mix.npy").write_bytes(b"edited after the build")

    with pytest.raises(ReleaseError, match="mix.npy"):
        model_release.pack("demo", "v1", root=tmp_path, out_dir=tmp_path / "dist")


@pytest.mark.parametrize(
    ("entries", "pinned", "error"),
    [
        ({"mix.npy": b"vectors"}, "0" * 64, "sha256"),  # not the pinned download
        ({"mix.npy": b"tampered"}, None, "mix.npy"),  # contents don't match the manifest
        ({"../escape.npy": b"x"}, None, "Unexpected entry"),  # a path outside the version folder
    ],
)
def test_fetch_rejects_a_zip_that_doesnt_check_out(tmp_path, entries, pinned, error):
    folder = build_demo(tmp_path / "built")
    archive = tmp_path / "demo-v1.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(folder / "manifest.json", "manifest.json")
        for name, data in entries.items():
            zf.writestr(name, data)
    root = tmp_path / "installed"

    with pytest.raises(ReleaseError, match=error):
        model_release.fetch("demo", "v1", sha256=pinned or model_store.sha256(archive), url=archive.as_uri(), root=root)

    assert list((root / "demo").iterdir()) == []  # no version, staging folder, zip or escaped file


def test_fetch_needs_a_pinned_sha256(tmp_path, monkeypatch):
    forbid_downloads(monkeypatch)

    with pytest.raises(ReleaseError, match="No sha256"):
        model_release.fetch("demo", "v1", sha256=None, root=tmp_path)
