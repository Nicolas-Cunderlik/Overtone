import hashlib

import pytest
from overtone_core.ml import model_store


def test_builds_numbered_versions_with_a_manifest(tmp_path):
    for payload in (b"first", b"second"):
        with model_store.new_version("demo", source={"run": "abc"}, params={"k": 3}, root=tmp_path) as build:
            (build.path / "data.bin").write_bytes(payload)

    assert build.version == "v2"
    assert build.path == tmp_path / "demo" / "v2"
    assert model_store.versions("demo", root=tmp_path) == ["v1", "v2"]
    manifest = model_store.read_manifest("demo", "v2", root=tmp_path)
    assert manifest["model"] == "demo"
    assert manifest["version"] == "v2"
    assert manifest["source"] == {"run": "abc"}
    assert manifest["params"] == {"k": 3}
    assert manifest["files"] == {"data.bin": {"sha256": hashlib.sha256(b"second").hexdigest(), "bytes": 6}}
    assert (tmp_path / "demo" / "v1" / "data.bin").read_bytes() == b"first"  # earlier versions untouched


def test_failed_build_leaves_no_version(tmp_path):
    with pytest.raises(RuntimeError), model_store.new_version("demo", root=tmp_path) as build:
        (build.path / "half.bin").write_bytes(b"x")
        raise RuntimeError("crashed mid-build")

    assert model_store.versions("demo", root=tmp_path) == []
    assert list((tmp_path / "demo").iterdir()) == []  # staging folder removed


def test_versions_sort_numerically_and_ignore_other_folders(tmp_path):
    for name in ("v10", "v2", "v1", "scratch", ".staging-x"):
        (tmp_path / "demo" / name).mkdir(parents=True)

    assert model_store.versions("demo", root=tmp_path) == ["v1", "v2", "v10"]
    assert model_store.versions("missing", root=tmp_path) == []


def test_verify_reports_missing_and_changed_files(tmp_path):
    with model_store.new_version("demo", root=tmp_path) as build:
        (build.path / "a.bin").write_bytes(b"a")
        (build.path / "b.bin").write_bytes(b"b")
    assert model_store.verify(build.path) == []

    (build.path / "a.bin").write_bytes(b"changed")
    (build.path / "b.bin").unlink()
    assert model_store.verify(build.path) == ["a.bin doesn't match its size or sha256", "missing b.bin"]

    (build.path / "manifest.json").unlink()
    assert model_store.verify(build.path)[0].startswith("unreadable manifest.json")
