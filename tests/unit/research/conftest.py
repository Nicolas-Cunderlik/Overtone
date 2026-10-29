from types import SimpleNamespace

import pytest
from stemsim.registry import Registry
from stemsim.schema import Membership, NormalizedDataset


@pytest.fixture
def registry(tmp_path):
    return Registry(data_dir=tmp_path)


@pytest.fixture
def add_dataset(registry):
    """add(name, tracks, judgments=()): writes a dataset to the temp registry, every track a member."""

    def add(name, tracks, judgments=()):
        adapter = SimpleNamespace(name=name, target="seeds", metric=None, license="-", url="-")
        members = [Membership(t.track_id, "") for t in tracks]
        registry.replace_dataset(adapter, NormalizedDataset(list(tracks), members, list(judgments)))

    return add
