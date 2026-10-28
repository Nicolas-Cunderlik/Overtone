import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# stemsim is a research package, not installed - import it from its folder
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research" / "stem_similarity"))

from stemsim.registry import Registry  # noqa: E402
from stemsim.schema import Membership, NormalizedDataset  # noqa: E402


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
