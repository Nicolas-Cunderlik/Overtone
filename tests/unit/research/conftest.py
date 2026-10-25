import sys
from pathlib import Path

import pytest

# stemsim is a research package, not installed - import it from its folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stemsim.registry import Registry  # noqa: E402


@pytest.fixture
def registry(tmp_path):
    return Registry(data_dir=tmp_path)
