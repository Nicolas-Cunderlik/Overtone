"""The adapter contract. Adding a dataset = one subclass + one line in
datasets/__init__.py; nothing downstream changes.

  fetch(registry, **options)  Get the raw files into registry.raw_dir(name).
                              Dataset-specific, may use the network, may be
                              incremental (Last.fm crawls a little more each
                              run). Must be safe to re-run.
  normalize(registry)         Raw files -> NormalizedDataset (schema.py). Pure:
                              reads only raw_dir (plus the registry, for
                              datasets built on another's tracks), no network,
                              so it's unit-testable from fixture files.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import ClassVar

from ..schema import NormalizedDataset


class DatasetAdapter(ABC):
    name: ClassVar[str]
    target: ClassVar[str]  # schema.PERCEPTUAL / BEHAVIORAL / SEEDS
    metric: ClassVar[str | None]  # schema.TRIPLET_AGREEMENT / SPEARMAN / NDCG
    license: ClassVar[str]
    url: ClassVar[str]

    def fetch(self, registry, **options):  # noqa: B027 - datasets with no raw files skip it
        """Download raw files. Default: nothing to fetch."""

    @abstractmethod
    def normalize(self, registry) -> NormalizedDataset: ...


def write_json_atomic(path, obj):
    """Writes JSON so a crash, kill or sleep mid-write never leaves a broken
    file under the final name: write a temp file, force it to disk, rename.
    (Without the fsync, Windows can record the file's size but not its
    contents - the Last.fm cache once held 66 zero-filled files after the PC slept.)"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
