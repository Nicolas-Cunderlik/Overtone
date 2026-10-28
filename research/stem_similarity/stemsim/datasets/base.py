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

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import ClassVar

from .. import config
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


class RateLimiter:
    """At most `per_second` calls per second, spaced evenly."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self._next = 0.0

    def wait(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval


def http_get(url, params=None, limiter=None, headers=None):
    """GET with the project user agent and timeout; returns the body bytes."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    if limiter:
        limiter.wait()
    request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_S) as response:
        return response.read()


def download(url, path, chunk_bytes=1 << 20, progress=None, attempts=8):
    """Streams `url` to `path` via a .part file; no-op if `path` already exists.
    Safe for multi-GB files: a dropped connection resumes from the bytes
    already on disk (HTTP Range), and a file is only renamed into place once
    its size matches the server's Content-Length - servers can close a
    connection early without any error, which would otherwise pass a
    truncated file off as complete."""
    if path.exists():
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": config.USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_S) as response:
                resumed = response.status == 206  # else the server ignored Range: start over
                total = int(response.headers.get("Content-Length", 0)) + (have if resumed else 0)
                done = have if resumed else 0
                with open(tmp, "ab" if resumed else "wb") as out:
                    while chunk := response.read(chunk_bytes):
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                    out.flush()
                    os.fsync(out.fileno())  # on disk before the rename (see write_json_atomic)
            if not total or done >= total:
                tmp.replace(path)
                return path
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # range starts at the end: the .part is already complete
                tmp.replace(path)
                return path
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            pass
        time.sleep(min(60, 2**attempt))
    raise ConnectionError(f"Download still incomplete after {attempts} attempts (re-run to resume): {url}")


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
