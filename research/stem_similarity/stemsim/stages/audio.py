"""Stage: get every track's audio onto disk (data/audio/<source>/...).

One resolver per Track.audio_source, so a new audio source is one class:
  deezer  30 s preview found by artist + title (same matching rules as the
          app's dedup: same artist, similar title, close duration when known)
  pilot   the pilot's previews in research/stem_similarity/audio/ (what was rated)
  mtat    clip extracted from the MagnaTagATune archive (fetched by
          `ingest magnatagatune --with-audio`); waits, without failing, until
          the archive exists
  other   (e.g. fma) recorded as unavailable until a resolver exists

Outcomes go to stage_status under "audio" with the file path (relative to the
data dir) as the artifact. Network errors are `failed` and retried on the
next run; "no match on Deezer" is `unavailable` (re-check by clearing it).
"""

import hashlib
import json
import logging
import os
import urllib.error
import zipfile

from overtone.ml import dedup

from .. import config
from ..datasets.base import RateLimiter, download, http_get
from ..datasets.magnatagatune import archive_path
from ..registry import DONE, FAILED, UNAVAILABLE

logger = logging.getLogger(__name__)

STAGE = "audio"


class DeezerResolver:
    source = "deezer"
    SEARCH_URL = "https://api.deezer.com/search"
    DURATION_TOLERANCE_S = 12
    REQUESTS_PER_SECOND = 8  # Deezer allows ~50 requests / 5 s

    def __init__(self, registry):
        self.registry = registry
        self.limiter = RateLimiter(self.REQUESTS_PER_SECOND)

    def resolve(self, track):
        if not track["artist"] or not track["title"]:
            return UNAVAILABLE, "no artist/title to search by", None
        duration = json.loads(track["meta"]).get("duration_s")
        query = f"{track['artist']} {dedup.base_title(track['title'])}"
        try:
            hits = json.loads(http_get(self.SEARCH_URL, {"q": query, "limit": 10}, self.limiter)).get("data", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return FAILED, f"search failed: {exc}", None
        want_artist = dedup.normalize_artist(track["artist"])
        candidates = [
            hit
            for hit in hits
            if hit.get("preview")
            and dedup.normalize_artist(hit["artist"]["name"]) == want_artist
            and not (duration and abs(hit["duration"] - duration) > self.DURATION_TOLERANCE_S)
        ]
        # Exact (normalized) title matches first, then fuzzy plain-version matches
        wanted = dedup.normalize_title(track["title"])
        ranked = sorted(candidates, key=lambda hit: dedup.normalize_title(hit["title"]) != wanted)
        for hit in ranked:
            if not same_recording(track["title"], hit["title"]):
                continue
            name = hashlib.sha1(track["track_id"].encode(), usedforsecurity=False).hexdigest()[:20] + ".mp3"
            path = self.registry.audio_dir(self.source) / name
            try:
                self.limiter.wait()
                download(hit["preview"], path)
            except (urllib.error.URLError, TimeoutError) as exc:
                return FAILED, f"preview download failed: {exc}", None
            detail = json.dumps({"deezer_id": hit["id"], "deezer_title": hit["title"]})
            return DONE, detail, str(path.relative_to(self.registry.data_dir))
        return UNAVAILABLE, "no matching Deezer preview", None


def same_recording(wanted_title, found_title):
    """Whether a search hit is the recording we asked for, not another version.

    Equal after the app's title normalization (packaging like "feat." or
    "Remastered" dropped) is a match. Beyond that, a fuzzy match (typos,
    punctuation) is only trusted between two plain titles - if either names a
    distinct recording ("Remix", "Live", "Acoustic", "Extended", or anything
    unrecognised), it must match exactly. Measured on the 300-song pilot: the
    old fuzzy-only rule accepted ~20 wrong versions (another remix, the live
    cut, the extended mix)."""
    wanted, found = dedup.normalize_title(wanted_title), dedup.normalize_title(found_title)
    if wanted == found:
        return True
    if _version_words(wanted) != _version_words(found):
        return False  # e.g. "I Wanna Rock" vs "I Wanna Rock Nasty Mix" - no brackets to split on
    plain = wanted == dedup.base_title(wanted_title) and found == dedup.base_title(found_title)
    return plain and dedup.titles_similar(wanted_title, found_title)


def _version_words(normalized_title):
    """Words marking a distinct recording (remix, live, acoustic...) anywhere in the title."""
    return {m.group(0) for m in dedup._DISTINCT.finditer(dedup._CANONICAL_MIX.sub("", normalized_title))}


class MtatResolver:
    source = "mtat"

    def __init__(self, registry):
        self.registry = registry
        self.archive = None
        if archive_path(registry).exists():
            self.archive = zipfile.ZipFile(archive_path(registry))
            self.members = {name.lstrip("./"): name for name in self.archive.namelist()}

    def resolve(self, track):
        if self.archive is None:
            return None  # archive not downloaded yet - leave pending
        member = self.members.get(track["audio_ref"])
        if member is None:
            return UNAVAILABLE, "clip not in the MagnaTagATune archive", None
        path = self.registry.audio_dir(self.source) / track["audio_ref"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(self.archive.read(member))
        return DONE, None, str(path.relative_to(self.registry.data_dir))


class PilotResolver:
    """The pilot's own Deezer previews (research/stem_similarity/audio/), kept
    as-is because they're exactly what the listener rated."""

    source = "pilot"

    def __init__(self, registry):
        self.registry = registry

    def resolve(self, track):
        path = config.ROOT / track["audio_ref"]
        if not path.exists():
            return UNAVAILABLE, f"pilot file missing: {track['audio_ref']}", None
        return DONE, None, os.path.relpath(path, self.registry.data_dir)


class NoSourceResolver:
    def __init__(self, source):
        self.source = source

    def resolve(self, track):
        return UNAVAILABLE, f"no audio source for {self.source!r} yet", None


RESOLVERS = {resolver.source: resolver for resolver in (DeezerResolver, MtatResolver, PilotResolver)}


def run(registry, source=None, dataset=None, limit=None):
    """Resolves audio for pending tracks; returns {state: count}."""
    resolvers = {}
    counts = {}
    pending = registry.pending(STAGE, audio_source=source, dataset=dataset, limit=limit)
    for i, track in enumerate(pending, start=1):
        src = track["audio_source"]
        if src not in resolvers:
            resolvers[src] = RESOLVERS[src](registry) if src in RESOLVERS else NoSourceResolver(src)
        outcome = resolvers[src].resolve(track)
        if outcome is None:
            counts["waiting"] = counts.get("waiting", 0) + 1
            continue
        state, detail, artifact = outcome
        registry.set_status(track["track_id"], STAGE, state, detail, artifact)
        counts[state] = counts.get(state, 0) + 1
        if i % 50 == 0:
            logger.info("audio: %d/%d processed %s", i, len(pending), counts)
    return counts
