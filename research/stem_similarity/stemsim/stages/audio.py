"""Stage: get every track's audio onto disk (data/audio/<source>/...).

One resolver per Track.audio_source, so a new audio source is one class:
  deezer  30 s preview by ISRC (from the isrc stage), else by artist + title
          search (overtone_core.clients.deezer); waits, without failing,
          until the isrc stage has run
  pilot   the pilot's previews in research/stem_similarity/audio/ (what was rated)
  mtat    clip extracted from the MagnaTagATune archive (fetched by
          `ingest magnatagatune --with-audio`); waits, without failing, until
          the archive exists
  other   (e.g. fma) recorded as unavailable until a resolver exists

Without --dataset, tracks that only config.NO_AUDIO_DATASETS include are
skipped (Last.fm-only similar songs).

Outcomes go to stage_status under "audio" with the file path (relative to the
data dir) as the artifact. Network errors are `failed` and retried on the
next run; "no match on Deezer" is `unavailable` (re-check by clearing it).
"""

import hashlib
import json
import logging
import os
import zipfile

from overtone_core.clients.deezer import DeezerClient, DeezerError

from .. import config
from ..datasets.magnatagatune import archive_path
from ..registry import DONE, FAILED, UNAVAILABLE

logger = logging.getLogger(__name__)

STAGE = config.AUDIO_STAGE


class DeezerResolver:
    """Previews through overtone_core.clients.deezer, the same lookup the app
    uses for live songs."""

    source = "deezer"

    def __init__(self, registry, client=None):
        self.registry = registry
        self.client = client or DeezerClient()

    def resolve(self, track):
        isrc = self.registry.status(track["track_id"], config.ISRC_STAGE)
        if isrc is None or isrc[0] == FAILED:
            return None  # wait for the isrc stage
        found = json.loads(isrc[1]) if isrc[0] == DONE else {}
        duration = json.loads(track["meta"]).get("duration_s") or found.get("duration_s")
        try:
            match = self.client.find_preview(track["title"], track["artist"], found.get("isrc"), duration)
            if match is None:
                if not track["artist"] or not track["title"]:
                    return UNAVAILABLE, "no artist/title to search by", None
                return UNAVAILABLE, "no matching Deezer preview", None
            hit, matched_by = match
            name = hashlib.sha1(track["track_id"].encode(), usedforsecurity=False).hexdigest()[:20] + ".mp3"
            path = self.registry.audio_dir(self.source) / name
            self.client.download_preview(hit, path)
        except DeezerError as exc:
            return FAILED, str(exc), None
        detail = json.dumps({"deezer_id": hit["id"], "deezer_title": hit["title"], "matched_by": matched_by})
        return DONE, detail, str(path.relative_to(self.registry.data_dir))


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

    source = config.PILOT

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
    """Resolves audio for pending tracks; returns {state: count}.
    Without `dataset`, skips tracks that only config.NO_AUDIO_DATASETS include."""
    resolvers = {}
    counts = {}
    pending = registry.pending(
        STAGE,
        audio_source=source,
        dataset=dataset,
        limit=limit,
        exclude_only_in=() if dataset else config.NO_AUDIO_DATASETS,
    )
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
