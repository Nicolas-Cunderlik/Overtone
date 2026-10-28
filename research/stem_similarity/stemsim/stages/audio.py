"""Stage: get every track's audio onto disk (data/audio/<source>/...).

One resolver per Track.audio_source, so a new audio source is one class:
  deezer  30 s preview, looked up by the ISRC the isrc stage found (the exact
          recording), else found by artist + title (same matching rules as
          the app's dedup: same artist, similar title, close duration when
          known). Waits, without failing, until the isrc stage has run.
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

STAGE = config.AUDIO_STAGE

# Search hits further than this from the known duration are another cut
DURATION_TOLERANCE_S = 12
# How close durations must be before same_recording accepts a title that only
# adds plain words ("I Like the Way (The Kissing Game)" vs "I Like The Way").
# Measured on the pilot's 25 titles that differ from Deezer's: at 2 s it adds
# 2 right matches and no wrong ones; at 12 s it lets in the acoustic
# "Unpeeled" cut (3.5 s off) and another version (10.4 s off). Wrong versions
# can sit at 0 s, so the title rule still has to rule out remix/live/... words.
EXTRA_WORDS_DURATION_S = 2

NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, json.JSONDecodeError)
DEEZER_NO_DATA = 800  # Deezer's error code for "not found"; others (e.g. 4, quota) are transient


class DeezerResolver:
    source = "deezer"
    SEARCH_URL = "https://api.deezer.com/search"
    TRACK_URL = "https://api.deezer.com/track"
    REQUESTS_PER_SECOND = 8  # Deezer allows ~50 requests / 5 s

    def __init__(self, registry):
        self.registry = registry
        self.limiter = RateLimiter(self.REQUESTS_PER_SECOND)

    def resolve(self, track):
        isrc = self.registry.status(track["track_id"], config.ISRC_STAGE)
        if isrc is None or isrc[0] == FAILED:
            return None  # wait for `stemsim isrc` - title search alone would miss or mis-pick versions
        found = json.loads(isrc[1]) if isrc[0] == DONE else {}
        if found.get("isrc"):
            outcome = self._by_isrc(track, found["isrc"])
            if outcome is not None:
                return outcome
        duration = json.loads(track["meta"]).get("duration_s") or found.get("duration_s")
        return self._by_title(track, duration)

    def _by_isrc(self, track, isrc):
        """The Deezer track with this ISRC, or None to fall back to title search.

        Measured on 200 random catalog songs that title search had marked
        unavailable: 69% got a preview this way, 28% are on Deezer without a
        preview, 3% aren't there."""
        try:
            hit = json.loads(http_get(f"{self.TRACK_URL}/isrc:{isrc}", limiter=self.limiter))
        except NETWORK_ERRORS as exc:
            return FAILED, f"isrc lookup failed: {exc}", None
        if "error" in hit:
            if hit["error"].get("code") != DEEZER_NO_DATA:
                return FAILED, f"isrc lookup failed: {hit['error']}", None
            return None
        # Deezer sometimes returns an empty record (an id, no title) for an ISRC
        if not hit.get("title") or not hit.get("preview") or not isrc_match_plausible(track, hit):
            return None
        return self._download(track, hit, "isrc")

    def _by_title(self, track, duration):
        if not track["artist"] or not track["title"]:
            return UNAVAILABLE, "no artist/title to search by", None
        query = f"{track['artist']} {dedup.base_title(track['title'])}"
        try:
            hits = json.loads(http_get(self.SEARCH_URL, {"q": query, "limit": 10}, self.limiter)).get("data", [])
        except NETWORK_ERRORS as exc:
            return FAILED, f"search failed: {exc}", None
        with_preview = [hit for hit in hits if hit.get("preview")]
        hit = next(
            matching_hits(
                track,
                duration,
                with_preview,
                lambda h: h["title"],
                lambda h: [h["artist"]["name"]],
                lambda h: h["duration"],
            ),
            None,
        )
        if hit is None:
            return UNAVAILABLE, "no matching Deezer preview", None
        return self._download(track, hit, "title")

    def _download(self, track, hit, matched_by):
        name = hashlib.sha1(track["track_id"].encode(), usedforsecurity=False).hexdigest()[:20] + ".mp3"
        path = self.registry.audio_dir(self.source) / name
        try:
            self.limiter.wait()
            download(hit["preview"], path)
        except (urllib.error.URLError, TimeoutError) as exc:
            return FAILED, f"preview download failed: {exc}", None
        detail = json.dumps({"deezer_id": hit["id"], "deezer_title": hit["title"], "matched_by": matched_by})
        return DONE, detail, str(path.relative_to(self.registry.data_dir))


def isrc_match_plausible(track, hit):
    """ISRC data has errors: in the 138 ISRC matches of the 200-song sample,
    "Con Brio - I Wanna" mapped to Kenan Bell's "Above". Rejected only when
    the title AND the artist both disagree - either alone is common on real
    matches (romanised titles, renamed artists, collaborations credited to
    another artist first)."""
    want = dedup.normalize_artist(track["artist"] or "")
    names = [hit["artist"]["name"], *(c["name"] for c in hit.get("contributors", []))]
    return any(dedup.normalize_artist(name) == want for name in names) or dedup.titles_similar(
        track["title"] or "", hit["title"]
    )


def matching_hits(track, duration, hits, title_of, artists_of, duration_of):
    """Search hits that are this track's recording, best first: same artist,
    within DURATION_TOLERANCE_S when the duration is known, and
    same_recording - exact (normalized) title matches before fuzzy ones.
    Shared by the Deezer title search and the isrc stage's Spotify search."""
    want_artist = dedup.normalize_artist(track["artist"])
    wanted = dedup.normalize_title(track["title"])
    candidates = [
        hit
        for hit in hits
        if any(dedup.normalize_artist(name) == want_artist for name in artists_of(hit))
        and not (duration and abs(duration_of(hit) - duration) > DURATION_TOLERANCE_S)
    ]
    for hit in sorted(candidates, key=lambda hit: dedup.normalize_title(title_of(hit)) != wanted):
        close = bool(duration) and abs(duration_of(hit) - duration) <= EXTRA_WORDS_DURATION_S
        if same_recording(track["title"], title_of(hit), durations_match=close):
            yield hit


def same_recording(wanted_title, found_title, durations_match=False):
    """Whether a search hit is the recording we asked for, not another version.

    Equal after the app's title normalization (packaging like "feat." or
    "Remastered" dropped) is a match. Beyond that, a fuzzy match (typos,
    punctuation) is only trusted between two plain titles - if either names a
    distinct recording ("Remix", "Live", "Acoustic", "Extended", or anything
    unrecognised), it must match exactly. Measured on the 300-song pilot: the
    old fuzzy-only rule accepted ~20 wrong versions (another remix, the live
    cut, the extended mix).

    The one exception: with `durations_match` (within EXTRA_WORDS_DURATION_S),
    a title that is the other plus plain words is accepted - "C.R.E.A.M."
    vs "C.R.E.A.M. (Cash Rules Everything Around Me)". Unknown words alone
    can't tell a subtitle from a version ("(Unpeeled)" is an acoustic album),
    so this needs the duration."""
    wanted, found = dedup.normalize_title(wanted_title), dedup.normalize_title(found_title)
    if wanted == found:
        return True
    if _version_words(wanted) != _version_words(found):
        return False  # e.g. "I Wanna Rock" vs "I Wanna Rock Nasty Mix" - no brackets to split on
    if durations_match and _adds_only_plain_words(wanted_title, found_title):
        return True
    plain = wanted == dedup.base_title(wanted_title) and found == dedup.base_title(found_title)
    return plain and dedup.titles_similar(wanted_title, found_title)


def _adds_only_plain_words(title_a, title_b):
    """Whether the titles share a base title and one is the other plus words
    that don't name a version, in order - "Ibiza (X Vs Y) - Z Remix" vs
    "Ibiza (Z Remix)" adds "x vs y"."""
    if dedup.base_title(title_a) != dedup.base_title(title_b):
        return False
    shorter, longer = sorted((dedup.normalize_title(title_a).split(), dedup.normalize_title(title_b).split()), key=len)
    extra, matched = [], 0
    for word in longer:
        if matched < len(shorter) and word == shorter[matched]:
            matched += 1
        else:
            extra.append(word)
    return matched == len(shorter) and not _version_words(" ".join(extra))


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
