"""Stage: find each Deezer-bound track's ISRC on Spotify, so the audio stage
can fetch that exact recording instead of matching by title.

Tracks with a Spotify id are looked up in batches; the rest, and ids without
an ISRC, are searched by artist + title under the audio stage's matching
rules. Only tracks still without audio are looked up, and without --dataset,
not tracks that only config.NO_AUDIO_DATASETS include.

Outcomes: `done` with detail {"isrc", "duration_s", "spotify_id",
"spotify_title", "via"}; `unavailable` when no matching track has an ISRC;
`failed` on network errors, retried next run.
"""

import json
import logging

from overtone.ml import dedup

from .. import config
from ..clients.http import NETWORK_ERRORS
from ..clients.spotify import MAX_IDS_PER_REQUEST, SpotifyClient
from ..registry import DONE, FAILED, UNAVAILABLE
from .audio import matching_hits

logger = logging.getLogger(__name__)

STAGE = config.ISRC_STAGE
SEARCH_LIMIT = 10


def run(registry, dataset=None, limit=None, client=None):
    """Looks up ISRCs for Deezer-bound tracks still without audio; returns {state: count}.
    Without `dataset`, skips tracks that only config.NO_AUDIO_DATASETS include."""
    pending = registry.pending(
        STAGE,
        audio_source="deezer",
        dataset=dataset,
        limit=limit,
        skip_done=config.AUDIO_STAGE,
        exclude_only_in=() if dataset else config.NO_AUDIO_DATASETS,
    )
    if not pending:
        return {}
    client = client or SpotifyClient.from_env()
    counts = {}

    def record(track, state, detail):
        registry.set_status(track["track_id"], STAGE, state, detail)
        counts[state] = counts.get(state, 0) + 1
        if sum(counts.values()) % 500 == 0:
            logger.info("isrc: %d/%d processed %s", sum(counts.values()), len(pending), counts)

    by_id = [track for track in pending if json.loads(track["meta"]).get("spotify_track_id")]
    to_search = [track for track in pending if not json.loads(track["meta"]).get("spotify_track_id")]
    for start in range(0, len(by_id), MAX_IDS_PER_REQUEST):
        batch = by_id[start : start + MAX_IDS_PER_REQUEST]
        ids = [json.loads(track["meta"])["spotify_track_id"] for track in batch]
        try:
            found = client.get("tracks", {"ids": ",".join(ids)}).get("tracks") or []
        except NETWORK_ERRORS as exc:
            for track in batch:
                record(track, FAILED, f"spotify lookup failed: {exc}")
            continue
        hits = dict(zip(ids, found, strict=False))
        for track, spotify_id in zip(batch, ids, strict=True):
            hit = hits.get(spotify_id)
            if hit and _isrc(hit):
                record(track, DONE, _detail(hit, "spotify_id"))
            else:
                to_search.append(track)  # removed from Spotify, or no ISRC on that release
    for track in to_search:
        record(track, *_search(client, track))
    return counts


def _search(client, track):
    if not track["artist"] or not track["title"]:
        return UNAVAILABLE, "no artist/title to search by"
    query = f"{track['artist']} {dedup.base_title(track['title'])}"
    try:
        response = client.get("search", {"q": query, "type": "track", "limit": SEARCH_LIMIT})
    except NETWORK_ERRORS as exc:
        return FAILED, f"spotify search failed: {exc}"
    hits = [hit for hit in (response.get("tracks") or {}).get("items") or [] if hit and _isrc(hit)]
    duration = json.loads(track["meta"]).get("duration_s")
    hit = next(
        matching_hits(
            track,
            duration,
            hits,
            lambda h: h["name"],
            lambda h: [artist["name"] for artist in h["artists"]],
            lambda h: h["duration_ms"] / 1000,
        ),
        None,
    )
    if hit is None:
        return UNAVAILABLE, "no matching Spotify track with an ISRC"
    return DONE, _detail(hit, "search")


def _isrc(hit):
    return (hit.get("external_ids") or {}).get("isrc")


def _detail(hit, via):
    return json.dumps(
        {
            "isrc": _isrc(hit),
            "duration_s": hit["duration_ms"] / 1000,
            "spotify_id": hit["id"],
            "spotify_title": hit["name"],
            "via": via,
        }
    )
