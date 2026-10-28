"""Stage: find each Deezer-bound track's ISRC on Spotify, so the audio stage
can fetch that exact recording instead of matching by title.

Tracks with a Spotify id are looked up in batches; the rest, and ids without
an ISRC, are searched by artist + title under the audio stage's matching
rules. Only tracks still without audio are looked up.

Outcomes: `done` with detail {"isrc", "duration_s", "spotify_id",
"spotify_title", "via"}; `unavailable` when no matching track has an ISRC;
`failed` on network errors, retried next run. Uses app-only auth, so the
app's user token is never touched.
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.request

from overtone.ml import dedup
from overtone.secrets import get_secret

from .. import config
from ..datasets.base import RateLimiter, http_get
from ..registry import DONE, FAILED, UNAVAILABLE
from .audio import NETWORK_ERRORS, matching_hits

logger = logging.getLogger(__name__)

STAGE = config.ISRC_STAGE
AUTH_URL = "https://accounts.spotify.com/api/token"
API_URL = "https://api.spotify.com/v1"
BATCH_SIZE = 50  # Spotify's maximum ids per /tracks call
SEARCH_LIMIT = 10
REQUESTS_PER_SECOND = 5
ATTEMPTS = 3
MAX_RETRY_AFTER_S = 120  # longer back-offs fail the batch rather than stall the run


class SpotifyClient:
    """Minimal Spotify Web API client: GETs with an app-only token, refreshed
    on 401; short 429 back-offs are waited out."""

    def __init__(self, client_id, client_secret):
        self._credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        self._token = None
        self._expires_at = 0.0
        self.limiter = RateLimiter(REQUESTS_PER_SECOND)

    @classmethod
    def from_env(cls):
        client_id, client_secret = get_secret("SPOTIPY_CLIENT_ID"), get_secret("SPOTIPY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise SystemExit("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set in .env")
        return cls(client_id, client_secret)

    def _access_token(self):
        if self._token is None or time.monotonic() >= self._expires_at:
            request = urllib.request.Request(
                AUTH_URL,
                data=b"grant_type=client_credentials",
                headers={
                    "Authorization": f"Basic {self._credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": config.USER_AGENT,
                },
            )
            with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_S) as response:
                body = json.load(response)
            self._token = body["access_token"]
            self._expires_at = time.monotonic() + body.get("expires_in", 3600) - 60
        return self._token

    def get(self, path, params):
        for attempt in range(ATTEMPTS):
            try:
                headers = {"Authorization": f"Bearer {self._access_token()}"}
                return json.loads(http_get(f"{API_URL}/{path}", params, self.limiter, headers=headers))
            except urllib.error.HTTPError as exc:
                if attempt == ATTEMPTS - 1 or exc.code not in (401, 429):
                    raise
                if exc.code == 401:
                    self._token = None  # expired early: fetch a new one
                    continue
                wait = _retry_after(exc)
                if wait > MAX_RETRY_AFTER_S:
                    raise
                logger.info("isrc: Spotify rate limit, waiting %d s", wait)
                time.sleep(wait)
        raise AssertionError("unreachable")


def _retry_after(exc):
    try:
        return int(exc.headers.get("Retry-After", 5))
    except (TypeError, ValueError):
        return 5


def run(registry, dataset=None, limit=None, client=None):
    """Looks up ISRCs for Deezer-bound tracks still without audio; returns {state: count}."""
    pending = registry.pending(STAGE, audio_source="deezer", dataset=dataset, limit=limit, skip_done=config.AUDIO_STAGE)
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
    for start in range(0, len(by_id), BATCH_SIZE):
        batch = by_id[start : start + BATCH_SIZE]
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
