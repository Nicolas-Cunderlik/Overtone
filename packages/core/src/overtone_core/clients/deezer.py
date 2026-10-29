"""Deezer API client: finds a track's 30 s preview by ISRC, else by artist +
title search, and fetches it. The research audio stage and the app's live
inference both use it, so catalog and live songs get their audio the same way.
"""

import http.client
import json
import urllib.error

from .. import matching
from ..ml import dedup
from .http import NETWORK_ERRORS, RateLimiter, download, http_get

SEARCH_URL = "https://api.deezer.com/search"
TRACK_URL = "https://api.deezer.com/track"
REQUESTS_PER_SECOND = 8  # Deezer allows ~50 requests / 5 s
SEARCH_LIMIT = 10
NO_DATA = 800  # Deezer's error code for "not found"; others (e.g. 4, quota) are transient
# What fetching a preview can raise once http's own retries are spent
DOWNLOAD_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead)


class DeezerError(Exception):
    """A transient failure (network, quota): retry later, don't treat it as no match."""


class DeezerClient:
    def __init__(self):
        self.limiter = RateLimiter(REQUESTS_PER_SECOND)

    def find_preview(self, title, artist, isrc=None, duration_s=None):
        """The Deezer track to take the preview from, as (hit, matched_by)
        with matched_by "isrc" or "title"; None if Deezer has no matching
        preview. Raises DeezerError on a transient failure.

        The ISRC hit is used only if it has a preview and passes
        matching.isrc_match_plausible; otherwise the title search decides,
        under matching.matching_hits (duration_s tightens it)."""
        if isrc:
            hit = self._by_isrc(isrc)
            # Deezer can return a stub record (an id, no title)
            if (
                hit is not None
                and hit.get("title")
                and hit.get("preview")
                and matching.isrc_match_plausible(
                    title,
                    artist,
                    hit["title"],
                    [hit["artist"]["name"], *(c["name"] for c in hit.get("contributors", []))],
                )
            ):
                return hit, "isrc"
        if not artist or not title:
            return None
        query = f"{artist} {dedup.base_title(title)}"
        hits = self._get(SEARCH_URL, {"q": query, "limit": SEARCH_LIMIT}, "search").get("data", [])
        hit = next(
            matching.matching_hits(
                title,
                artist,
                duration_s,
                [hit for hit in hits if hit.get("preview")],
                lambda h: h["title"],
                lambda h: [h["artist"]["name"]],
                lambda h: h["duration"],
            ),
            None,
        )
        return None if hit is None else (hit, "title")

    def download_preview(self, hit, path):
        """Saves the hit's preview MP3 to `path` (no-op if it exists)."""
        try:
            self.limiter.wait()
            download(hit["preview"], path)
        except DOWNLOAD_ERRORS as exc:
            raise DeezerError(f"preview download failed: {exc}") from exc

    def preview_bytes(self, hit):
        """The hit's preview MP3, in memory."""
        try:
            return http_get(hit["preview"], limiter=self.limiter)
        except DOWNLOAD_ERRORS as exc:
            raise DeezerError(f"preview download failed: {exc}") from exc

    def _by_isrc(self, isrc):
        """The Deezer record for this ISRC, or None if Deezer has none."""
        hit = self._get(f"{TRACK_URL}/isrc:{isrc}", None, "isrc lookup")
        if "error" in hit:
            if hit["error"].get("code") != NO_DATA:
                raise DeezerError(f"isrc lookup failed: {hit['error']}")
            return None
        return hit

    def _get(self, url, params, what):
        try:
            return json.loads(http_get(url, params, self.limiter))
        except NETWORK_ERRORS as exc:
            raise DeezerError(f"{what} failed: {exc}") from exc
