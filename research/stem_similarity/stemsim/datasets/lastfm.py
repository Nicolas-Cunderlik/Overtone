"""Last.fm track.getSimilar - behavioral similarity, crawled live.

fetch() asks Last.fm for the songs similar to each seed (the app's catalog
songs), a capped batch per run, caching every response as one JSON file per
seed - including "track not found", so nothing is requested twice. Seeds are
visited in a hashed order, so a partial crawl is a random sample of the
catalog rather than its first artists alphabetically. Re-run to grow it.

normalize() turns each cached list into pairs (seed, similar song) with
Last.fm's match score (0-1) as the value, grouped by seed for ranking metrics
(nDCG). Similar songs become `song:` tracks, so the ones already in the
catalog merge with it, and the rest get Deezer previews like any other song.

Etiquette: Last.fm's API terms ask for a reasonable request rate and an
identifiable user agent; we stay at REQUESTS_PER_SECOND and back off on
rate-limit errors. Key: LASTFM_API_KEY in the repo's .env.
"""

import hashlib
import json
import logging
import time

from overtone.ml import dedup
from overtone.secrets import get_secret

from ..clients.http import NETWORK_ERRORS, RateLimiter, http_get
from ..schema import BEHAVIORAL, NDCG, PAIR, Judgment, Membership, NormalizedDataset, song_track, song_track_id
from .base import DatasetAdapter, write_json_atomic

logger = logging.getLogger(__name__)

API_URL = "https://ws.audioscrobbler.com/2.0/"
REQUESTS_PER_SECOND = 4
SIMILAR_PER_SEED = 50
# Last.fm error codes (https://www.last.fm/api/errorcodes)
NOT_FOUND = 6
RATE_LIMITED = 29
RETRY_LATER = {8, 11, 16, RATE_LIMITED}  # operation failed / offline / temporary / rate limited
FATAL = {4, 9, 10, 26}  # auth failed / bad session / invalid key / suspended key
MAX_CONSECUTIVE_FAILURES = 10


def _cache_name(seed_id):
    return hashlib.sha1(seed_id.encode(), usedforsecurity=False).hexdigest()[:20] + ".json"


class LastFm(DatasetAdapter):
    name = "lastfm"
    target = BEHAVIORAL
    metric = NDCG
    license = "Last.fm API terms (non-commercial use, attribution)"
    url = "https://www.last.fm/api/show/track.getSimilar"

    def __init__(self, seeds_dataset="spotify_catalog"):
        self.seeds_dataset = seeds_dataset

    def fetch(self, registry, limit=200, **_):
        api_key = get_secret("LASTFM_API_KEY")
        if not api_key:
            raise RuntimeError("LASTFM_API_KEY is not set in .env")
        raw = registry.raw_dir(self.name)
        seeds = registry.dataset_tracks(self.seeds_dataset)
        if not seeds:
            raise RuntimeError(f"No seed tracks - ingest {self.seeds_dataset!r} first")
        todo = [
            s
            for s in sorted(seeds, key=lambda s: _cache_name(s["track_id"]))
            if not (raw / _cache_name(s["track_id"])).exists()
        ]
        limiter = RateLimiter(REQUESTS_PER_SECOND)
        fetched = failures = 0
        for seed in todo[:limit]:
            # Catalog titles carry "(feat. X)" / "- Remastered" / "- Remix" qualifiers
            # Last.fm often doesn't know; retry with the bare title when that differs
            titles = [seed["title"]]
            if dedup.base_title(seed["title"]) != dedup.normalize_title(seed["title"]):
                titles.append(dedup.base_title(seed["title"]))
            for query_title in titles:
                body = self._get_similar(api_key, seed["artist"], query_title, limiter)
                if body.get("error") or body.get("similartracks", {}).get("track"):
                    break
            error = body.get("error")
            if error in FATAL:
                raise RuntimeError(f"Last.fm rejected the API key (error {error}): {body.get('message')}")
            if error is not None and error != NOT_FOUND:
                failures += 1
                logger.warning("Last.fm error %s for %s: %s", error, seed["track_id"], body.get("message"))
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError("Too many consecutive Last.fm failures - stopping; re-run later")
                if error == RATE_LIMITED:
                    time.sleep(60)
                continue  # not cached, so it's retried next run
            failures = 0
            similar = body.get("similartracks", {}).get("track", []) if error is None else []
            record = {
                "seed": seed["track_id"],
                "artist": seed["artist"],
                "title": seed["title"],
                "query_title": query_title,
                "fetched_at": time.time(),
                "not_found": error == NOT_FOUND,
                "similar": similar,
            }
            write_json_atomic(raw / _cache_name(seed["track_id"]), record)
            fetched += 1
        logger.info("Last.fm: fetched %d seeds this run, %d still to go", fetched, max(0, len(todo) - fetched))
        return fetched

    @staticmethod
    def _get_similar(api_key, artist, title, limiter):
        params = {
            "method": "track.getsimilar",
            "artist": artist,
            "track": title,
            "autocorrect": 1,
            "limit": SIMILAR_PER_SEED,
            "api_key": api_key,
            "format": "json",
        }
        try:
            return json.loads(http_get(API_URL, params, limiter))
        except NETWORK_ERRORS as exc:
            return {"error": getattr(exc, "code", -1), "message": str(exc)}

    def normalize(self, registry):
        out = NormalizedDataset()
        tracks, members = {}, {}
        best = {}  # (seed, similar) -> match; versions of one song collapse to its best score
        for path in sorted(registry.raw_dir(self.name).glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Can't happen for new files (atomic writes); older ones can be
                # damaged by a crash. Delete it and the next fetch refetches it.
                logger.warning("Unreadable Last.fm cache file %s - delete it and re-run fetch", path.name)
                continue
            seed_id = record["seed"]
            for item in record["similar"]:
                artist = item.get("artist", {}).get("name")
                title = item.get("name")
                if not artist or not title:
                    continue
                similar_id = song_track_id(title, artist)
                if similar_id == seed_id:
                    continue  # Last.fm sometimes lists another version of the seed itself
                tracks.setdefault(similar_id, song_track(title, artist, lastfm_url=item.get("url")))
                members.setdefault(similar_id, item.get("url") or similar_id)
                key = (seed_id, similar_id)
                best[key] = max(best.get(key, 0.0), float(item.get("match", 0)))
            if record["similar"]:
                members.setdefault(seed_id, seed_id)
        out.judgments = [
            Judgment(PAIR, seed, similar, value=match, group=seed) for (seed, similar), match in best.items()
        ]
        out.tracks = list(tracks.values())
        out.members = [Membership(track_id, local_id) for track_id, local_id in sorted(members.items())]
        return out
