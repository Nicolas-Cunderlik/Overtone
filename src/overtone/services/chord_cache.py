"""On-disk cache of validated chord suggestions, keyed by Spotify track id.

Entries expire after 24 hours. Survives restarts, so replaying a song (or
reopening the app) doesn't re-bill the LLM call. Only validated output is
stored; entries that no longer match the current schema are treated as misses,
so changing the schema never serves stale-format data.
"""

import json
import logging
import os
import re
import time

from pydantic import ValidationError

from .. import config
from ..music.chord_schema import ChordSuggestions, validate_suggestions

logger = logging.getLogger(__name__)

TTL_SECONDS = 24 * 60 * 60


class ChordCache:
    def __init__(self, cache_dir=config.CHORD_CACHE_DIR, ttl_seconds=TTL_SECONDS):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds

    def _path(self, track_id):
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", track_id)
        return os.path.join(self.cache_dir, f"{safe}.json")

    def get(self, track_id):
        path = self._path(track_id)
        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.warning("Unreadable chord cache entry %s; ignoring", path)
            return None

        if time.time() - entry.get("created_at", 0) > self.ttl_seconds:
            return None
        try:
            suggestions = ChordSuggestions.model_validate(entry["suggestions"])
        except (KeyError, ValidationError):
            return None
        if validate_suggestions(suggestions):
            return None
        return suggestions

    def put(self, track_id, suggestions):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._path(track_id)
        entry = {"created_at": time.time(), "track_id": track_id, "suggestions": suggestions.model_dump()}
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(tmp, path)  # atomic, so a crash never leaves a half-written entry
        except OSError:
            logger.exception("Failed to write chord cache entry %s", path)
