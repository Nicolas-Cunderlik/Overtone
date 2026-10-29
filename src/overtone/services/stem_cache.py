"""On-disk cache of live songs' stem vectors, keyed by Spotify track id.

Entries never expire: a song's vectors only change with the analysis itself,
and every entry sits under a folder named for the embedding model, so a new
model starts an empty cache. Only vectors are stored (~10 KB per song), never
the preview audio.
"""

import logging
import os
import re

import numpy as np
from overtone_core.ml.stems import CHANNELS, EMBED_MODEL

from .. import config

logger = logging.getLogger(__name__)


class StemCache:
    def __init__(self, cache_dir=config.STEM_CACHE_DIR):
        self.cache_dir = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_-]", "_", EMBED_MODEL))

    def get(self, track_id):
        """(emb (channels, dims), rms_db (channels,)) for the track, or None."""
        path = self._path(track_id)
        try:
            with np.load(path) as entry:
                emb, rms_db = entry["emb"], entry["rms_db"]
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError):
            logger.warning("Unreadable stem cache entry %s; ignoring", path)
            return None
        if emb.shape[0] != len(CHANNELS) or rms_db.shape != (len(CHANNELS),):
            return None
        return emb, rms_db

    def put(self, track_id, emb, rms_db):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._path(track_id)
        tmp = path + ".tmp.npz"
        try:
            np.savez(tmp, emb=emb, rms_db=rms_db)
            os.replace(tmp, path)
        except OSError:
            logger.exception("Failed to write stem cache entry %s", path)

    def _path(self, track_id):
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", track_id)
        return os.path.join(self.cache_dir, f"{safe}.npz")
