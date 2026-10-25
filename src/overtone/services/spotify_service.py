"""Spotify service wrapper - lazy init and safe calls."""

import logging
import os
import threading

from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOauthError

from .. import auth

logger = logging.getLogger(__name__)


class SpotifyService:
    def __init__(self):
        self._client = None
        # The client is used from PlaybackPollWorker and PlayTrackWorker
        # threads. Serialize calls so two threads never refresh the
        # token at once - Spotify rotates refresh tokens, and a concurrent
        # refresh can leave one caller holding a revoked token.
        self._lock = threading.RLock()

    def _ensure_client(self):
        if self._client is None:
            try:
                self._client = auth.get_spotify_client()
            except Exception:
                logger.exception("Failed to initialize Spotify client")
                self._client = None

    def current_playback(self):
        with self._lock:
            return self._current_playback()

    def _current_playback(self):
        self._ensure_client()
        if not self._client:
            return None
        try:
            return self._client.current_playback()
        except SpotifyOauthError as e:
            if e.error == 'invalid_grant':
                # Refresh token expired/revoked: retrying the cached token will
                # never succeed. Drop it so the next poll re-runs the login flow.
                logger.warning("Spotify refresh token rejected (%s); clearing cached token", e.error_description)
                self._reset_auth()
            else:
                logger.exception("Spotify auth error")
            return None
        except Exception:
            logger.exception("Error fetching current playback")
            return None

    def play_track(self, track_id, fallback_device_id=None):
        """Starts playing `track_id` now, replacing the current track.

        Returns "played", or "opened" if playback control was refused (e.g.
        403 Premium required / track restricted) and the track was opened in
        the Spotify app instead. Raises on any other failure.
        """
        uri = f"spotify:track:{track_id}"
        with self._lock:
            self._ensure_client()
            if not self._client:
                raise RuntimeError("Spotify client unavailable")
            try:
                self._client.start_playback(uris=[uri])
                return "played"
            except SpotifyException as e:
                if e.http_status == 404 and fallback_device_id:
                    # NO_ACTIVE_DEVICE: Spotify is open but idle - target the
                    # device we last saw playing explicitly
                    logger.info("No active device; retrying on %s", fallback_device_id)
                    self._client.start_playback(device_id=fallback_device_id, uris=[uri])
                    return "played"
                if e.http_status == 403:
                    logger.warning("Playback control refused (%s); opening %s instead", e.msg, uri)
                    os.startfile(uri)  # noqa: S606 - a spotify: URI built from a track id, opened by the Spotify app
                    return "opened"
                raise

    def _reset_auth(self):
        cache_path = getattr(self._client.auth_manager.cache_handler, 'cache_path', None)
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                logger.exception("Failed to remove token cache %s", cache_path)
        self._client = None
