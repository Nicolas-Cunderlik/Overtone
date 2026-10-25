"""Worker threads for Spotify playback polling and control, so a slow Spotify
request (or the first-run login flow, which waits on the browser) never
blocks the UI."""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class PlaybackPollWorker(QThread):
    polled = pyqtSignal(object)  # current_playback() state dict, or None

    def __init__(self, spotify_service, parent=None):
        super().__init__(parent)
        self.spotify_service = spotify_service

    def run(self):
        try:
            state = self.spotify_service.current_playback()
        except Exception:
            logger.exception("Playback poll failed")
            state = None
        self.polled.emit(state)


class PlayTrackWorker(QThread):
    succeeded = pyqtSignal(str, str)  # track_id, outcome ("played" / "opened")
    failed = pyqtSignal(str)  # track_id

    def __init__(self, spotify_service, track_id, fallback_device_id=None, parent=None):
        super().__init__(parent)
        self.spotify_service = spotify_service
        self.track_id = track_id
        self.fallback_device_id = fallback_device_id

    def run(self):
        try:
            outcome = self.spotify_service.play_track(self.track_id, self.fallback_device_id)
            self.succeeded.emit(self.track_id, outcome)
        except Exception:
            logger.exception("Failed to play track %s", self.track_id)
            self.failed.emit(self.track_id)
