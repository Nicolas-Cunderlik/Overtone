"""Main GUI for Overtone using services and worker threads for network tasks."""

import logging
import os
import re
import time

from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .. import config
from ..music.midi import progression_to_midi
from ..services import spotify_window_service
from ..services.ai_service import AIService
from ..services.similarity_service import SimilarityService
from ..services.spotify_service import SpotifyService
from ..utils.logging_config import configure_logging
from ..workers.network_worker import FetchDataWorker
from ..workers.playback_worker import PlaybackPollWorker, PlayTrackWorker
from ..workers.similarity_worker import SimilarTracksWorker
from . import theme
from .clickable_list import ClickableListBox
from .fullscreen_image import FullscreenImageWindow
from .title_bar import TITLE_BAR_HEIGHT, TitleBar

configure_logging()
logger = logging.getLogger(__name__)

# Poll cadence (ms). While playing we use a relaxed baseline (catches manual
# pause/skip/seek) but shrink the wait to land right on the track's end, so
# changes still feel instant without polling every second for no reason.
POLL_MS_ACTIVE = 5000
POLL_MS_IDLE = 4000
POLL_MS_MIN = 500
# Consecutive empty polls tolerated before treating playback as stopped, so a
# single transient API hiccup doesn't flash the UI to "No song playing."
IDLE_GRACE_POLLS = 3
# After a click-to-play, how long to wait before re-polling - long enough for
# Spotify to report the new track as current
PLAY_REFRESH_DELAY_MS = 700
# Longest the window waits on close for running workers, all together
SHUTDOWN_WAIT_MS = 6000

# Docking is event-driven (WinEventHook on Spotify's window); this slow poll
# only finds Spotify after it launches and acts as a safety net.
DOCK_POLL_MS = 1000

# Width shared by the album cover and the boxes beneath it
CARD_WIDTH = 260
# Black margin around the content. None on the right: Spotify's window, docked
# against that edge, already has its own black padding on its left.
CONTENT_MARGIN = 9
WINDOW_WIDTH = 291  # 282px content frames + left margin only
ANALYTICS_BOX_WIDTH = (CARD_WIDTH - 4) // 2
ANALYTICS_BOX_HEIGHT = 150

_PANEL_STYLE = f"background-color: {theme.PANEL_BG}; border-radius: 10px;"
_FEEDBACK_INPUT_STYLE = f"""
QPlainTextEdit {{
    background-color: {theme.PANEL_BG}; color: {theme.TEXT}; border: none; border-radius: 6px;
    padding: 4px; font-size: 11px;
}}
"""
_FEEDBACK_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {theme.HOVER_BG}; color: {theme.MUTED_TEXT}; border: none; border-radius: 6px;
    font-family: "{theme.ICON_FONT}"; font-size: 14px;
}}
QPushButton:hover {{ background-color: {theme.BUTTON_HOVER_BG}; color: {theme.TEXT}; }}
QPushButton:pressed {{ background-color: {theme.BUTTON_PRESSED_BG}; }}
"""

MIDI_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")


def _safe_filename(name, max_length=120):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return name[:max_length] or "progression"


def _unique_path(directory, filename):
    """`filename` in `directory`, with " (2)", " (3)"... appended if taken."""
    stem, ext = os.path.splitext(filename)
    path, n = os.path.join(directory, filename), 2
    while os.path.exists(path):
        path, n = os.path.join(directory, f"{stem} ({n}){ext}"), n + 1
    return path


class SpotifyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_TITLE)
        self.setWindowIcon(QIcon(config.APP_ICON_PATH))
        # Frameless so the custom TitleBar can stay black when unfocused. No
        # minimise: the window's visibility follows Spotify's, so a separate
        # minimised state would only drift out of sync - close it instead.
        # (Without WS_MINIMIZEBOX, the taskbar button and Win+M can't minimise
        # it either.)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # Docking shows/hides us as Spotify appears and disappears - that must
        # never pull focus away from Spotify
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Width is fixed; height grows to match Spotify's (see _dock_to_spotify),
        # with the content as the floor - extra height is black padding below it
        # (minimum height is set from the content's natural height, below)
        self.setFixedWidth(WINDOW_WIDTH)
        self.setStyleSheet(f"""
            QWidget {{ background-color: {theme.WINDOW_BG}; }}
            QToolTip {{ color: {theme.TEXT}; background-color: {theme.TOOLTIP_BG};
                        border: 1px solid {theme.HOVER_BG}; padding: 4px; }}
        """)

        # UI elements
        self.content_layout = QVBoxLayout()
        self.title_frame = QFrame()
        self.stats_frame = QFrame()

        # 93 rather than 100, with a 2px bottom margin below: the title font's
        # descent is large (22px vs 46px ascent) and all-caps text never uses
        # it, so centring the line box left the lettering 8px above centre.
        # (Leaves 30px above the lettering and 31px below - the extra pixel
        # was judged by eye on the real app.)
        self.title_frame.setFixedHeight(93)

        self.title_frame.setStyleSheet(_PANEL_STYLE)
        self.stats_frame.setStyleSheet(_PANEL_STYLE)

        self.title_layout = QVBoxLayout()
        self.body_layout = QVBoxLayout()
        self.body_stats_layout = QHBoxLayout()

        self.title_label = QLabel("OVERTONE")

        self.album_cover_label = QLabel(self)
        self.album_cover_label.setPixmap(QPixmap(CARD_WIDTH, CARD_WIDTH))
        # connect click to show fullscreen
        self.album_cover_label.mousePressEvent = self._album_clicked

        title_font_id = QFontDatabase.addApplicationFont(config.TITLE_FONT_PATH)
        font_families = QFontDatabase.applicationFontFamilies(title_font_id)
        if font_families:
            iceberg_font = QFont(font_families[0], 30)
            self.title_label.setFont(iceberg_font)

        self.title_label.setStyleSheet(f"color: {theme.TEXT};")

        self.title_layout.addWidget(self.title_label)

        # ai and similar-tracks frames
        font_family = font_families[0] if font_families else None
        self.chord_box = ClickableListBox(
            "AI Chord Suggestions", font_family, ANALYTICS_BOX_WIDTH, ANALYTICS_BOX_HEIGHT
        )
        self.chord_box.item_clicked.connect(self._download_progression)

        self.similar_tracks = ClickableListBox("Similar Tracks", font_family, ANALYTICS_BOX_WIDTH, ANALYTICS_BOX_HEIGHT)
        self.similar_tracks.item_clicked.connect(self._play_track)

        # Exactly CARD_WIDTH wide and centred, flush with the cover above
        self.body_stats_layout.setSpacing(0)
        self.body_stats_layout.addStretch(1)
        self.body_stats_layout.addWidget(self.chord_box)
        self.body_stats_layout.addSpacing(4)
        self.body_stats_layout.addWidget(self.similar_tracks)
        self.body_stats_layout.addStretch(1)

        self.feedback_frame = self._build_feedback_box(font_family)

        # Cover on top, then the analytics boxes, then feedback on those results
        self.body_layout.setSpacing(8)
        self.body_layout.addWidget(self.album_cover_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.body_layout.addLayout(self.body_stats_layout)
        self.body_layout.addWidget(self.feedback_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        self.title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_layout.setContentsMargins(9, 9, 9, 2)
        self.title_frame.setLayout(self.title_layout)
        self.stats_frame.setLayout(self.body_layout)

        self.content_layout.addWidget(self.title_frame)
        self.content_layout.addSpacing(6)
        self.content_layout.addWidget(self.stats_frame)
        # Nested under the title bar, so it no longer gets the style's
        # top-level margins automatically
        # No top margin: the title frame sits directly under the title bar
        self.content_layout.setContentsMargins(CONTENT_MARGIN, 0, 0, CONTENT_MARGIN)
        self.content_layout.setSpacing(0)  # gaps come from the explicit addSpacing calls

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(TitleBar(self, config.APP_ICON_PATH))
        # Fixed-height container (at the content's natural height) so the
        # content keeps its exact layout and any extra window height lands in
        # the stretch below as black padding
        content = QWidget()
        content.setLayout(self.content_layout)
        content.setFixedHeight(content.sizeHint().height())
        root_layout.addWidget(content)
        root_layout.addStretch(1)
        self.setLayout(root_layout)
        self.setMinimumHeight(content.height() + TITLE_BAR_HEIGHT)
        self.resize(WINDOW_WIDTH, self.minimumHeight())

        # Services
        self.spotify_service = SpotifyService()
        self.ai_service = AIService()
        self.similarity_service = SimilarityService()

        # State
        self.last_song_id = None
        self.current_worker = None
        self.current_similarity_worker = None
        self._workers = []  # keeps superseded workers alive until they finish
        self._idle_streak = 0
        self._poll_worker = None
        self._repoll = False  # a poll was requested while one was in flight
        self._play_worker = None
        self._chord_suggestions = None  # ChordSuggestions for _chord_track, once ready
        self._chord_track = None
        self._last_device_id = None  # fallback target when Spotify has no active device
        self._album_bytes = None  # current cover, reused by the fullscreen view

        # Timer - interval adapts based on playback state (see check_playback)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_playback)
        self.timer.start(POLL_MS_ACTIVE)

        # Docking - follows the Spotify window so this looks like part of it
        self._spotify_hwnd = None
        self._spotify_hook = None
        self.dock_timer = QTimer(self)
        self.dock_timer.timeout.connect(self._dock_to_spotify)
        self.dock_timer.start(DOCK_POLL_MS)

        # Initial check
        self.check_playback()

    def _build_feedback_box(self, font_family):
        """Feedback on the current recommendations: optional free text, then
        thumbs up/down. Not wired yet - the thumbs will be the submit trigger
        and the text optional extra context (see feedback design notes)."""
        frame = QFrame()
        frame.setFixedWidth(CARD_WIDTH)
        frame.setStyleSheet(f"background-color: {theme.BOX_BG}; border-radius: 8px;")

        title = QLabel("Rate These Suggestions")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {theme.TEXT}; font-size: 10px;")
        if font_family:
            title.setFont(QFont(font_family, 10))

        self.feedback_input = QPlainTextEdit()
        self.feedback_input.setPlaceholderText("Optional: what was off, or spot on?")
        self.feedback_input.setFixedHeight(52)
        self.feedback_input.setStyleSheet(_FEEDBACK_INPUT_STYLE)

        # Segoe MDL2 Assets glyphs: Like / Dislike
        self.feedback_up_btn = QPushButton("")
        self.feedback_down_btn = QPushButton("")
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        for btn, tip in ((self.feedback_up_btn, "Good suggestions"), (self.feedback_down_btn, "Poor suggestions")):
            btn.setFixedHeight(30)
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet(_FEEDBACK_BUTTON_STYLE)
            buttons.addWidget(btn)

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self.feedback_input)
        layout.addLayout(buttons)
        return frame

    def _dock_to_spotify(self):
        try:
            hwnd = self._spotify_hwnd
            if hwnd is None or not spotify_window_service.is_window_valid(hwnd):
                hwnd = spotify_window_service.find_spotify_window()
                self._set_spotify_window(hwnd)

            if hwnd is None or spotify_window_service.is_minimized(hwnd):
                self._hide_docked()
                return

            rect = spotify_window_service.get_visible_rect(hwnd)
            if rect is None:
                self._hide_docked()
                return

            # Win32 works in physical pixels; Qt sizes are logical
            dpr = self.devicePixelRatioF()
            width = round(self.width() * dpr)
            min_height = round(self.minimumHeight() * dpr)
            left, top, _right, bottom = rect
            # Match Spotify's height, but never shrink below our content - if
            # Spotify is shorter we simply extend past its bottom edge
            bounds = (left - width, top, width, max(bottom - top, min_height))

            own_hwnd = int(self.winId())
            spotify_window_service.set_window_bounds(own_hwnd, *bounds)

            if not self.isVisible():
                self.show()

            if spotify_window_service.get_foreground_window() == hwnd:
                spotify_window_service.bring_to_top(own_hwnd)
            elif self.isActiveWindow():
                spotify_window_service.place_below(hwnd, own_hwnd)
        except Exception:
            logger.exception("Failed to dock to Spotify window")

    def _set_spotify_window(self, hwnd):
        if self._spotify_hook is not None and self._spotify_hook.hwnd != hwnd:
            self._spotify_hook.unhook()
            self._spotify_hook = None
        self._spotify_hwnd = hwnd
        if hwnd is not None and self._spotify_hook is None:
            self._spotify_hook = spotify_window_service.WindowEventHook(hwnd, self._on_spotify_event)

    def _on_spotify_event(self, event):
        ws = spotify_window_service
        if event == ws.EVENT_SYSTEM_MINIMIZESTART:
            self._hide_docked()  # hide now rather than after the minimise animation
        elif event in (ws.EVENT_OBJECT_DESTROY, ws.EVENT_OBJECT_HIDE):
            self._set_spotify_window(None)
            self._hide_docked()
        else:
            self._dock_to_spotify()

    def changeEvent(self, a0):
        if a0 is not None and a0.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            # Keep the pair together in z-order: when we're activated, pull
            # Spotify up to sit directly beneath us (Spotify's own activation is
            # handled by its EVENT_SYSTEM_FOREGROUND hook, which raises us)
            self._pull_spotify_below()
        super().changeEvent(a0)

    def _pull_spotify_below(self):
        if self._spotify_hwnd is None or not self.isActiveWindow():
            return
        try:
            spotify_window_service.place_below(self._spotify_hwnd, int(self.winId()))
        except Exception:
            logger.exception("Failed to sync z-order with Spotify")

    def _hide_docked(self):
        if self.isVisible():
            self.hide()

    def closeEvent(self, a0):
        # Closing is how the user exits. Stop the timers explicitly - closing
        # only hides the window, so a still-running dock timer would re-show it
        # if anything (e.g. the fullscreen cover) kept the app alive.
        self.dock_timer.stop()
        self.timer.stop()
        if self._spotify_hook is not None:
            self._spotify_hook.unhook()
            self._spotify_hook = None
        # A QThread destroyed while running aborts the process. Ask every worker
        # to stop (similar-tracks workers check between steps), then wait for
        # all of them within one shared deadline.
        workers = [w for w in (self._poll_worker, self._play_worker, *self._workers) if w is not None]
        for worker in workers:
            worker.requestInterruption()
        deadline = time.monotonic() + SHUTDOWN_WAIT_MS / 1000
        for worker in workers:
            worker.wait(max(0, int((deadline - time.monotonic()) * 1000)))
        still_running = [w for w in workers if w.isRunning()]
        if still_running:
            # A model load, a CPU analysis or an AI request can outlast the wait
            # and can't be stopped mid-step. Exit without Qt's teardown, which
            # would abort on the running thread; every cache is written via a
            # temp file + rename, so nothing is left half-written.
            logger.warning("Exiting with %d worker(s) still running", len(still_running))
            logging.shutdown()
            os._exit(0)
        super().closeEvent(a0)
        QApplication.quit()

    def _album_clicked(self, ev):
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            try:
                self.show_fullscreen_cover()
            except Exception:
                logger.exception("Error showing fullscreen cover")

    def check_playback(self):
        """Starts a background poll of Spotify's playback state. The request
        runs on PlaybackPollWorker; _on_playback_polled handles the result on
        the UI thread. At most one poll is in flight - a timer tick during one
        is skipped (see _refresh_now for requests that mustn't be)."""
        if self._poll_worker is not None:
            return
        worker = PlaybackPollWorker(self.spotify_service, parent=self)
        worker.polled.connect(self._on_playback_polled)
        worker.finished.connect(self._on_poll_finished)
        self._poll_worker = worker
        worker.start()

    def _refresh_now(self):
        """Polls now, or right after the in-flight poll if there is one - that
        poll may have been sent before the change we want to see."""
        if self._poll_worker is not None:
            self._repoll = True
        else:
            self.check_playback()

    def _on_poll_finished(self):
        if self._poll_worker is not None:
            self._poll_worker.deleteLater()
        self._poll_worker = None
        if self._repoll:
            self._repoll = False
            self.check_playback()

    def _on_playback_polled(self, state):
        try:
            item = state.get('item') if state else None
            if state and state.get('device'):
                self._last_device_id = state['device'].get('id') or self._last_device_id

            if not item:
                self._idle_streak += 1
                if self._idle_streak >= IDLE_GRACE_POLLS and self.last_song_id is not None:
                    self.last_song_id = None
                    self.timer.setInterval(POLL_MS_IDLE)
                    self._show_idle()
                return

            self._idle_streak = 0
            track_id = item.get('id')
            is_playing = state.get('is_playing', False)

            # Recomputed every poll (not just on change) so the wait keeps
            # shrinking as a track nears its end instead of staying fixed.
            self.timer.setInterval(self._next_interval(state, item, is_playing))

            # Only refetch album/AI/similar on an actual track change - a
            # pause/resume of the same track shouldn't redo that work.
            if track_id == self.last_song_id:
                return

            self.last_song_id = track_id
            self.chord_box.set_loading()
            self.similar_tracks.set_loading()
            self.start_worker(item)

        except Exception:
            logger.exception("Handling playback state failed")
            self.timer.setInterval(POLL_MS_IDLE)

    def _next_interval(self, state, item, is_playing):
        if not is_playing:
            return POLL_MS_IDLE
        duration_ms = item.get('duration_ms')
        progress_ms = state.get('progress_ms')
        if duration_ms is None or progress_ms is None:
            return POLL_MS_ACTIVE
        remaining_ms = duration_ms - progress_ms
        return max(min(POLL_MS_ACTIVE, remaining_ms + 250), POLL_MS_MIN)

    def _show_idle(self):
        self.chord_box.set_idle()
        self.similar_tracks.set_idle()
        # An analysis can outlast playback; drop its result rather than show it over the idle state
        if self.current_similarity_worker is not None:
            self.current_similarity_worker.requestInterruption()
            self.current_similarity_worker = None

    def start_worker(self, track):
        # QThread.terminate() is unsafe if a worker is blocked in a network
        # call, so a superseded worker is left to finish in the background;
        # self._workers keeps it alive, and the sender checks below discard
        # its results once it's no longer self.current_worker.
        try:
            self._chord_suggestions, self._chord_track = None, track
            self._album_bytes = None
            worker = FetchDataWorker(track, self.ai_service)
            worker.album_bytes.connect(self.on_album_bytes)
            worker.ai_ready.connect(self.on_ai_ready)
            worker.error.connect(self.on_worker_error)
            worker.finished.connect(lambda: self._workers.remove(worker))
            self._workers.append(worker)
            self.current_worker = worker
            worker.start()
        except Exception:
            logger.exception("Failed to start FetchDataWorker")
        self._start_similarity_worker(track)

    def _start_similarity_worker(self, track):
        # A superseded analysis is asked to stop at its next step, so it
        # doesn't hold the shared models while the new song waits
        try:
            if self.current_similarity_worker is not None:
                self.current_similarity_worker.requestInterruption()
            worker = SimilarTracksWorker(track, self.similarity_service)
            worker.analyzing.connect(self.on_similarity_analyzing)
            worker.similar_ready.connect(self.on_similar_ready)
            worker.finished.connect(lambda: self._workers.remove(worker))
            self._workers.append(worker)
            self.current_similarity_worker = worker
            worker.start()
        except Exception:
            logger.exception("Failed to start SimilarTracksWorker")

    def on_album_bytes(self, data: bytes):
        if self.sender() is not self.current_worker:
            return
        try:
            self._album_bytes = data
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            pixmap = pixmap.scaled(
                CARD_WIDTH, CARD_WIDTH, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )

            # Rounded pixmap
            rounded = QPixmap(CARD_WIDTH, CARD_WIDTH)
            rounded.fill(QColor("transparent"))
            painter = QPainter(rounded)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(pixmap))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(0, 0, CARD_WIDTH, CARD_WIDTH, 8, 8)
            painter.end()

            self.album_cover_label.setPixmap(rounded)
            self.album_cover_label.setVisible(True)
        except Exception:
            logger.exception("Failed to render album image")

    def on_ai_ready(self, suggestions):
        if self.sender() is not self.current_worker:
            return
        try:
            self._chord_suggestions = suggestions
            if suggestions is None:
                self.chord_box.set_placeholders(["Unavailable for", "this track"])
                return
            items = []
            for i, progression in enumerate(suggestions.progressions):
                symbols = " - ".join(chord.symbol for chord in progression.chords)
                bars = sum(c.duration_beats for c in progression.chords) / progression.beats_per_bar
                tooltip = (
                    f"Click to download MIDI\n{progression.title}\n"
                    f"{progression.key} · {progression.tempo_bpm} BPM · "
                    f"{progression.beats_per_bar}/4 · {bars:g} bars\n{symbols}"
                )
                items.append((i, symbols, tooltip))
            self.chord_box.set_items(items)
        except Exception:
            logger.exception("Failed to update chord suggestions")

    def on_similarity_analyzing(self, loads_models):
        if self.sender() is not self.current_similarity_worker:
            return
        if loads_models:
            self.similar_tracks.set_placeholders(["Analyzing...", "First song loads", "the models (~1 min)"])
        else:
            self.similar_tracks.set_placeholders(["Analyzing..."])

    def on_similar_ready(self, similar):
        if self.sender() is not self.current_similarity_worker:
            return
        try:
            if not similar:  # the reason is in the log
                self.similar_tracks.set_placeholders(["Couldn't find", "similar tracks"])
                return
            self.similar_tracks.set_items(
                [
                    (
                        t["track_id"],
                        f"{t['track_name']} - {t['track_artist']}",
                        f"Click to play now\n{t['track_name']} - {t['track_artist']}",
                    )
                    for t in similar
                ]
            )
        except Exception:
            logger.exception("Failed to update similar tracks")

    def _play_track(self, track_id):
        # One request at a time - ignore clicks while the last is in flight
        if self._play_worker is not None:
            return
        worker = PlayTrackWorker(self.spotify_service, track_id, self._last_device_id, parent=self)
        worker.succeeded.connect(self._on_play_succeeded)
        worker.failed.connect(lambda tid: self.similar_tracks.flash(tid, "Couldn't play this track"))
        worker.finished.connect(self._on_play_finished)
        self._play_worker = worker
        worker.start()

    def _on_play_finished(self):
        if self._play_worker is not None:
            self._play_worker.deleteLater()
        self._play_worker = None

    def _download_progression(self, index):
        suggestions, track = self._chord_suggestions, self._chord_track
        if suggestions is None or track is None:
            return
        progression = suggestions.progressions[index]
        try:
            artist = ", ".join(a.get('name') for a in track.get('artists', []))
            stem = f"{artist} - {track.get('name', 'Unknown')} - {index + 1} {progression.title}"
            path = _unique_path(MIDI_DOWNLOAD_DIR, _safe_filename(stem) + ".mid")
            os.makedirs(MIDI_DOWNLOAD_DIR, exist_ok=True)
            with open(path, "wb") as f:
                f.write(progression_to_midi(progression))
            logger.info("Saved MIDI to %s", path)
            self.chord_box.flash(index, "Saved to Downloads")
        except Exception:
            logger.exception("Failed to save MIDI")
            self.chord_box.flash(index, "Couldn't save MIDI")

    def _on_play_succeeded(self, track_id, outcome):
        logger.info("Similar track %s %s", track_id, outcome)
        # Poll shortly instead of waiting out the current interval (up to
        # POLL_MS_ACTIVE), so the cover/chords/similar tracks follow the click
        QTimer.singleShot(PLAY_REFRESH_DELAY_MS, self._refresh_now)

    def on_worker_error(self, msg):
        logger.error("Worker error: %s", msg)

    def show_fullscreen_cover(self):
        # Reuses the bytes FetchDataWorker already downloaded - no network on
        # the UI thread, and nothing to show until the cover has arrived
        if self._album_bytes is None:
            return
        self.fullscreen_window = FullscreenImageWindow(self._album_bytes)
        self.fullscreen_window.show()
