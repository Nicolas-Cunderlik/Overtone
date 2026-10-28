"""Where the app's files live, and env-overridable settings.

Paths never depend on the working directory. Three kinds of files:
  - Shipped with the app: assets (inside the package) and the trained model
    (models/ at PROJECT_ROOT). PROJECT_ROOT is the repo root when running
    from source; in the PyInstaller build it's the unpacked bundle
    (sys._MEIPASS), which mirrors the same layout.
  - Edited by the user: the .env with API keys - in the repo root, or next to
    the exe when frozen.
  - Written at runtime: login token, log, caches - in the per-user app data
    dir (%LOCALAPPDATA%\\Overtone on Windows), never in the repo.
"""

import os
import sys

from platformdirs import user_data_dir

APP_NAME = "Overtone"
FROZEN = getattr(sys, "frozen", False)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(PACKAGE_DIR)))

ASSET_DIR = os.path.join(PACKAGE_DIR, "assets")
APP_ICON_PATH = os.path.join(ASSET_DIR, "SpotifyAppLogo.png")
TITLE_FONT_PATH = os.path.join(ASSET_DIR, "KdamThmorPro-Regular.ttf")  # optional, not in git

ENV_FILE = os.environ.get(
    "OVERTONE_ENV_FILE",
    os.path.join(os.path.dirname(sys.executable), ".env") if FROZEN else os.path.join(PROJECT_ROOT, ".env"),
)

APP_DATA_DIR = os.environ.get("OVERTONE_DATA_DIR", user_data_dir(APP_NAME, appauthor=False))
SPOTIFY_TOKEN_CACHE_PATH = os.path.join(APP_DATA_DIR, "spotify_token.json")
LOG_FILE = os.environ.get("OVERTONE_LOG_FILE", os.path.join(APP_DATA_DIR, "logs", "overtone.log"))
CHORD_CACHE_DIR = os.environ.get("OVERTONE_CHORD_CACHE_DIR", os.path.join(APP_DATA_DIR, "cache", "chords"))
APP_TITLE = os.environ.get("OVERTONE_APP_TITLE", APP_NAME)
