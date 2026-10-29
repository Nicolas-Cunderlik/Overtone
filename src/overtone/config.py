"""Where the app's own files live, and env-overridable settings.

Paths never depend on the working directory. Locations shared with the
research pipeline (repo root, models/, the .env with API keys) are in
overtone_core.config. Here:
  - Shipped with the app: assets, inside the package.
  - Written at runtime: login token, log, caches - in the per-user app data
    dir (%LOCALAPPDATA%\\Overtone on Windows), never in the repo.
"""

import os

from platformdirs import user_data_dir

APP_NAME = "Overtone"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(PACKAGE_DIR, "assets")
APP_ICON_PATH = os.path.join(ASSET_DIR, "SpotifyAppLogo.png")
TITLE_FONT_PATH = os.path.join(ASSET_DIR, "KdamThmorPro-Regular.ttf")  # optional, not in git

APP_DATA_DIR = os.environ.get("OVERTONE_DATA_DIR", user_data_dir(APP_NAME, appauthor=False))
SPOTIFY_TOKEN_CACHE_PATH = os.path.join(APP_DATA_DIR, "spotify_token.json")
LOG_FILE = os.environ.get("OVERTONE_LOG_FILE", os.path.join(APP_DATA_DIR, "logs", "overtone.log"))
CHORD_CACHE_DIR = os.environ.get("OVERTONE_CHORD_CACHE_DIR", os.path.join(APP_DATA_DIR, "cache", "chords"))
APP_TITLE = os.environ.get("OVERTONE_APP_TITLE", APP_NAME)
