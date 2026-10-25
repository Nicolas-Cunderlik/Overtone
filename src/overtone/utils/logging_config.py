"""Configure structured logging for the application."""

import logging
import os
import sys
from logging import FileHandler, Formatter, StreamHandler

from .. import config


def configure_logging():
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    fmt = Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # The windowed (console=False) build has no stderr to write to
    if sys.stderr is not None:
        ch = StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    fh = FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Third-party DEBUG output is noisy; spotipy/urllib3's includes the client secret (Basic auth
    # header) and refresh tokens - never let those reach the log.
    for name in ("spotipy", "urllib3", "openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.INFO)


# Auto-configure on import for convenience
configure_logging()
