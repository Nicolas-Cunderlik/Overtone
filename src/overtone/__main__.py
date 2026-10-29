"""Entry point: `python -m overtone`, or the `overtone` command once installed.

`--self-test` runs headless (no window, no Spotify): exit code 0 = OK.
"""

import logging
import sys

# torch must load before PyQt5: on Windows, once Qt's libraries are loaded,
# torch's own fail to initialise (WinError 1114 in c10.dll), so a later import
# for analysing a song would crash. Costs ~2 s at startup.
import torch  # noqa: F401

# isort: split
from PyQt5.QtWidgets import QApplication

from overtone.ui.main_window import SpotifyApp

# logging is configured by overtone.utils.logging_config on import
logger = logging.getLogger(__name__)


def self_test():
    """Imports the whole app (done above) and the stem analysis after it, as
    a live analysis does (fails if torch isn't loaded before Qt), then loads
    the pinned stem model and scores one catalog song - enough to catch a
    broken install or missing artifacts without touching Spotify, Deezer or
    showing a window."""
    from overtone_core.ml import stem_analysis  # noqa: F401
    from overtone_core.ml.stem_model import StemModel

    model = StemModel()
    results = model.similar_to(*model.vectors(0), k=3, exclude_rows=[0])
    logger.info(
        "Self-test: stem model %s (%d songs), lookup returned %d results", model.version, len(model), len(results)
    )
    return 0 if len(results) == 3 else 1


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    app = QApplication(argv)
    window = SpotifyApp()  # noqa: F841 - must stay referenced while the app runs
    # window.show() is intentionally omitted - the docking timer in
    # SpotifyApp shows/hides the window as it tracks Spotify's window.
    logger.info("Overtone started")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
