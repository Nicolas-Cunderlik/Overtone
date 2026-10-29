"""Entry point: `python -m overtone`, or the `overtone` command once installed.

`--self-test` runs headless (no window, no Spotify): exit code 0 = OK.
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from overtone.ui.main_window import SpotifyApp

# logging is configured by overtone.utils.logging_config on import
logger = logging.getLogger(__name__)


def self_test():
    """Imports the whole app (done above), then loads the bundled similarity
    model and runs one lookup - enough to catch a broken build or missing
    artifacts without touching Spotify or showing a window."""
    from overtone_core.ml.similarity_model import SimilarityModel

    model = SimilarityModel()
    results = model.find_similar(model.track_ids[0], k=3)
    logger.info("Self-test: model loaded (%d songs), lookup returned %d results", len(model.track_ids), len(results))
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
