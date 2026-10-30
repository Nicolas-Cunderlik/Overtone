"""Entry point: `python -m overtone`, or once installed the `overtone`
command (console) or `overtone-gui` (no console window; the installer's
shortcuts start it).

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

SELF_TEST_AUDIO_S = 5  # synthetic audio the self-test analyses
SELF_TEST_SR = 44_100


def self_test():
    """Imports the whole app (done above) and the stem analysis after it, as
    a live analysis does (fails if torch isn't loaded before Qt), loads the
    pinned stem model and scores one catalog song, then analyses a few
    seconds of synthetic audio with Demucs and MuQ-MuLan. Catches a broken
    install, a torch that can't run, or missing artifacts without touching
    Spotify, Deezer or showing a window. The first run downloads the model
    weights (~4 GB), so the installer runs it."""
    import numpy as np
    from overtone_core.ml import stem_analysis
    from overtone_core.ml.stem_model import StemModel
    from overtone_core.ml.stems import CHANNELS, EMBED_DIMS

    model = StemModel()
    results = model.similar_to(*model.vectors(0), k=3, exclude_rows=[0])
    logger.info(
        "Self-test: stem model %s (%d songs), lookup returned %d results", model.version, len(model), len(results)
    )

    seconds = np.arange(SELF_TEST_AUDIO_S * SELF_TEST_SR) / SELF_TEST_SR
    noise = np.random.default_rng(0).standard_normal(seconds.size)
    tone = 0.3 * np.sin(2 * np.pi * 220 * seconds) + 0.05 * noise
    analysis = stem_analysis.analyse(np.stack([tone, tone]).astype(np.float32), SELF_TEST_SR)
    analysed = analysis.emb.shape == (1, len(CHANNELS), EMBED_DIMS) and bool(np.isfinite(analysis.emb).all())
    logger.info("Self-test: analysis %s", "OK" if analysed else f"returned {analysis.emb.shape} or non-finite values")
    return 0 if len(results) == 3 and analysed else 1


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
