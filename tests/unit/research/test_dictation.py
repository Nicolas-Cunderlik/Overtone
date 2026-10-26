"""dictation.py with a stand-in model - no Whisper, no download."""

from types import SimpleNamespace

import dictation
import numpy as np
import pytest


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        return [SimpleNamespace(text=" same reese bass,"), SimpleNamespace(text=" thumpy kick ")], None


def ready(model):
    voice = dictation.Dictation(loader=lambda name: model)
    assert voice.loaded.wait(5)
    return voice


def test_transcribe_joins_segments_and_passes_the_vocabulary_prompt():
    model = FakeModel()
    voice = ready(model)
    pcm = np.zeros(dictation.SAMPLE_RATE, dtype="<f4").tobytes()
    assert voice.status == "ready"
    assert voice.transcribe(pcm) == "same reese bass, thumpy kick"
    samples, options = model.calls[0]
    assert samples.dtype == np.float32 and len(samples) == dictation.SAMPLE_RATE
    assert options["initial_prompt"] == dictation.PROMPT and options["language"] == "en"


@pytest.mark.parametrize(
    ("n_bytes", "message"),
    [(3, "float32"), (0, "no audio"), (4 * (dictation.SAMPLE_RATE * dictation.MAX_SECONDS + 1), "longer than")],
    ids=["torn sample", "empty", "too long"],
)
def test_bad_recordings_are_refused(n_bytes, message):
    with pytest.raises(ValueError, match=message):
        ready(FakeModel()).transcribe(bytes(n_bytes))


def test_a_failed_load_leaves_the_app_working_without_dictation():
    def broken(name):
        raise ImportError("No module named 'faster_whisper'")

    voice = dictation.Dictation(loader=broken)
    assert voice.loaded.wait(5)
    assert voice.status == "unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        voice.transcribe(np.zeros(10, dtype="<f4").tobytes())
