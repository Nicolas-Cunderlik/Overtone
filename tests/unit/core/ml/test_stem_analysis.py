import numpy as np
import pytest
import soundfile as sf
import torch
from overtone_core.ml import stem_analysis
from overtone_core.ml.stem_analysis import AudioTooShort
from overtone_core.ml.stems import CHANNELS, EMBED_DIMS, STEM_SR, STEMS


def fake_embedder(waves):
    """Each channel's vector is filled with its excerpt's length in seconds, so tests can read the window back."""
    return np.full((len(waves), EMBED_DIMS), waves.shape[1] / STEM_SR, np.float32)


def test_separate_puts_the_mix_first_then_the_stems_trimmed_to_the_shortest():
    # Stems as constant waves (drums=1, bass=2, ...), one sample shorter each, so order and trimming show
    def separator(wav, sr):
        return {name: torch.full((STEM_SR - i,), float(i + 1)) for i, name in enumerate(STEMS)}

    wav = np.full((2, STEM_SR), 0.5, np.float32)

    channels = stem_analysis.separate(wav, STEM_SR, separator)

    assert channels.shape == (len(CHANNELS), STEM_SR - (len(STEMS) - 1))
    assert channels[:, 0].tolist() == pytest.approx([0.5, 1.0, 2.0, 3.0, 4.0])  # mix, drums, bass, vocals, other


def test_embeds_the_full_audio_plus_extra_windows_inside_it():
    channels = torch.full((len(CHANNELS), 10 * STEM_SR), 0.1)
    extra = [(2.0, 4.0), (2.0, 4.0), (8.0, 15.0), (9.5, 12.0), (12.0, 14.0)]

    result = stem_analysis.embed_channels(channels, extra, fake_embedder)

    # duplicate dropped, (8, 15) clipped to the end, (9.5, 10) shorter than MIN_WINDOW_S, (12, 14) after the end
    assert result.windows.tolist() == [[0.0, 10.0], [2.0, 4.0], [8.0, 10.0]]
    assert result.emb[:, 0, 0].tolist() == pytest.approx([10.0, 2.0, 2.0])
    assert result.rms_db.shape == (3, len(CHANNELS))
    assert result.rms_db[0, 0] == pytest.approx(20 * np.log10(0.1), abs=1e-3)


def test_audio_shorter_than_one_window_raises():
    channels = torch.zeros((len(CHANNELS), STEM_SR // 2))

    with pytest.raises(AudioTooShort, match="0.5 s, shorter than 1 s"):
        stem_analysis.embed_channels(channels, [], fake_embedder)


def test_load_audio_returns_stereo_for_a_mono_file(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.linspace(-0.5, 0.5, 4410, dtype=np.float32), 44_100)

    wav, sr = stem_analysis.load_audio(path)

    assert (wav.shape, sr) == ((2, 4410), 44_100)
    np.testing.assert_array_equal(wav[0], wav[1])
