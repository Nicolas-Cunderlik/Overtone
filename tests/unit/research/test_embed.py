import numpy as np
import torch
from overtone_core.ml.stems import CHANNELS, STEM_SR
from stemsim import config
from stemsim.registry import DONE, UNAVAILABLE
from stemsim.schema import Track
from stemsim.stages import embed


def test_audio_too_short_for_a_window_is_unavailable_not_done(registry, add_dataset, monkeypatch):
    full, stub = Track("mtat:full", "mtat"), Track("mtat:stub", "mtat")
    add_dataset("magnatagatune", [full, stub])  # a "seeds" dataset: stems separated in memory
    for track in (full, stub):
        registry.set_status(track.track_id, config.AUDIO_STAGE, DONE, artifact="unused")
    seconds = {full.track_id: 30.0, stub.track_id: 0.996}  # the stub is like Deezer's ~1 s placeholder
    monkeypatch.setattr(
        embed,
        "load_channels",
        lambda _registry, track_id, _separator: (
            torch.zeros(len(CHANNELS), int(seconds[track_id] * STEM_SR)),
            False,
        ),
    )
    monkeypatch.setattr(embed, "get_embedder", lambda: lambda waves: np.ones((len(waves), 512), np.float32))
    monkeypatch.setattr(embed, "get_separator", lambda: None)

    assert embed.run(registry) == {DONE: 1, UNAVAILABLE: 1}

    state, detail = registry.status(stub.track_id, config.EMBED_STAGE)
    assert (state, detail) == (UNAVAILABLE, "audio is 0.996 s, shorter than 1 s")
    assert registry.output(stub.track_id, config.EMBED_STAGE) is None
    path, _detail = registry.output(full.track_id, config.EMBED_STAGE)
    np.testing.assert_allclose(np.load(path)["windows"][0], [0.0, 30.0])  # done: window 0 is the full audio
