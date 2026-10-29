import numpy as np
import pandas as pd
import pytest
from stemsim import config, index
from stemsim.registry import DONE, FAILED
from stemsim.schema import CHANNELS, song_track

from overtone.ml import model_store

WEIGHTS = {
    "final": {"mix": 0.5, "drums": 0.1, "bass": 0.1, "vocals": 0.1, "other": 0.2},
    "z_mean": dict.fromkeys(CHANNELS, 0.4),
    "z_std": dict.fromkeys(CHANNELS, 0.2),
    "ci_low": dict.fromkeys(CHANNELS, 0.0),  # extra keys in weights.json are ignored
}


def embed(registry, track, hot_dim=0, rms_db=-20.0, n_windows=2, channels=CHANNELS):
    """Writes an embedding file and marks it done. Window 0 (the full audio) is
    3 * one-hot at `hot_dim` in every channel at `rms_db`; window 1 (an excerpt)
    points at the last dim at 0 dB, so tests can tell which window was read."""
    emb = np.zeros((2, len(CHANNELS), index.EMBED_DIMS), np.float32)
    emb[0, :, hot_dim] = 3.0
    emb[1, :, -1] = 1.0
    levels = np.array([[rms_db] * len(CHANNELS), [0.0] * len(CHANNELS)], np.float32)
    windows = np.array([[0.0, 30.0], [10.0, 20.0]])
    relative = f"embeddings/{abs(hash(track.track_id))}.npz"
    (registry.data_dir / "embeddings").mkdir(exist_ok=True)
    np.savez(
        registry.data_dir / relative,
        windows=windows[:n_windows],
        channels=np.array(channels),
        emb=emb[:n_windows],
        rms_db=levels[:n_windows],
    )
    # Stored with "\" as on Windows
    registry.set_status(track.track_id, config.EMBED_STAGE, DONE, artifact=relative.replace("/", "\\"))


def test_build_writes_catalog_songs_full_audio_in_row_order(registry, add_dataset, tmp_path):
    b_song = song_track("B Song", "Artist", spotify_track_id="sp-b")
    a_song = song_track("A Song", "Artist", spotify_track_id="sp-a")
    failed = song_track("Broken", "Artist", spotify_track_id="sp-x")
    add_dataset("spotify_catalog", [b_song, a_song, failed])
    other = song_track("Elsewhere", "Artist")
    add_dataset("dim_sim", [other])  # embedded, but not a catalog song
    embed(registry, b_song, hot_dim=2, rms_db=-60.0)
    embed(registry, a_song, hot_dim=1)
    embed(registry, other)
    registry.set_status(failed.track_id, config.EMBED_STAGE, FAILED, "corrupt")

    built = index.build(registry, WEIGHTS, silent_db=-45.0, source={"weights_run": "run1"}, root=tmp_path)

    assert built.version == "v1"
    tracks = pd.read_csv(built.path / "tracks.csv")
    assert tracks["spotify_track_id"].tolist() == ["sp-a", "sp-b"]  # sorted by track_id; failed one skipped
    assert tracks["title"].tolist() == ["A Song", "B Song"]
    unit_full_audio = np.zeros((2, index.EMBED_DIMS), np.float16)
    unit_full_audio[0, 1] = unit_full_audio[1, 2] = 1.0
    for channel in CHANNELS:
        vectors = np.load(built.path / f"{channel}.npy")
        assert vectors.dtype == np.float16
        np.testing.assert_array_equal(vectors, unit_full_audio)
    np.testing.assert_array_equal(np.load(built.path / "rms_db.npy")[:, 0], [-20.0, -60.0])

    manifest = model_store.read_manifest("stem-muq", "v1", root=tmp_path)
    params = manifest["params"]
    assert params["channels"] == list(CHANNELS)
    assert params["weights"] == WEIGHTS["final"]
    assert params["z_mean"] == WEIGHTS["z_mean"]
    assert params["silent_db"] == -45.0
    assert manifest["source"]["weights_run"] == "run1"
    assert manifest["source"]["songs"] == 2
    assert set(manifest["files"]) == {"tracks.csv", "rms_db.npy", *(f"{c}.npy" for c in CHANNELS)}


def test_build_refuses_an_empty_catalog(registry, tmp_path):
    with pytest.raises(ValueError, match="no spotify_catalog songs"):
        index.build(registry, WEIGHTS, silent_db=-45.0, source={}, root=tmp_path / "models")
    assert model_store.versions("stem-muq", root=tmp_path / "models") == []


@pytest.mark.parametrize("bad", [{"n_windows": 0}, {"channels": CHANNELS[::-1]}], ids=["no_windows", "channel_order"])
def test_build_rejects_an_embedding_it_cannot_read_as_full_audio_per_channel(registry, add_dataset, tmp_path, bad):
    song = song_track("Stub", "Artist", spotify_track_id="sp-s")
    add_dataset("spotify_catalog", [song])
    embed(registry, song, **bad)

    with pytest.raises(ValueError, match="at least the full audio"):
        index.build(registry, WEIGHTS, silent_db=-45.0, source={}, root=tmp_path / "models")
    assert model_store.versions("stem-muq", root=tmp_path / "models") == []
