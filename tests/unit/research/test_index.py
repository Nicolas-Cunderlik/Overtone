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


@pytest.fixture
def embed(registry):
    """embed(track, scale, rms_db): writes an embedding file whose full-audio
    window is `scale` everywhere, plus a 10-20 s excerpt window, and marks it done."""

    def write(track, scale, rms_db=-20.0):
        relative = f"embeddings/{abs(hash(track.track_id))}.npz"
        path = registry.data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        emb = np.stack([np.full((len(CHANNELS), index.EMBED_DIMS), scale), np.ones((len(CHANNELS), index.EMBED_DIMS))])
        windows = np.array([[0.0, 30.0], [10.0, 20.0]])
        levels = np.full((2, len(CHANNELS)), rms_db)
        np.savez(path, windows=windows, channels=np.array(CHANNELS), emb=emb.astype(np.float32), rms_db=levels)
        registry.set_status(track.track_id, config.EMBED_STAGE, DONE, artifact=relative.replace("/", "\\"))

    return write


def test_build_writes_catalog_songs_in_row_order(registry, add_dataset, embed, tmp_path):
    b_song = song_track("B Song", "Artist", spotify_track_id="sp-b")
    a_song = song_track("A Song", "Artist", spotify_track_id="sp-a")
    failed = song_track("Broken", "Artist", spotify_track_id="sp-x")
    add_dataset("spotify_catalog", [b_song, a_song, failed])
    other = song_track("Elsewhere", "Artist")
    add_dataset("dim_sim", [other])  # embedded, but not a catalog song
    embed(b_song, 2.0, rms_db=-60.0)
    embed(a_song, 3.0)
    embed(other, 1.0)
    registry.set_status(failed.track_id, config.EMBED_STAGE, FAILED, "corrupt")

    built = index.build(registry, WEIGHTS, silent_db=-45.0, source={"weights_run": "run1"}, root=tmp_path)

    assert built.version == "v1"
    tracks = pd.read_csv(built.path / "tracks.csv")
    assert tracks["spotify_track_id"].tolist() == ["sp-a", "sp-b"]  # sorted by track_id; failed one skipped
    assert tracks["title"].tolist() == ["A Song", "B Song"]
    for channel in CHANNELS:
        vectors = np.load(built.path / f"{channel}.npy")
        assert vectors.shape == (2, index.EMBED_DIMS)
        assert vectors.dtype == np.float16
        np.testing.assert_allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-3)
    levels = np.load(built.path / "rms_db.npy")
    np.testing.assert_array_equal(levels[:, 0], [-20.0, -60.0])  # full-audio window, in row order

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
        index.build(registry, WEIGHTS, silent_db=-45.0, source={}, root=tmp_path)
    assert model_store.versions("stem-muq", root=tmp_path) == []


def test_build_rejects_an_embedding_without_the_full_audio(registry, add_dataset, tmp_path):
    song = song_track("Stub", "Artist", spotify_track_id="sp-s")
    add_dataset("spotify_catalog", [song])
    np.savez(
        tmp_path / "empty.npz",
        windows=np.zeros((0, 2)),
        channels=np.array(CHANNELS),
        emb=np.zeros((0, len(CHANNELS), index.EMBED_DIMS)),
        rms_db=np.zeros((0, len(CHANNELS))),
    )
    registry.set_status(song.track_id, config.EMBED_STAGE, DONE, artifact="empty.npz")

    with pytest.raises(ValueError, match="0 windows"):
        index.build(registry, WEIGHTS, silent_db=-45.0, source={}, root=tmp_path / "models")
    assert model_store.versions("stem-muq", root=tmp_path / "models") == []
