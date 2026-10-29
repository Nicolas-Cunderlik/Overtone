import numpy as np
import pandas as pd
import pytest
from overtone_core.ml import model_store
from overtone_core.ml.paths import STEM_CHANNEL_FILE, STEM_LEVELS_FILE, STEM_MODEL, STEM_TRACKS_FILE
from overtone_core.ml.stem_model import StemModel
from overtone_core.ml.stems import CHANNELS

LOUD, SILENT = -20.0, -60.0
# z = (cosine - 0) / 1, so every channel's z-score is its cosine
PARAMS = {
    "channels": list(CHANNELS),
    "weights": {"mix": 0.5, "drums": 0.2, "bass": 0.1, "vocals": 0.1, "other": 0.1},
    "z_mean": dict.fromkeys(CHANNELS, 0.0),
    "z_std": dict.fromkeys(CHANNELS, 1.0),
    "silent_db": -45.0,
}


def unit(*values):
    v = np.array(values, np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def build(tmp_path):
    """build(songs) -> StemModel over a version in tmp_path. Each song:
    (spotify_id, title, artist, {channel: vector}, {channel: rms_db}); missing
    channels get (1, 0, 0) at LOUD."""

    def make(songs, params=PARAMS):
        with model_store.new_version(STEM_MODEL, params=params, root=tmp_path) as built:
            pd.DataFrame(
                [(f"song:{i}", sid, title, artist) for i, (sid, title, artist, _, _) in enumerate(songs)],
                columns=["track_id", "spotify_track_id", "title", "artist"],
            ).to_csv(built.path / STEM_TRACKS_FILE, index=False)
            for channel in CHANNELS:
                vectors = [song[3].get(channel, unit(1, 0, 0)) for song in songs]
                np.save(built.path / STEM_CHANNEL_FILE.format(channel=channel), np.array(vectors, np.float16))
            levels = [[song[4].get(channel, LOUD) for channel in CHANNELS] for song in songs]
            np.save(built.path / STEM_LEVELS_FILE, np.array(levels, np.float32))
        return StemModel(built.version, root=tmp_path)

    return make


def test_finds_catalog_songs_by_spotify_id_or_by_title_and_artist(build):
    model = build([("id0", "Song A", "Artist", {}, {}), ("id1", "Song B", "Artist", {}, {})])

    assert model.find(spotify_track_id="id1") == 1
    assert model.find(spotify_track_id="unknown", title="Song B (feat. Guest) - Remastered", artist="Artist") == 1
    assert model.find(spotify_track_id="unknown", title="Song B (Live)", artist="Artist") is None
    assert model.find(spotify_track_id="unknown") is None


def test_ranks_by_weighted_channel_scores_and_excludes_the_query(build):
    same, half, other = unit(1, 0, 0), unit(1, 1, 0), unit(0, 1, 0)
    model = build(
        [
            ("query", "Query", "A", {}, {}),
            ("same_drums", "Same drums", "B", {"mix": other, "drums": same}, {}),
            ("same_mix", "Same mix", "C", {"mix": same, "drums": other}, {}),
            ("half_mix", "Half mix", "D", {"mix": half, "drums": other}, {}),
        ]
    )
    query = model.find(spotify_track_id="query")

    results = model.similar_to(*model.vectors(query), k=2, exclude_rows=[query])

    # mix weighs 0.5, drums 0.2: same mix (0.5 + 0.3 from bass/vocals/other) beats half mix, then same drums
    assert [r["track_id"] for r in results] == ["same_mix", "half_mix"]
    assert results[0]["track_name"] == "Same mix"
    assert results[0]["track_artist"] == "C"
    assert results[0]["score"] == pytest.approx(0.5 + 0.0 + 0.1 + 0.1 + 0.1, abs=1e-3)
    assert results[0]["channels"] == pytest.approx({"mix": 1.0, "drums": 0.0, "bass": 1.0, "vocals": 1.0, "other": 1.0})


def test_silent_channels_score_minus_one_against_audible_and_zero_against_silent(build):
    model = build(
        [
            ("silent_bass", "Silent bass", "A", {}, {"bass": SILENT}),
            ("loud_bass", "Loud bass", "B", {}, {}),
        ]
    )
    emb, levels = model.vectors(model.find(spotify_track_id="silent_bass"))

    by_id = {r["track_id"]: r["channels"]["bass"] for r in model.similar_to(emb, levels, k=2)}

    assert by_id == pytest.approx({"silent_bass": 0.0, "loud_bass": -1.0})


def test_k_is_capped_by_the_songs_left(build):
    model = build([("a", "A", "X", {}, {}), ("b", "B", "X", {}, {})])

    assert len(model.similar_to(*model.vectors(0), k=3, exclude_rows=[0])) == 1
    assert model.similar_to(*model.vectors(0), k=3, exclude_rows=[0, 1]) == []


def test_rejects_an_index_with_other_channels(build):
    with pytest.raises(ValueError, match="channels"):
        build([("a", "A", "X", {}, {})], params={**PARAMS, "channels": ["mix", "drums"]})
