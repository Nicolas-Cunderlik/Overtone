import numpy as np
import pytest
from overtone_core.ml.stems import CHANNELS

from overtone.services.stem_cache import StemCache

EMB = np.arange(len(CHANNELS) * 4, dtype=np.float32).reshape(len(CHANNELS), 4)
DB = np.full(len(CHANNELS), -20.0, np.float32)


def test_stores_and_returns_vectors_across_instances(tmp_path):
    StemCache(tmp_path).put("4uLU6hMCjMI75M1A2tKUQC", EMB, DB)

    emb, rms_db = StemCache(tmp_path).get("4uLU6hMCjMI75M1A2tKUQC")

    np.testing.assert_array_equal(emb, EMB)
    np.testing.assert_array_equal(rms_db, DB)
    assert StemCache(tmp_path).get("never_stored") is None


@pytest.mark.parametrize(
    ("write", "description"),
    [
        (lambda path: path.write_bytes(b"not an npz"), "unreadable file"),
        (lambda path: np.savez(path, emb=EMB[:2], rms_db=DB[:2]), "fewer channels than the model"),
        (lambda path: np.savez(path, emb=EMB), "missing levels"),
    ],
)
def test_bad_entries_are_misses_so_the_song_is_analysed_again(tmp_path, write, description):
    cache = StemCache(tmp_path)
    cache.put("track", EMB, DB)
    (entry,) = tmp_path.rglob("track.npz")

    write(entry)

    assert cache.get("track") is None, description


def test_track_ids_become_safe_file_names(tmp_path):
    cache = StemCache(tmp_path)
    cache.put("../escape:id", EMB, DB)

    assert [p.name for p in tmp_path.rglob("*.npz")] == ["___escape_id.npz"]
    assert cache.get("../escape:id") is not None
