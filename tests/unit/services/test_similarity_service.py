import numpy as np
import pytest
from overtone_core.ml.stems import CHANNELS

from overtone.services.similarity_service import SimilarityService
from overtone.services.stem_cache import StemCache

LIVE_EMB = np.full((len(CHANNELS), 4), 0.5, np.float32)
LIVE_DB = np.full(len(CHANNELS), -20.0, np.float32)


class FakeModel:
    """Catalog of one song, "catalog_id"; similar_to echoes what it was given."""

    def find(self, spotify_track_id, title, artist):
        return 0 if spotify_track_id == "catalog_id" else None

    def vectors(self, row):
        return np.zeros((len(CHANNELS), 4), np.float32), np.zeros(len(CHANNELS), np.float32)

    def similar_to(self, emb, rms_db, k=3, exclude_rows=()):
        return [
            {
                "track_id": "result",
                "track_name": "Result",
                "track_artist": "Artist",
                "score": float(emb.sum()),
                "channels": dict.fromkeys(CHANNELS, 0.0),
                "excluded": list(exclude_rows),
            }
        ]


class FakeDeezer:
    def __init__(self, match=("hit", "isrc")):
        self.match = match
        self.calls = []

    def find_preview(self, title, artist, isrc, duration_s):
        self.calls.append((title, artist, isrc, duration_s))
        return None if self.match is None else ({"id": 9, "preview": "url"}, self.match[1])

    def preview_bytes(self, hit):
        return b"mp3"


def spotify_track(track_id):
    return {
        "id": track_id,
        "name": "Song",
        "artists": [{"name": "Artist"}, {"name": "Guest"}],
        "external_ids": {"isrc": "ISRC1"},
        "duration_ms": 200_000,
    }


@pytest.fixture
def make_service(tmp_path):
    def make(deezer=None, analyse=None, model_factory=FakeModel):
        analysed = []

        def default_analyse(audio):
            analysed.append(audio)
            return LIVE_EMB, LIVE_DB

        service = SimilarityService(
            model_factory=model_factory,
            deezer=deezer or FakeDeezer(),
            analyse=analyse or default_analyse,
            cache=StemCache(tmp_path),
        )
        return service, analysed

    return make


def test_catalog_song_uses_stored_vectors_and_excludes_itself(make_service):
    deezer = FakeDeezer()
    service, analysed = make_service(deezer)
    analyzing = []

    results = service.find_similar(spotify_track("catalog_id"), on_analyzing=analyzing.append)

    assert results[0]["excluded"] == [0]
    assert (deezer.calls, analysed, analyzing) == ([], [], [])


def test_other_song_is_analysed_from_its_deezer_preview_once_then_cached_on_disk(make_service):
    deezer = FakeDeezer()
    service, analysed = make_service(deezer)
    analyzing = []

    first = service.find_similar(spotify_track("live_id"), on_analyzing=analyzing.append)
    other = service.find_similar(spotify_track("other_live_id"), on_analyzing=analyzing.append)
    restarted, analysed_after_restart = make_service(deezer)  # same cache folder
    again = restarted.find_similar(spotify_track("live_id"), on_analyzing=analyzing.append)

    assert deezer.calls[0] == ("Song", "Artist", "ISRC1", 200.0)  # first artist, ISRC and duration from Spotify
    assert (analysed, analysed_after_restart) == ([b"mp3", b"mp3"], [])
    assert analyzing == [True, False]  # only the first analysis loads the models; the cache hit announces none
    assert first[0]["score"] == other[0]["score"] == again[0]["score"] == pytest.approx(LIVE_EMB.sum())
    assert first[0]["excluded"] == []


@pytest.mark.parametrize(
    ("deezer", "analyse", "model_factory"),
    [
        (FakeDeezer(match=None), None, FakeModel),  # no preview on Deezer
        (None, lambda audio: 1 / 0, FakeModel),  # analysis crashed
        (None, None, lambda: 1 / 0),  # model files missing
    ],
)
def test_failures_return_none(make_service, deezer, analyse, model_factory):
    service, _ = make_service(deezer, analyse, model_factory)

    assert service.find_similar(spotify_track("live_id")) is None


def test_cancelled_request_skips_the_analysis(make_service):
    service, analysed = make_service()

    assert service.find_similar(spotify_track("live_id"), cancelled=lambda: True) is None
    assert analysed == []
