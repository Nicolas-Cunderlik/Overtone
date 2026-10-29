import json

import pytest
from overtone_core.clients import deezer as deezer_api
from stemsim import config
from stemsim.registry import DONE, FAILED, UNAVAILABLE
from stemsim.schema import Membership, NormalizedDataset, song_track
from stemsim.stages.audio import DeezerResolver

ISRC_URL = f"{deezer_api.TRACK_URL}/isrc:"
SEARCH_URL = deezer_api.SEARCH_URL


class FakeAdapter:
    name, target, metric, license, url = "catalog", "seeds", None, "-", "-"


def deezer_track(title, artist, duration=200, preview="https://cdn/preview.mp3", contributors=()):
    return {
        "id": 1,
        "title": title,
        "artist": {"name": artist},
        "duration": duration,
        "preview": preview,
        "contributors": [{"name": name} for name in contributors],
    }


@pytest.fixture
def resolve(registry, monkeypatch):
    """resolve(title, artist, isrc_status, deezer={url prefix: JSON}, **meta) -> outcome, requested URLs."""

    def run(title, artist, isrc_status, deezer=None, **meta):
        track = song_track(title, artist, **meta)
        registry.replace_dataset(
            FakeAdapter(), NormalizedDataset(tracks=[track], members=[Membership(track.track_id, "1")])
        )
        if isrc_status:
            registry.set_status(track.track_id, config.ISRC_STAGE, *isrc_status)
        requested = []

        def http_get(url, params=None, limiter=None, headers=None):
            requested.append(url)
            return json.dumps(next(body for prefix, body in (deezer or {}).items() if url.startswith(prefix)))

        monkeypatch.setattr(deezer_api, "http_get", http_get)
        monkeypatch.setattr(deezer_api, "download", lambda url, path: path)
        return DeezerResolver(registry).resolve(registry.dataset_tracks("catalog")[0]), requested

    return run


@pytest.mark.parametrize("isrc_status", [None, (FAILED, "network")])
def test_waits_for_the_isrc_stage(resolve, isrc_status):
    assert resolve("Song", "Artist", isrc_status) == (None, [])


@pytest.mark.parametrize(
    ("isrc_hit", "expected"),
    [
        (deezer_track("I Wanna", "Con Brio"), (DONE, "isrc")),
        (deezer_track("Je Veux", "Other", contributors=["Con Brio"]), (DONE, "isrc")),  # credited artist
        (deezer_track("Above", "Kenan Bell"), (DONE, "title")),  # title and artist both differ
        (deezer_track("I Wanna", "Con Brio", preview=""), (DONE, "title")),
        ({"id": 1}, (DONE, "title")),  # stub record
        ({"error": {"code": deezer_api.NO_DATA}}, (DONE, "title")),
        ({"error": {"code": 4}}, (FAILED, None)),  # quota: retry, don't fall back
    ],
)
def test_isrc_lookup_or_title_fallback(resolve, isrc_hit, expected):
    deezer = {ISRC_URL: isrc_hit, SEARCH_URL: {"data": [deezer_track("I Wanna", "Con Brio")]}}
    (state, detail, _), _ = resolve("I Wanna", "Con Brio", (DONE, json.dumps({"isrc": "X1"})), deezer)
    assert (state, json.loads(detail)["matched_by"] if state == DONE else None) == expected


@pytest.mark.parametrize(
    ("own_duration", "spotify_duration", "deezer_duration", "expected"),
    [
        (200, None, 201, DONE),
        (200, None, 205, UNAVAILABLE),  # extra words need durations within 2 s
        (None, 200, 200, DONE),  # Spotify's duration stands in for a missing one
        (None, None, 200, UNAVAILABLE),  # no duration: strict titles
    ],
)
def test_title_search_extra_words_need_close_durations(
    resolve, own_duration, spotify_duration, deezer_duration, expected
):
    isrc_status = (DONE, json.dumps({"duration_s": spotify_duration}))
    deezer = {
        SEARCH_URL: {
            "data": [deezer_track("C.R.E.A.M. (Cash Rules Everything Around Me)", "Wu-Tang Clan", deezer_duration)]
        }
    }
    meta = {"duration_s": own_duration} if own_duration else {}
    (state, _, _), _ = resolve("C.R.E.A.M.", "Wu-Tang Clan", isrc_status, deezer, **meta)
    assert state == expected
