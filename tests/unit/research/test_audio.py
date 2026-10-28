import json

import pytest
from stemsim import config
from stemsim.registry import DONE, FAILED, UNAVAILABLE
from stemsim.schema import Membership, NormalizedDataset, song_track
from stemsim.stages import audio
from stemsim.stages.audio import DeezerResolver, same_recording


class FakeAdapter:
    name, target, metric, license, url = "catalog", "seeds", None, "-", "-"


def add_track(registry, title, artist, **meta):
    track = song_track(title, artist, **meta)
    registry.replace_dataset(
        FakeAdapter(), NormalizedDataset(tracks=[track], members=[Membership(track.track_id, "1")])
    )
    return registry.dataset_tracks("catalog")[0]


@pytest.fixture
def deezer(monkeypatch):
    """Fakes Deezer: `responses` maps a URL prefix to the JSON it returns; every
    requested URL is logged and downloads only record their URL."""
    fake = {"responses": {}, "requested": [], "downloaded": []}

    def http_get(url, params=None, limiter=None, headers=None):
        fake["requested"].append(url)
        for prefix, body in fake["responses"].items():
            if url.startswith(prefix):
                return json.dumps(body).encode()
        raise AssertionError(f"unexpected request {url}")

    def download(url, path):
        fake["downloaded"].append(url)
        return path

    monkeypatch.setattr(audio, "http_get", http_get)
    monkeypatch.setattr(audio, "download", download)
    return fake


def deezer_track(title, artist, duration=200, preview="https://cdn/preview.mp3", contributors=()):
    return {
        "id": 1,
        "title": title,
        "artist": {"name": artist},
        "duration": duration,
        "preview": preview,
        "contributors": [{"name": name} for name in contributors],
    }


ISRC_URL = f"{DeezerResolver.TRACK_URL}/isrc:"
SEARCH_URL = DeezerResolver.SEARCH_URL


# --- same_recording ---------------------------------------------------------


@pytest.mark.parametrize(
    ("wanted", "found"),
    [
        ("I Like the Way (The Kissing Game)", "I Like The Way"),
        ("C.R.E.A.M.", "C.R.E.A.M. (Cash Rules Everything Around Me)"),
        (
            "Ibiza (Desaparecidos Vs Walter Master J) - Raf Marchesini & Simone Farina 2K19 Remix Extended",
            "Ibiza (Raf Marchesini & Simone Farina 2k19 Remix Extended)",
        ),
    ],
)
def test_extra_plain_words_match_only_when_durations_do(wanted, found):
    assert same_recording(wanted, found, durations_match=True)
    assert not same_recording(wanted, found)


@pytest.mark.parametrize(
    ("wanted", "found"),
    [
        ("Blue Sky - Michael Brun Remix", "Blue Sky (Michael Brun Extended Remix)"),  # extra word is a version
        ("Close To Me", "Close To Me (Live)"),
        ("Waves", "Waves (Timbaland Remix)"),
        ("Skyline - Klave Remix", "Skyline (Alex Schulz Remix)"),  # different remixer, not extra words
        ("Part 1 (Intro)", "Part 2"),  # different base title
    ],
)
def test_other_versions_never_match_even_with_equal_durations(wanted, found):
    assert not same_recording(wanted, found, durations_match=True)


def test_packaging_and_plain_typos_still_match():
    assert same_recording("Wake Me Up - Radio Edit", "Wake Me Up")
    assert same_recording("Girl I'm on the Way", "Girl Im on the Way")


# --- DeezerResolver -----------------------------------------------------------


def test_deezer_waits_for_the_isrc_stage(registry, deezer):
    track = add_track(registry, "Song", "Artist")
    assert DeezerResolver(registry).resolve(track) is None
    registry.set_status(track["track_id"], config.ISRC_STAGE, FAILED, "network")
    assert DeezerResolver(registry).resolve(track) is None
    assert deezer["requested"] == []


def test_deezer_uses_the_isrc_match_without_searching(registry, deezer):
    track = add_track(registry, "Skyline - Klave Remix", "Artist")
    registry.set_status(track["track_id"], config.ISRC_STAGE, DONE, json.dumps({"isrc": "X1"}))
    deezer["responses"][ISRC_URL] = deezer_track("Skyline (Klave Remix)", "Artist")

    state, detail, artifact = DeezerResolver(registry).resolve(track)

    assert state == DONE and json.loads(detail)["matched_by"] == "isrc"
    assert artifact.startswith("audio")
    assert deezer["requested"] == [f"{ISRC_URL}X1"]


def test_isrc_match_with_other_artist_credited_is_kept(registry, deezer):
    track = add_track(registry, "No Hook", "YBN Nahmir")
    registry.set_status(track["track_id"], config.ISRC_STAGE, DONE, json.dumps({"isrc": "X1"}))
    deezer["responses"][ISRC_URL] = deezer_track("No Hook", "Ybn Almighty Jay", contributors=["YBN Nahmir"])
    assert DeezerResolver(registry).resolve(track)[0] == DONE


@pytest.mark.parametrize(
    "isrc_hit",
    [
        deezer_track("Above", "Kenan Bell"),  # ISRC data error: title and artist both differ
        deezer_track("I Wanna", "Con Brio", preview=""),  # no preview on Deezer
        {"error": {"type": "DataException", "message": "no data", "code": 800}},
        {"id": 63615937, "type": "track"},  # empty record
    ],
)
def test_unusable_isrc_match_falls_back_to_title_search(registry, deezer, isrc_hit):
    track = add_track(registry, "I Wanna", "Con Brio", duration_s=200)
    registry.set_status(track["track_id"], config.ISRC_STAGE, DONE, json.dumps({"isrc": "X1"}))
    deezer["responses"][ISRC_URL] = isrc_hit
    deezer["responses"][SEARCH_URL] = {"data": [deezer_track("I Wanna", "Con Brio", preview="https://cdn/right.mp3")]}

    state, detail, _ = DeezerResolver(registry).resolve(track)

    assert state == DONE and json.loads(detail)["matched_by"] == "title"
    assert deezer["downloaded"] == ["https://cdn/right.mp3"]


def test_deezer_quota_error_on_isrc_lookup_is_retried_not_skipped(registry, deezer):
    track = add_track(registry, "Song", "Artist")
    registry.set_status(track["track_id"], config.ISRC_STAGE, DONE, json.dumps({"isrc": "X1"}))
    deezer["responses"][ISRC_URL] = {"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}}
    assert DeezerResolver(registry).resolve(track)[0] == FAILED
    assert len(deezer["requested"]) == 1


@pytest.mark.parametrize(("deezer_duration", "expected"), [(201, DONE), (205, UNAVAILABLE)])
def test_title_search_accepts_extra_plain_words_only_within_two_seconds(registry, deezer, deezer_duration, expected):
    track = add_track(registry, "C.R.E.A.M.", "Wu-Tang Clan", duration_s=200)
    registry.set_status(track["track_id"], config.ISRC_STAGE, UNAVAILABLE, "no matching Spotify track")
    deezer["responses"][SEARCH_URL] = {
        "data": [deezer_track("C.R.E.A.M. (Cash Rules Everything Around Me)", "Wu-Tang Clan", deezer_duration)]
    }

    assert DeezerResolver(registry).resolve(track)[0] == expected


def test_title_search_uses_the_spotify_duration_when_the_dataset_has_none(registry, deezer):
    track = add_track(registry, "C.R.E.A.M.", "Wu-Tang Clan")  # e.g. dim_sim: no duration of its own
    registry.set_status(track["track_id"], config.ISRC_STAGE, DONE, json.dumps({"duration_s": 200}))
    deezer["responses"][SEARCH_URL] = {
        "data": [deezer_track("C.R.E.A.M. (Cash Rules Everything Around Me)", "Wu-Tang Clan", 200)]
    }
    assert DeezerResolver(registry).resolve(track)[0] == DONE
