import http.client
import json
import urllib.error

import pytest
from overtone_core.clients import deezer
from overtone_core.clients.deezer import DeezerClient, DeezerError

ISRC_URL = f"{deezer.TRACK_URL}/isrc:"


def deezer_track(title, artist, duration=200, preview="https://cdn/preview.mp3", contributors=(), deezer_id=1):
    return {
        "id": deezer_id,
        "title": title,
        "artist": {"name": artist},
        "duration": duration,
        "preview": preview,
        "contributors": [{"name": name} for name in contributors],
    }


@pytest.fixture
def fake_api(monkeypatch):
    """fake_api({url prefix: JSON body}): http_get answers from the table."""

    def install(responses):
        def http_get(url, params=None, limiter=None, headers=None):
            return json.dumps(next(body for prefix, body in responses.items() if url.startswith(prefix)))

        monkeypatch.setattr(deezer, "http_get", http_get)

    return install


@pytest.mark.parametrize(
    ("isrc_hit", "matched_by"),
    [
        (deezer_track("I Wanna", "Con Brio"), "isrc"),
        (deezer_track("Je Veux", "Other", contributors=["Con Brio"]), "isrc"),  # credited artist
        (deezer_track("Above", "Kenan Bell"), "title"),  # title and artist both differ
        (deezer_track("I Wanna", "Con Brio", preview=""), "title"),
        ({"id": 1}, "title"),  # stub record
        ({"error": {"code": deezer.NO_DATA}}, "title"),
    ],
)
def test_isrc_hit_is_used_only_if_plausible_else_title_search(fake_api, isrc_hit, matched_by):
    search_hit = deezer_track("I Wanna", "Con Brio", deezer_id=2)
    fake_api({ISRC_URL: isrc_hit, deezer.SEARCH_URL: {"data": [search_hit]}})

    hit, how = DeezerClient().find_preview("I Wanna", "Con Brio", isrc="X1")

    assert (hit["id"], how) == ((1, "isrc") if matched_by == "isrc" else (2, "title"))


@pytest.mark.parametrize(
    ("duration_s", "deezer_duration", "found"),
    [
        (200, 201, True),
        (200, 205, False),  # extra words need durations within 2 s
        (None, 200, False),  # no duration: strict titles
    ],
)
def test_title_search_accepts_extra_words_only_with_close_durations(fake_api, duration_s, deezer_duration, found):
    title = "C.R.E.A.M. (Cash Rules Everything Around Me)"
    fake_api({deezer.SEARCH_URL: {"data": [deezer_track(title, "Wu-Tang Clan", deezer_duration)]}})

    match = DeezerClient().find_preview("C.R.E.A.M.", "Wu-Tang Clan", duration_s=duration_s)

    assert (match is not None) == found


def test_quota_error_raises_instead_of_falling_back(fake_api):
    fake_api({ISRC_URL: {"error": {"code": 4}}})

    with pytest.raises(DeezerError, match="isrc lookup failed"):
        DeezerClient().find_preview("I Wanna", "Con Brio", isrc="X1")


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("down"), TimeoutError(), ConnectionError("reset"), http.client.IncompleteRead(b"")],
)
def test_preview_network_errors_become_deezer_errors(monkeypatch, tmp_path, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(deezer, "http_get", fail)
    monkeypatch.setattr(deezer, "download", fail)
    client, hit = DeezerClient(), deezer_track("Song", "Artist")

    with pytest.raises(DeezerError, match="preview download failed"):
        client.preview_bytes(hit)
    with pytest.raises(DeezerError, match="preview download failed"):
        client.download_preview(hit, tmp_path / "x.mp3")
