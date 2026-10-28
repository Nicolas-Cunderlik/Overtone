import json
import urllib.error

import pytest
from stemsim import config
from stemsim.registry import DONE, FAILED, UNAVAILABLE
from stemsim.schema import Membership, NormalizedDataset, song_track
from stemsim.stages import isrc


class FakeAdapter:
    name, target, metric, license, url = "catalog", "seeds", None, "-", "-"


def add_tracks(registry, *tracks):
    registry.replace_dataset(
        FakeAdapter(), NormalizedDataset(tracks=list(tracks), members=[Membership(t.track_id, "") for t in tracks])
    )


def spotify_track(spotify_id, name, artists, isrc_code="ISRC1", duration_s=200):
    return {
        "id": spotify_id,
        "name": name,
        "artists": [{"name": artist} for artist in artists],
        "duration_ms": duration_s * 1000,
        "external_ids": {"isrc": isrc_code} if isrc_code else {},
    }


class FakeSpotify:
    """`tracks` answers /tracks by id; `search` maps a query to its hits."""

    def __init__(self, tracks=(), search=None, error=None):
        self.tracks = {t["id"]: t for t in tracks}
        self.search = search or {}
        self.error = error
        self.calls = []

    def get(self, path, params):
        self.calls.append((path, params))
        if self.error:
            raise self.error
        if path == "tracks":
            return {"tracks": [self.tracks.get(i) for i in params["ids"].split(",")]}
        return {"tracks": {"items": self.search.get(params["q"], [])}}


def outcome(registry, track):
    state, detail = registry.status(track.track_id, config.ISRC_STAGE)
    return state, (json.loads(detail) if state == DONE else detail)


def test_known_spotify_ids_are_looked_up_in_batches(registry, monkeypatch):
    monkeypatch.setattr(isrc, "BATCH_SIZE", 2)
    tracks = [song_track(f"Song {i}", "Artist", spotify_track_id=f"sp{i}") for i in range(3)]
    add_tracks(registry, *tracks)
    client = FakeSpotify([spotify_track(f"sp{i}", f"Song {i}", ["Artist"], f"ISRC{i}") for i in range(3)])

    assert isrc.run(registry, client=client) == {DONE: 3}

    assert [path for path, _ in client.calls] == ["tracks", "tracks"]
    state, detail = outcome(registry, tracks[2])
    assert state == DONE
    assert detail == {
        "isrc": "ISRC2",
        "duration_s": 200,
        "spotify_id": "sp2",
        "spotify_title": "Song 2",
        "via": "spotify_id",
    }


def test_tracks_without_an_id_or_isrc_are_searched_by_name(registry):
    no_id = song_track("C.R.E.A.M.", "Wu-Tang Clan")
    gone = song_track("Gone", "Artist", spotify_track_id="removed")
    add_tracks(registry, no_id, gone)
    client = FakeSpotify(
        search={
            "Wu-Tang Clan c r e a m": [
                spotify_track("x", "C.R.E.A.M.", ["Other Artist"], "WRONG"),  # artist differs
                spotify_track("y", "C.R.E.A.M. (Cash Rules Everything Around Me)", ["Wu-Tang Clan"], "NOID"),
                spotify_track("z", "C.R.E.A.M.", ["Wu-Tang Clan"], "RIGHT"),
            ],
            "Artist gone": [spotify_track("g", "Gone", ["Artist"], None)],  # no ISRC on the only hit
        }
    )

    isrc.run(registry, client=client)

    state, detail = outcome(registry, no_id)
    assert (state, detail["isrc"], detail["via"]) == (DONE, "RIGHT", "search")
    assert outcome(registry, gone)[0] == UNAVAILABLE


def test_search_without_a_duration_keeps_the_strict_title_rule(registry):
    track = song_track("I Like the Way (The Kissing Game)", "Hi-Five")  # dim_sim: no duration to compare
    add_tracks(registry, track)
    client = FakeSpotify(search={"Hi-Five i like the way": [spotify_track("a", "I Like The Way", ["Hi-Five"])]})

    isrc.run(registry, client=client)

    assert outcome(registry, track)[0] == UNAVAILABLE


def test_only_tracks_still_without_audio_are_looked_up(registry):
    has_audio = song_track("Done", "Artist", spotify_track_id="sp1")
    add_tracks(registry, has_audio)
    registry.set_status(has_audio.track_id, config.AUDIO_STAGE, DONE, artifact="audio/deezer/x.mp3")
    client = FakeSpotify()

    assert isrc.run(registry, client=client) == {}
    assert client.calls == []


def test_network_errors_are_failed_and_retried_next_run(registry):
    track = song_track("Song", "Artist", spotify_track_id="sp1")
    add_tracks(registry, track)

    isrc.run(registry, client=FakeSpotify(error=urllib.error.URLError("down")))
    assert outcome(registry, track)[0] == FAILED

    isrc.run(registry, client=FakeSpotify([spotify_track("sp1", "Song", ["Artist"])]))
    assert outcome(registry, track)[0] == DONE


class FakeResponse(urllib.error.HTTPError):
    def __init__(self, code, retry_after=None):
        super().__init__("https://api.spotify.com", code, "error", {"Retry-After": retry_after}, None)


@pytest.fixture
def spotify_http(monkeypatch):
    """SpotifyClient with a fixed token and scripted HTTP replies (exceptions or bodies)."""
    replies, slept = [], []

    def http_get(url, params=None, limiter=None, headers=None):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply).encode()

    monkeypatch.setattr(isrc, "http_get", http_get)
    monkeypatch.setattr(isrc.time, "sleep", slept.append)
    client = isrc.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "_access_token", lambda: "token")
    return client, replies, slept


def test_client_waits_out_a_short_rate_limit(spotify_http):
    client, replies, slept = spotify_http
    replies += [FakeResponse(429, "3"), {"ok": True}]
    assert client.get("tracks", {}) == {"ok": True}
    assert slept == [3]


def test_client_gives_up_on_a_long_rate_limit(spotify_http):
    client, replies, slept = spotify_http
    replies.append(FakeResponse(429, str(isrc.MAX_RETRY_AFTER_S + 1)))
    with pytest.raises(urllib.error.HTTPError):
        client.get("tracks", {})
    assert slept == []
