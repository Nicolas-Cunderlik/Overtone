import json
import urllib.error

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


def spotify_track(spotify_id, name, artist="Artist", isrc_code="ISRC1"):
    return {
        "id": spotify_id,
        "name": name,
        "artists": [{"name": artist}],
        "duration_ms": 200_000,
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
        self.calls.append(path)
        if self.error:
            raise self.error
        if path == "tracks":
            return {"tracks": [self.tracks.get(i) for i in params["ids"].split(",")]}
        return {"tracks": {"items": self.search.get(params["q"], [])}}


def test_looks_up_ids_in_batches_and_searches_the_rest(registry, monkeypatch):
    monkeypatch.setattr(isrc, "MAX_IDS_PER_REQUEST", 2)
    by_id = [song_track(f"Song {i}", "Artist", spotify_track_id=f"sp{i}") for i in range(3)]
    no_isrc = song_track("Gone", "Artist", spotify_track_id="gone")
    no_id = song_track("Hit", "Artist")
    has_audio = song_track("Done", "Artist", spotify_track_id="sp9")
    add_tracks(registry, *by_id, no_isrc, no_id, has_audio)
    registry.set_status(has_audio.track_id, config.AUDIO_STAGE, DONE, artifact="x.mp3")
    client = FakeSpotify(
        tracks=[
            *(spotify_track(f"sp{i}", f"Song {i}", isrc_code=f"ISRC{i}") for i in range(3)),
            spotify_track("gone", "Gone", isrc_code=None),
        ],
        search={
            "Artist hit": [
                spotify_track("x", "Hit", artist="Someone Else"),
                spotify_track("y", "Hit", isrc_code="RIGHT"),
            ],
            "Artist gone": [],
        },
    )

    assert isrc.run(registry, client=client) == {DONE: 4, UNAVAILABLE: 1}

    assert client.calls == ["tracks", "tracks", "search", "search"]  # has_audio never looked up
    assert json.loads(registry.status(by_id[2].track_id, config.ISRC_STAGE)[1]) == {
        "isrc": "ISRC2",
        "duration_s": 200,
        "spotify_id": "sp2",
        "spotify_title": "Song 2",
        "via": "spotify_id",
    }
    assert json.loads(registry.status(no_id.track_id, config.ISRC_STAGE)[1])["isrc"] == "RIGHT"
    assert registry.status(no_isrc.track_id, config.ISRC_STAGE)[0] == UNAVAILABLE


def test_network_errors_fail_and_are_retried_next_run(registry):
    track = song_track("Song", "Artist", spotify_track_id="sp1")
    add_tracks(registry, track)

    isrc.run(registry, client=FakeSpotify(error=urllib.error.URLError("down")))
    assert registry.status(track.track_id, config.ISRC_STAGE)[0] == FAILED

    isrc.run(registry, client=FakeSpotify([spotify_track("sp1", "Song")]))
    assert registry.status(track.track_id, config.ISRC_STAGE)[0] == DONE


def test_skips_tracks_only_no_audio_datasets_include_unless_named(registry, add_dataset, monkeypatch):
    monkeypatch.setattr(config, "NO_AUDIO_DATASETS", ("behavioral",))
    skipped = song_track("Skipped", "Artist", spotify_track_id="sp1")
    shared = song_track("Shared", "Artist", spotify_track_id="sp2")
    add_dataset("behavioral", [skipped, shared])
    add_dataset("catalog", [shared])
    client = FakeSpotify([spotify_track("sp1", "Skipped"), spotify_track("sp2", "Shared")])

    assert isrc.run(registry, client=client) == {DONE: 1}
    assert registry.status(skipped.track_id, config.ISRC_STAGE) is None
    assert isrc.run(registry, dataset="behavioral", client=client) == {DONE: 1}  # naming the dataset opts in
    assert registry.status(skipped.track_id, config.ISRC_STAGE)[0] == DONE
