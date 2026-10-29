import json

import pytest
from overtone_core.clients.deezer import DeezerError
from stemsim import config
from stemsim.registry import DONE, FAILED, UNAVAILABLE
from stemsim.schema import Membership, NormalizedDataset, song_track
from stemsim.stages.audio import DeezerResolver


class FakeAdapter:
    name, target, metric, license, url = "catalog", "seeds", None, "-", "-"


class FakeDeezer:
    """find_preview answers `outcome` (a match, None, or an exception to raise) and records its arguments."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []
        self.downloads = []

    def find_preview(self, title, artist, isrc=None, duration_s=None):
        self.calls.append((title, artist, isrc, duration_s))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def download_preview(self, hit, path):
        self.downloads.append(path)


@pytest.fixture
def resolve(registry):
    """resolve(client, isrc_status, **meta) -> outcome, for one song "Song" by "Artist"."""

    def run(client, isrc_status, **meta):
        track = song_track("Song", "Artist", **meta)
        registry.replace_dataset(
            FakeAdapter(), NormalizedDataset(tracks=[track], members=[Membership(track.track_id, "1")])
        )
        if isrc_status:
            registry.set_status(track.track_id, config.ISRC_STAGE, *isrc_status)
        return DeezerResolver(registry, client=client).resolve(registry.dataset_tracks("catalog")[0])

    return run


@pytest.mark.parametrize("isrc_status", [None, (FAILED, "network")])
def test_waits_for_the_isrc_stage(resolve, isrc_status):
    client = FakeDeezer(None)

    assert resolve(client, isrc_status) is None
    assert client.calls == []


@pytest.mark.parametrize(
    ("own_duration", "isrc_detail", "expected_call"),
    [
        (200, {"isrc": "X1", "duration_s": 190}, ("Song", "Artist", "X1", 200)),  # own duration wins
        (None, {"isrc": "X1", "duration_s": 190}, ("Song", "Artist", "X1", 190)),  # Spotify's stands in
        (None, {}, ("Song", "Artist", None, None)),
    ],
)
def test_passes_the_isrc_and_best_known_duration(resolve, own_duration, isrc_detail, expected_call):
    client = FakeDeezer(None)
    meta = {"duration_s": own_duration} if own_duration else {}

    resolve(client, (DONE, json.dumps(isrc_detail)), **meta)

    assert client.calls == [expected_call]


def test_match_is_downloaded_and_recorded(resolve, registry):
    client = FakeDeezer(({"id": 7, "title": "Song (Remastered)"}, "isrc"))

    state, detail, artifact = resolve(client, (DONE, json.dumps({"isrc": "X1"})))

    assert state == DONE
    assert json.loads(detail) == {"deezer_id": 7, "deezer_title": "Song (Remastered)", "matched_by": "isrc"}
    assert client.downloads == [registry.data_dir / artifact]
    assert artifact.endswith(".mp3")


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (None, (UNAVAILABLE, "no matching Deezer preview")),
        (DeezerError("isrc lookup failed: quota"), (FAILED, "isrc lookup failed: quota")),  # retried next run
    ],
)
def test_no_match_is_unavailable_and_transient_errors_fail(resolve, outcome, expected):
    state, detail, artifact = resolve(FakeDeezer(outcome), (DONE, json.dumps({"isrc": "X1"})))

    assert (state, detail, artifact) == (*expected, None)
