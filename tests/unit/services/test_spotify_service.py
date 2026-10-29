import pytest
from spotipy.exceptions import SpotifyException

from overtone.services import spotify_service
from overtone.services.spotify_service import SpotifyService


class FakeSpotify:
    """Records start_playback calls; track() answers from `albums` or raises."""

    def __init__(self, albums, fail_first_play_with=None):
        self.albums = albums
        self.fail_first_play_with = fail_first_play_with
        self.plays = []
        self.lookups = 0

    def track(self, track_id, market=None):
        self.lookups += 1
        if track_id not in self.albums:
            raise SpotifyException(404, -1, "not found")
        album, playable_id = self.albums[track_id]
        return {"uri": f"spotify:track:{playable_id}", "album": {"uri": f"spotify:album:{album}"}}

    def start_playback(self, **request):
        self.plays.append(request)
        if self.fail_first_play_with and len(self.plays) == 1:
            raise SpotifyException(self.fail_first_play_with, -1, "refused")


@pytest.fixture
def service_with(monkeypatch):
    def make(client):
        monkeypatch.setattr(spotify_service.auth, "get_spotify_client", lambda: client)
        return SpotifyService()

    return make


def test_plays_the_track_inside_its_album_and_remembers_the_album(service_with):
    # "relinked": the version Spotify plays in the user's market
    client = FakeSpotify({"t1": ("a1", "relinked")})
    service = service_with(client)

    assert service.play_track("t1") == "played"
    assert service.play_track("t1") == "played"

    expected = {"context_uri": "spotify:album:a1", "offset": {"uri": "spotify:track:relinked"}}
    assert client.plays == [expected, expected]
    assert client.lookups == 1


def test_falls_back_to_the_bare_track_when_the_album_lookup_fails(service_with):
    client = FakeSpotify({})

    assert service_with(client).play_track("t1") == "played"
    assert client.plays == [{"uris": ["spotify:track:t1"]}]


def test_no_active_device_retries_the_same_request_on_the_last_device(service_with):
    client = FakeSpotify({"t1": ("a1", "t1")}, fail_first_play_with=404)

    assert service_with(client).play_track("t1", fallback_device_id="desktop") == "played"
    album_request = {"context_uri": "spotify:album:a1", "offset": {"uri": "spotify:track:t1"}}
    assert client.plays == [album_request, {"device_id": "desktop", **album_request}]
