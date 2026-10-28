import json
import urllib.error

import pytest
from stemsim.clients import spotify


@pytest.mark.parametrize(("retry_after", "waits"), [(3, True), (spotify.MAX_RETRY_AFTER_S + 1, False)])
def test_client_waits_out_only_short_rate_limits(monkeypatch, retry_after, waits):
    replies = [urllib.error.HTTPError("url", 429, "rate limited", {"Retry-After": str(retry_after)}, None), {"ok": 1}]
    slept = []

    def http_get(url, params=None, limiter=None, headers=None):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)

    monkeypatch.setattr(spotify, "http_get", http_get)
    monkeypatch.setattr(spotify.time, "sleep", slept.append)
    client = spotify.SpotifyClient("id", "secret")
    monkeypatch.setattr(client, "_access_token", lambda: "token")

    if waits:
        assert client.get("tracks", {}) == {"ok": 1}
        assert slept == [retry_after]
    else:
        with pytest.raises(urllib.error.HTTPError):
            client.get("tracks", {})
        assert slept == []
