"""Spotify Web API client with app-only auth (client credentials), so the
app's user token is never touched."""

import base64
import json
import logging
import time
import urllib.error
import urllib.request

from .. import config
from ..secrets import get_secret
from .http import RateLimiter, http_get

logger = logging.getLogger(__name__)

AUTH_URL = "https://accounts.spotify.com/api/token"
API_URL = "https://api.spotify.com/v1"
MAX_IDS_PER_REQUEST = 50  # Spotify's limit for /tracks?ids=
REQUESTS_PER_SECOND = 5
ATTEMPTS = 3
MAX_RETRY_AFTER_S = 120  # longer back-offs fail the call rather than stall the run


class SpotifyClient:
    """Minimal Spotify Web API client: GETs with an app-only token, refreshed
    on 401; short 429 back-offs are waited out."""

    def __init__(self, client_id, client_secret):
        self._credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        self._token = None
        self._expires_at = 0.0
        self.limiter = RateLimiter(REQUESTS_PER_SECOND)

    @classmethod
    def from_env(cls):
        client_id, client_secret = get_secret("SPOTIPY_CLIENT_ID"), get_secret("SPOTIPY_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise SystemExit("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be set in .env")
        return cls(client_id, client_secret)

    def _access_token(self):
        if self._token is None or time.monotonic() >= self._expires_at:
            request = urllib.request.Request(
                AUTH_URL,
                data=b"grant_type=client_credentials",
                headers={
                    "Authorization": f"Basic {self._credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": config.USER_AGENT,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=config.HTTP_TIMEOUT_S) as response:
                    body = json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401):  # invalid_client: no request can succeed, so stop the run
                    raise SystemExit(f"Spotify rejected SPOTIPY_CLIENT_ID/SECRET (HTTP {exc.code})") from exc
                raise
            self._token = body["access_token"]
            self._expires_at = time.monotonic() + body.get("expires_in", 3600) - 60
        return self._token

    def get(self, path, params):
        """GET `API_URL/path` with `params`; returns the decoded JSON body."""
        for attempt in range(ATTEMPTS):
            try:
                headers = {"Authorization": f"Bearer {self._access_token()}"}
                return json.loads(http_get(f"{API_URL}/{path}", params, self.limiter, headers=headers))
            except urllib.error.HTTPError as exc:
                if attempt == ATTEMPTS - 1 or exc.code not in (401, 429):
                    raise
                if exc.code == 401:
                    self._token = None  # expired early: fetch a new one
                    continue
                wait = _retry_after(exc)
                if wait > MAX_RETRY_AFTER_S:
                    raise
                logger.info("Spotify rate limit, waiting %d s", wait)
                time.sleep(wait)
        raise AssertionError("unreachable")


def _retry_after(exc):
    try:
        return int(exc.headers.get("Retry-After", 5))
    except (TypeError, ValueError):
        return 5
