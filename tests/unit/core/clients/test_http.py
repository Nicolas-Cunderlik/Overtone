import ssl
import urllib.error

import pytest
import truststore
from overtone_core.clients import http

URL = "https://example.test/archive.zip"  # never reached: urlopen is replaced in every test


def fail_with(monkeypatch, error):
    """Makes every request raise `error` without waiting between retries;
    returns the SSL context of each request made."""
    contexts = []

    def urlopen(request, timeout, context):
        contexts.append(context)
        raise error

    monkeypatch.setattr(http.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(http.time, "sleep", lambda seconds: None)
    return contexts


def untrusted_certificate():
    reason = ssl.SSLCertVerificationError(1, "certificate verify failed: unable to get local issuer certificate")
    return urllib.error.URLError(reason)


@pytest.mark.parametrize(
    "request_url",
    [
        lambda path: http.http_get(URL),
        lambda path: http.download(URL, path),
    ],
    ids=["http_get", "download"],
)
def test_certificates_are_checked_by_the_os(tmp_path, monkeypatch, request_url):
    contexts = fail_with(monkeypatch, untrusted_certificate())

    with pytest.raises(urllib.error.URLError):
        request_url(tmp_path / "archive.zip")

    assert all(isinstance(context, truststore.SSLContext) for context in contexts)


def test_download_gives_up_on_an_untrusted_certificate_at_once(tmp_path, monkeypatch):
    contexts = fail_with(monkeypatch, untrusted_certificate())

    with pytest.raises(urllib.error.URLError, match="certificate verify failed"):
        http.download(URL, tmp_path / "archive.zip")

    assert len(contexts) == 1  # retrying can't fix a certificate


def test_download_retries_connection_errors_and_names_the_last_one(tmp_path, monkeypatch):
    contexts = fail_with(monkeypatch, urllib.error.URLError(ConnectionResetError("reset by peer")))

    with pytest.raises(ConnectionError, match="after 3 attempts .*reset by peer"):
        http.download(URL, tmp_path / "archive.zip", attempts=3)

    assert len(contexts) == 3
    assert not (tmp_path / "archive.zip").exists()
