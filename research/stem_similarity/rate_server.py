"""Local ABX rating app (SSM-8): serves rate.html, one session of prepared
triplets and their audio, and appends every answer to ratings.jsonl
(format: stemsim/ratings.py).

A session is a JSONL file of triplets, built ahead of time (the session
builder: X/A/B chosen, stems made, the split decided), so this server does no
selection and needs no torch. One row per triplet:
  id, session, design, split, x, a, b   see stemsim/ratings.py
  play     "mix", or the stem played in isolation
  audio    {clip: {source, ref}} - the full-mix file of each clip (ref relative
           to the data dir, like registry artifacts)
  stems    {clip: ref} of the played stem's file when play is a stem, else null
  window   null = the whole preview (the only choice so far)
  info     {clip: {title, artist, year}} - hidden while rating (knowing the
           artist biases the answer), revealed once the triplet is answered
  repeat_of, selection   optional: the earlier triplet a repeat swaps, and why
           the builder chose this triplet

Dictation: the note box's mic button records in the browser, which posts
16 kHz samples to /api/transcribe; dictation.py turns them into text with a
local Whisper model (loaded in the background at startup; --no-dictation
skips it). Only the text is kept.

The browser only names a triplet and a clip; file paths never leave the
session file, so no request can reach other files.

Run (or use scripts/launch_rating_server.py):
  .venv/Scripts/python rate_server.py [--session NAME_OR_PATH] [--rater ID]
Then open http://localhost:8765
"""

import argparse
import json
import logging
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dictation
from stemsim import config, ratings

HERE = Path(__file__).resolve().parent
RATINGS_PATH = HERE / "ratings.jsonl"
SESSIONS_DIR = config.DATA_DIR / "sessions"
PORT = 8765
DEFAULT_RATER = "nicolas"
AUDIO_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}


class Session:
    """One rater working through one session file."""

    def __init__(self, path, rater_id, ratings_path=RATINGS_PATH, data_dir=config.DATA_DIR):
        self.path = Path(path)
        self.rows = ratings.load_jsonl(self.path)
        if not self.rows:
            raise SystemExit(f"Session {self.path} is empty or missing")
        self.by_id = {row["id"]: row for row in self.rows}
        self.rater_id = rater_id
        self.ratings_path = Path(ratings_path)
        self.data_dir = Path(data_dir)
        self.lock = threading.Lock()

    def my_ratings(self):
        return [r for r in ratings.load_jsonl(self.ratings_path) if r["rater_id"] == self.rater_id]

    def state(self):
        mine = self.my_ratings()
        done = {r["id"] for r in mine}
        remaining = [row for row in self.rows if row["id"] not in done]
        rated_here = [r for r in mine if r["id"] in self.by_id]
        out = {
            "session": self.rows[0]["session"],
            "rater": self.rater_id,
            "rated": len(self.rows) - len(remaining),
            "total": len(self.rows),
            "consistency": consistency(mine),
            "median_s": _median_s(rated_here),
        }
        if remaining:
            out["current"] = self._public(remaining[0])
        if len(remaining) > 1:
            out["upcoming"] = self._public(remaining[1])["clips"]
        return out

    def _public(self, row):
        """What the browser may see of a triplet: no ids of songs, no design, no split."""
        return {"id": row["id"], "play": row["play"], "clips": {c: f"/audio/{row['id']}/{c}" for c in ratings.CLIPS}}

    def audio_file(self, row_id, clip):
        row = self.by_id.get(row_id)
        if row is None or clip not in ratings.CLIPS:
            return None
        ref = row["stems"][clip] if row["play"] != "mix" else row["audio"][clip]["ref"]
        path = self.data_dir / ref.replace("\\", "/")
        return path if path.is_file() else None

    def rate(self, answer):
        """Validates and appends one answer; returns the reveal for that triplet."""
        row = self.by_id.get(answer.get("id"))
        if row is None:
            raise ValueError(f"unknown triplet {answer.get('id')!r}")
        record = ratings.from_session(row, answer, self.rater_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self.lock:
            if any(r["id"] == row["id"] for r in self.my_ratings()):
                raise ValueError(f"triplet {row['id']} is already rated")
            ratings.append(self.ratings_path, record)
        return {"ok": True, "reveal": row.get("info") or {}, "choice": record["choice"], "strength": record["strength"]}


def consistency(rows):
    """(consistent, checked) over a rater's repeats: a repeat swaps A and B, so
    the same opinion flips the letter. Strength isn't compared, only direction."""
    by_id = {r["id"]: r for r in rows}
    flip = {ratings.A: ratings.B, ratings.B: ratings.A, ratings.TIE: ratings.TIE}
    pairs = [
        (r["choice"], by_id[r["repeat_of"]]["choice"])
        for r in rows
        if r.get("repeat_of") in by_id and ratings.BAD_CLIP not in (r["choice"], by_id[r["repeat_of"]]["choice"])
    ]
    return {"consistent": sum(flip[now] == before for now, before in pairs), "checked": len(pairs)}


def _median_s(rows):
    times = [r["timing"]["total_ms"] / 1000 for r in rows if (r.get("timing") or {}).get("total_ms")]
    return round(statistics.median(times)) if times else None


def find_session(name):
    """A session by path or name; with none given, the newest session in data/sessions."""
    if name:
        path = Path(name)
        return path if path.suffix == ".jsonl" else SESSIONS_DIR / f"{name}.jsonl"
    newest = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not newest:
        raise SystemExit(f"No sessions in {SESSIONS_DIR} - build one first, or pass --session PATH")
    return newest[-1]


def byte_range(header, size):
    """(start, end) inclusive for a single "bytes=a-b" Range header, None for
    no/unsupported range, or ValueError when it can't be satisfied. Browsers
    need range support to seek in audio they haven't fully loaded."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, _, last = header[len("bytes=") :].partition("-")
    if first:
        start, end = int(first), int(last) if last else size - 1
    else:
        start, end = size - int(last), size - 1  # suffix: the last N bytes
    end = min(end, size - 1)
    if start < 0 or start > end:
        raise ValueError(header)
    return start, end


def make_handler(session, voice=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype="application/json", status=200, headers=()):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers or [("Cache-Control", "no-store")]:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_audio(self, path):
            data = path.read_bytes()
            ctype = AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")
            headers = [("Accept-Ranges", "bytes"), ("Cache-Control", "private, max-age=86400")]
            try:
                span = byte_range(self.headers.get("Range"), len(data))
            except ValueError:
                return self._send(b"", ctype, 416, [("Content-Range", f"bytes */{len(data)}")])
            if span is None:
                return self._send(data, ctype, 200, headers)
            start, end = span
            headers.append(("Content-Range", f"bytes {start}-{end}/{len(data)}"))
            return self._send(data[start : end + 1], ctype, 206, headers)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send((HERE / "rate.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._send({**session.state(), "dictation": voice.status if voice else "off"})
            parts = self.path.split("/")
            if len(parts) == 4 and parts[1] == "audio":
                path = session.audio_file(parts[2], parts[3])
                if path is not None:
                    return self._send_audio(path)
            self._send({"error": "not found"}, status=404)

        def do_POST(self):
            if self.path not in ("/api/rate", "/api/transcribe"):
                return self._send({"error": "not found"}, status=404)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            try:
                if self.path == "/api/transcribe":
                    if voice is None:
                        raise RuntimeError("dictation is off (--no-dictation)")
                    return self._send({"text": voice.transcribe(body)})
                return self._send(session.rate(json.loads(body)))
            except (ValueError, TypeError) as exc:
                return self._send({"error": str(exc)}, status=400)
            except RuntimeError as exc:
                return self._send({"error": str(exc)}, status=503)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--session", help="session name (in data/sessions) or .jsonl path; default: the newest")
    parser.add_argument("--rater", default=DEFAULT_RATER)
    parser.add_argument("--ratings", default=RATINGS_PATH, help="ratings file to append to")
    parser.add_argument("--data-dir", default=config.DATA_DIR, help="where session audio refs point")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-dictation", action="store_true", help="don't load the speech-to-text model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    session = Session(find_session(args.session), args.rater, args.ratings, args.data_dir)
    voice = None if args.no_dictation else dictation.Dictation()
    state = session.state()
    print(f"Rating app on http://localhost:{args.port}  ({state['session']}: {state['rated']}/{state['total']} rated)")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session, voice)).serve_forever()


if __name__ == "__main__":
    main()
