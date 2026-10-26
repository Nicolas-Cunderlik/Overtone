"""Listeners' own ABX ratings (rate_server.py -> ratings.jsonl; format in ratings.py).

So far one listener, the project author: a music producer's judgments on
"which sounds more like X?", with reason tags and free-text notes ("same sort
of bass", "similar thumpy kick"). Few, but the only judgments here that are
per-stem, explained, and from someone who listens for sound selection rather
than genre. Every row carries its rater, so more raters change nothing here.

Normalization:
  - choice a -> triplet (X, A, B); choice b -> (X, B, A). Ties and bad clips
    give no triplet.
  - facet: "overall" for full-mix triplets; the stem's name ("drums",
    "bass", ...) for isolated-stem ones.
  - meta: rater, session, split (fit / validation), strength (1 slightly,
    2 clearly) and who decided it, the five reason tags, the note, the design
    and timing - kept for analysis and for fitting choices made later.

Tracks are what the listener actually heard (each row records it):
  - pilot ratings: the pilot's Deezer previews (audio/<spotify id>.mp3),
    seconds 10-20. Some pilot matches were a different version than the
    catalog song (e.g. a remix) - the judgment is still valid for the audio
    heard, so those tracks stay separate ("pilot:" ids) and are only linked to
    the catalog song in meta, flagged when the version differs.
  - later ratings: registry tracks, played from their audio-stage file;
    window null = the whole preview, which the embed stage's full-audio
    window already covers.
"""

import csv
import shutil

from .. import config, ratings
from ..schema import (
    OVERALL,
    PERCEPTUAL,
    TRIPLET,
    TRIPLET_AGREEMENT,
    Judgment,
    Membership,
    NormalizedDataset,
    Track,
    song_track_id,
)
from .base import DatasetAdapter

RATINGS = "ratings.jsonl"
PILOT_SONGS = "pilot_songs.csv"
PILOT = "pilot"  # audio source of the pilot's own previews (stages/audio.PilotResolver)


class ListenerAbx(DatasetAdapter):
    name = "listener_abx"
    target = PERCEPTUAL
    metric = TRIPLET_AGREEMENT
    license = "Raters' own ratings"
    url = "research/stem_similarity/rate_server.py"

    def fetch(self, registry, **_):
        """Snapshots the live ratings file, so normalize reads only raw_dir."""
        raw = registry.raw_dir(self.name)
        for name in (RATINGS, PILOT_SONGS):
            shutil.copyfile(config.ROOT / name, raw / name)

    def normalize(self, registry):
        raw = registry.raw_dir(self.name)
        out = NormalizedDataset()
        heard = {}  # track id -> (audio source, audio ref, window)
        for r in ratings.load_jsonl(raw / RATINGS):
            if r["choice"] not in (ratings.A, ratings.B):
                continue
            for clip in ratings.CLIPS:
                _add_heard(heard, r[clip], r["audio"][clip], r.get("window"))
            closer, farther = (r["a"], r["b"]) if r["choice"] == ratings.A else (r["b"], r["a"])
            out.judgments.append(
                Judgment(
                    TRIPLET,
                    r["x"],
                    closer,
                    farther,
                    facet=OVERALL if r["play"] == "mix" else r["play"],
                    meta={
                        "rater_id": r["rater_id"],
                        "session": r["session"],
                        "split": r["split"],
                        "design": r["design"],
                        "strength": r.get("strength"),
                        "strength_source": r.get("strength_source"),
                        "reasons": r.get("tags", []),
                        "note": r.get("note", ""),
                        "rating_id": r["id"],
                        "rated_at": r.get("rated_at"),
                        "listen_ms": (r.get("timing") or {}).get("total_ms"),
                    },
                )
            )

        pilot_songs = _pilot_songs(raw) if any(source == PILOT for source, _, _ in heard.values()) else {}
        for track_id, (source, ref, window) in sorted(heard.items()):
            start, end = window or (None, None)
            if source == PILOT:
                out.tracks.append(_pilot_track(track_id, ref, pilot_songs))
                out.members.append(Membership(track_id, track_id.split(":", 1)[1], start, end))
            else:
                out.tracks.append(Track(track_id, source))  # already registered: merging keeps its fields
                out.members.append(Membership(track_id, track_id, start, end))
        return out


def _add_heard(heard, track_id, audio, window):
    """A membership holds one window per track, so every rating of a track
    must have heard the same audio, over the same part of it."""
    entry = (audio["source"], audio.get("ref", ""), tuple(window) if window else None)
    if heard.setdefault(track_id, entry) != entry:
        raise ValueError(
            f"{track_id} was rated on different audio ({heard[track_id]} vs {entry}); "
            "one dataset can hold only one window per track"
        )


def _pilot_songs(raw):
    with open(raw / PILOT_SONGS, newline="", encoding="utf-8") as f:
        return {row["track_id"]: row for row in csv.DictReader(f)}


def _pilot_track(track_id, ref, songs):
    from ..stages.audio import same_recording  # stages import datasets; avoid a cycle at import time

    song = songs[track_id.split(":", 1)[1]]
    return Track(
        track_id,
        PILOT,
        audio_ref=ref,
        title=song["track_name"],
        artist=song["track_artist"],
        meta={
            "catalog_track": song_track_id(song["track_name"], song["track_artist"]),
            "deezer_title": song["deezer_title"],
            "version_differs": not same_recording(song["track_name"], song["deezer_title"]),
        },
    )
