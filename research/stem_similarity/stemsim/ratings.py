"""The ABX rating format: what one answer records, for any rater and any catalog size.

ratings.jsonl is append-only: rate_server.py adds one row per answer, all in
the format `check` describes. One row = one triplet (X, A, B), the rater, the
session, the split, exactly what audio was heard, and the answer.

Answer: choice a / b / tie, plus strength 1 ("slightly") or 2 ("clearly") for
a or b - a five-level scale. Strength is stored raw. How (or whether) it enters
fitting is an experiment decision, not a format one. `strength_source` says
who decided it: "rater", or "rater_confirmed" for a value inferred from the
rater's note and then checked by them.

Bad clip: choice "bad_clip" names the unusable clip(s) and why. It records no
judgment, and session building skips flagged tracks from then on.

Tags: five, four of them one-to-one with the Demucs stems the model weighs
(instruments = the "other" stem), plus one for whole-mix reasons. Finer reasons
(groove, era) are left to the free-text note and LLM enrichment (SSM-7).

Splits, decided per song so they survive catalog growth and rater count:
  A song is a validation song when a salted hash of its track id lands in the
  first VALIDATION_PERCENT of 100 buckets, so every new song gets a permanent
  side without a list to maintain. A triplet is validation when all three songs
  are, fit when none is, and is never generated otherwise. The split is also
  written on each row when its session is built, because `song:` ids follow the
  dedup rules, and a rule change could move a song.
  Pilot songs are always validation: the pilot ratings are the held-out set.

The pilot ratings (session "pilot", 21 rows from the first app) were converted
to this format on 2026-10-25. Their seven tags were mapped to the five
(drums/percussion -> drums, instruments/melody -> instruments, atmosphere/space
and era/production -> production/atmosphere, groove/tempo -> none; originals
kept in `tags_v1`). Their strengths and the one bad clip (rating 5, a
dialogue-only clip answered "can't tell") were inferred from the notes by an
LLM and confirmed by the rater (`inferred` keeps the evidence). Four pilot
ratings have no note, so no strength - the only rows allowed to lack one.
"""

import hashlib
import json

A, B, TIE, BAD_CLIP = "a", "b", "tie", "bad_clip"
CHOICES = (A, B, TIE, BAD_CLIP)
SLIGHTLY, CLEARLY = 1, 2
STRENGTHS = (SLIGHTLY, CLEARLY)
STRENGTH_SOURCES = ("rater", "rater_confirmed")
CLIPS = ("x", "a", "b")
BAD_CLIP_REASONS = ("silence", "dialogue", "wrong_version", "other")

TAGS = ("drums", "bass", "vocals", "instruments", "production/atmosphere")
TAG_STEM = {"drums": "drums", "bass": "bass", "vocals": "vocals", "instruments": "other"}

# Triplet designs (see the session builder)
CONTESTED = "contested"  # the models split most evenly on it - informative for fitting
NEIGHBOURHOOD = "neighbourhood"  # A and B from X's near neighbours, picked blind to the models
SHARED = "shared"  # also judged elsewhere (a public dataset, other raters) - agreement
REPEAT = "repeat"  # an earlier triplet with A/B swapped - self-consistency
DESIGNS = (CONTESTED, NEIGHBOURHOOD, SHARED, REPEAT)
PILOT_DESIGNS = ("stem_disagree", "near_far", "random", "stem_only")  # the pilot queue's (triplets.py)

FIT, VALIDATION = "fit", "validation"
SPLITS = (FIT, VALIDATION)
VALIDATION_PERCENT = 20
_SPLIT_SALT = "overtone-split-v1:"  # independent of the sha1 used for file paths

PLAYS = ("mix", "drums", "bass", "vocals", "other")
VERSION = 2


def song_split(track_id, always_validation=frozenset()):
    """FIT or VALIDATION for one song - stable for the life of its track id."""
    if track_id in always_validation:
        return VALIDATION
    digest = hashlib.sha1((_SPLIT_SALT + track_id).encode(), usedforsecurity=False).hexdigest()
    return VALIDATION if int(digest[:8], 16) % 100 < VALIDATION_PERCENT else FIT


def triplet_split(track_ids, always_validation=frozenset()):
    """The split all of a triplet's songs share, or None when they straddle it."""
    sides = {song_split(t, always_validation) for t in track_ids}
    return sides.pop() if len(sides) == 1 else None


def check(row):
    """Problems with a rating row, as a list of messages (empty = valid)."""
    problems = []
    for key in ("id", "rater_id", "session", "x", "a", "b"):
        if not isinstance(row.get(key), str) or not row[key]:
            problems.append(f"{key} must be a non-empty string")
    if row.get("v") != VERSION:
        problems.append(f"v must be {VERSION}")
    design = row.get("design")
    if design not in DESIGNS + PILOT_DESIGNS:
        problems.append(f"design must be one of {DESIGNS}")
    if row.get("split") not in SPLITS:
        problems.append(f"split must be one of {SPLITS}")
    if row.get("play") not in PLAYS:
        problems.append(f"play must be one of {PLAYS}")
    if design == REPEAT and not row.get("repeat_of"):
        problems.append("a repeat needs repeat_of")
    audio = row.get("audio") or {}
    if not all(isinstance(audio.get(clip), dict) and audio[clip].get("source") for clip in CLIPS):
        problems.append(f"audio needs a source for each of {CLIPS} (what was actually heard)")
    window = row.get("window")
    if window is not None and not (isinstance(window, list) and len(window) == 2 and window[0] < window[1]):
        problems.append("window must be null (the whole preview) or [start, end] seconds")

    choice, strength = row.get("choice"), row.get("strength")
    if choice not in CHOICES:
        problems.append(f"choice must be one of {CHOICES}")
    elif choice in (A, B):
        if strength is None and design not in PILOT_DESIGNS:
            problems.append(f"choosing a or b needs strength in {STRENGTHS}")
        elif strength is not None and strength not in STRENGTHS:
            problems.append(f"strength must be one of {STRENGTHS}")
    elif strength is not None:
        problems.append(f"{choice} takes no strength")
    if (strength is None) != (row.get("strength_source") is None) or row.get("strength_source") not in (
        None,
        *STRENGTH_SOURCES,
    ):
        problems.append(f"strength_source must be one of {STRENGTH_SOURCES} exactly when there is a strength")

    bad = row.get("bad_clip")
    if choice == BAD_CLIP:
        if not isinstance(bad, dict) or not bad.get("clips") or not set(bad["clips"]) <= set(CLIPS):
            problems.append(f"bad_clip needs clips, a non-empty subset of {CLIPS}")
        elif bad.get("reason") not in BAD_CLIP_REASONS:
            problems.append(f"bad_clip reason must be one of {BAD_CLIP_REASONS}")
    elif bad is not None:
        problems.append("bad_clip is only set when choice is bad_clip")

    unknown = set(row.get("tags", [])) - set(TAGS)
    if unknown:
        problems.append(f"unknown tags {sorted(unknown)}")
    return problems


# What a session row (one prepared triplet) passes on to the rating made from it.
# Session rows also carry `info` (titles, revealed after answering), which isn't kept.
SESSION_FIELDS = ("id", "session", "design", "split", "x", "a", "b", "play", "audio", "stems", "window")
OPTIONAL_SESSION_FIELDS = ("repeat_of", "selection")


def from_session(session_row, answer, rater_id, rated_at):
    """The rating row for one answer to one session row. Not yet checked."""
    strength = answer.get("strength")
    return {
        "v": VERSION,
        "rater_id": rater_id,
        **{key: session_row.get(key) for key in SESSION_FIELDS},
        **{key: session_row[key] for key in OPTIONAL_SESSION_FIELDS if session_row.get(key) is not None},
        "choice": answer.get("choice"),
        "strength": strength,
        "strength_source": "rater" if strength is not None else None,
        "tags": list(answer.get("tags") or []),
        "note": (answer.get("note") or "").strip(),
        "bad_clip": answer.get("bad_clip"),
        "timing": answer.get("timing") or {},
        "rated_at": rated_at,
    }


def load_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def append(path, row):
    """Appends one valid row; raises ValueError (with every problem) otherwise."""
    problems = check(row)
    if problems:
        raise ValueError("; ".join(problems))
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row) + "\n")


def flagged_tracks(rows):
    """{track id: reason} for every clip a rater marked bad."""
    flagged = {}
    for row in rows:
        if row["choice"] == BAD_CLIP:
            for clip in row["bad_clip"]["clips"]:
                flagged[row[clip]] = row["bad_clip"]["reason"]
    return flagged
