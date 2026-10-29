"""The ABX rating format: what one answer records, for any rater and any catalog size.

ratings.jsonl is append-only: the rating server (server.py) adds one row per answer, in the
format `check` describes. One row = one triplet (X, A, B), the rater, the
session, the split, exactly what audio was heard, and the answer.

Answer: choice a / b / tie, plus strength 1 ("slightly") or 2 ("clearly") for
a or b: a five-level scale. Strength is stored raw; how it enters fitting is an
experiment decision. `strength_source` says who decided it: "rater", or
"rater_confirmed" for a value inferred from the note and checked by the rater.

Bad clip: choice "bad_clip" names the unusable clip(s) and why. It records no
judgment, and session building skips flagged tracks from then on.

Tags: one per Demucs stem the model weighs (instruments = the "other" stem),
plus one for whole-mix reasons. Finer reasons (groove, era) go in the note,
for LLM enrichment.

Splits are per song, so they survive catalog growth and new raters: a song is
validation when a salted hash of its track id lands in the first
VALIDATION_PERCENT of 100 buckets, so every new song gets a permanent side with
no list to maintain. A triplet is validation when all three songs are, fit when
none is, and is never generated otherwise. Each row also records its split,
because `song:` ids follow the dedup rules and a rule change could move a song.
Pilot songs are always validation: the pilot ratings are the held-out set.

Pilot rows (design and session "pilot", from the first app's fixed queue) have
the same fields: `selection.rule` is that queue's own design, and the audio is
the pilot preview's seconds 10-20.
"""

import hashlib
import json

from overtone_core.ml.stems import CHANNELS

from .. import config

A, B, TIE, BAD_CLIP = "a", "b", "tie", "bad_clip"
CHOICES = (A, B, TIE, BAD_CLIP)
SLIGHTLY, CLEARLY = 1, 2
STRENGTHS = (SLIGHTLY, CLEARLY)
STRENGTH_SOURCES = ("rater", "rater_confirmed")
CLIPS = ("x", "a", "b")
BAD_CLIP_REASONS = ("silence", "dialogue", "wrong_version", "other")

TAGS = ("drums", "bass", "vocals", "instruments", "production/atmosphere")
TAG_STEM = {"drums": "drums", "bass": "bass", "vocals": "vocals", "instruments": "other"}

# Triplet designs (sessions.py builds them)
CONTESTED = "contested"  # the models split most evenly on it: informative for fitting
NEIGHBOURHOOD = "neighbourhood"  # a random pair from X's near neighbours
SHARED = "shared"  # also judged elsewhere (a public dataset): you vs the crowd
REPEAT = "repeat"  # an earlier triplet with A/B swapped: self-consistency
PILOT = config.PILOT  # the first app's fixed queue
DESIGNS = (CONTESTED, NEIGHBOURHOOD, SHARED, REPEAT, PILOT)

FIT, VALIDATION = "fit", "validation"
SPLITS = (FIT, VALIDATION)
VALIDATION_PERCENT = 20
_SPLIT_SALT = "overtone-split-v1:"  # so the split is independent of the sha1 file paths

PLAYS = CHANNELS  # MIX plays the full mix, a stem plays that stem alone
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
    missing, unknown = set(FIELDS) - set(row), set(row) - set(FIELDS)
    if missing or unknown:
        problems.append(f"fields must be exactly FIELDS (missing {sorted(missing)}, unknown {sorted(unknown)})")
    for key in ("id", "rater_id", "session", "x", "a", "b"):
        if not isinstance(row.get(key), str) or not row[key]:
            problems.append(f"{key} must be a non-empty string")
    if row.get("v") != VERSION:
        problems.append(f"v must be {VERSION}")
    design = row.get("design")
    if design not in DESIGNS:
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
        if strength not in STRENGTHS:
            problems.append(f"choosing a or b needs strength in {STRENGTHS}")
    elif strength is not None:
        problems.append(f"{choice} takes no strength")
    source = row.get("strength_source")
    if source is not None and source not in STRENGTH_SOURCES:
        problems.append(f"strength_source must be one of {STRENGTH_SOURCES}")
    elif (strength is None) != (source is None):
        problems.append("strength_source is set exactly when there is a strength")

    bad = row.get("bad_clip")
    if choice == BAD_CLIP:
        if not isinstance(bad, dict) or not bad.get("clips") or not set(bad["clips"]) <= set(CLIPS):
            problems.append(f"bad_clip needs clips, a non-empty subset of {CLIPS}")
        elif bad.get("reason") not in BAD_CLIP_REASONS:
            problems.append(f"bad_clip reason must be one of {BAD_CLIP_REASONS}")
    elif bad is not None:
        problems.append("bad_clip is only set when choice is bad_clip")

    unknown_tags = set(row.get("tags") or []) - set(TAGS)
    if unknown_tags:
        problems.append(f"unknown tags {sorted(unknown_tags)}")
    return problems


# A session (data/sessions/<rater>-<nnn>.jsonl, built by sessions.py, served by
# server.py) is one row per prepared triplet:
#   id, session, design, split, x, a, b   as in a rating row
#   play       MIX, or the stem played in isolation
#   audio      {clip: {source, ref}}: each clip's full-mix file, ref relative to the data dir
#   stems      {clip: ref} of the played stem's file when play is a stem, else null
#   window     null = the whole preview
#   info       {clip: {title, artist, year}}: hidden while rating, revealed after the answer; not kept
#   repeat_of  the earlier triplet a repeat swaps, else null
#   selection  why the builder chose this triplet ({rule, ...}; rule is the design, or the pilot queue's)
# A rating copies these fields from its session row:
SESSION_FIELDS = (
    "id",
    "session",
    "design",
    "split",
    "x",
    "a",
    "b",
    "play",
    "audio",
    "stems",
    "window",
    "repeat_of",
    "selection",
)
ANSWER_FIELDS = ("choice", "strength", "strength_source", "tags", "note", "bad_clip", "timing", "rated_at")
FIELDS = ("v", "rater_id", *SESSION_FIELDS, *ANSWER_FIELDS)  # every rating row has exactly these, in this order


def from_session(session_row, answer, rater_id, rated_at):
    """The rating row for one answer to one session row. Not yet checked."""
    strength = answer.get("strength")
    return {
        "v": VERSION,
        "rater_id": rater_id,
        **{key: session_row.get(key) for key in SESSION_FIELDS},
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
