"""Session builder: prepares one rating session (which triplets, which
audio) so the rating server (server.py) does no selection and needs no torch.

    python -m stemsim session [--rater nicolas] [--seed N] [--dry-run]

A session (SESSION_PLAN):
  3 contested      fit songs: the committee's most even splits (committee.py), for fitting
  3 neighbourhood  fit songs: a random pair from X's nearest neighbours, for fitting
  3 neighbourhood  validation songs, same rule: the held-out set
  1 shared         a public dataset's triplet (DIM-SIM's clean subset or MagnaTagATune) on fit songs: you vs the crowd
  1 repeat         one of this rater's earlier triplets with A/B swapped: self-consistency (none in a first session)
Rows are shuffled and A/B order is random, so position says nothing.

Every triplet:
  - catalog songs with audio and embeddings (shared: the dataset's own tracks, with audio)
  - all three songs on one side of the split (ratings.song_split); catalog copies of pilot songs are validation
  - X's artist differs from A's and B's, and A's from B's
  - no song twice in a session, no triplet the rater has already been given (same X, same pair)
  - no track any rater flagged as a bad clip
Neighbourhood = the union of X's NEIGHBOURS nearest songs (same split) under mix only, fitted stems and
the app model: the choices the app actually faces. Contested = CONTESTED_ANCHORS random anchors x
CONTESTED_PAIRS random neighbourhood pairs each, scored by the committee, most even splits first.

Every triplet plays the full preview of the full mix (window null). server.py can also play
isolated stems, but no design asks for them yet.

Writes config.SESSIONS_DIR/<rater>-<nnn>.jsonl (row format: ratings.py, above SESSION_FIELDS).
"""

import json
import logging
import random

import numpy as np
from overtone_core.ml.stems import MIX

from .. import config
from ..datasets.spotify_catalog import SpotifyCatalog
from ..registry import DONE
from ..schema import TRIPLET
from . import ratings
from .committee import disagreement

logger = logging.getLogger(__name__)

SESSION_PLAN = (  # (design, split, triplets)
    (ratings.CONTESTED, ratings.FIT, 3),
    (ratings.NEIGHBOURHOOD, ratings.FIT, 3),
    (ratings.NEIGHBOURHOOD, ratings.VALIDATION, 3),
    (ratings.SHARED, ratings.FIT, 1),
    (ratings.REPEAT, None, 1),
)
NEIGHBOURS = 50
CONTESTED_ANCHORS = 300
CONTESTED_PAIRS = 20
SHARED_SOURCES = ("dim_sim", "magnatagatune")


def seen_key(x, a, b):
    return x, frozenset((a, b))


class Picker:
    """Picks triplets from a pool under the session rules, tracking what's used."""

    def __init__(self, pool, committee, split_of, rng, seen=(), flagged=()):
        self.pool, self.committee, self.rng = pool, committee, rng
        self.seen = set(seen)
        self.used = set()  # track ids that can't appear (again) in this session
        self.free_rows = np.ones(len(pool.ids), bool)  # the same, per pool row
        self.exclude(flagged)
        splits = np.array([split_of(t) for t in pool.ids])
        self.side = {s: np.flatnonzero(splits == s) for s in ratings.SPLITS}

    def exclude(self, track_ids):
        """Keeps these tracks out of the rest of the session."""
        self.used.update(track_ids)
        self.free_rows[[self.pool.row[t] for t in track_ids if t in self.pool.row]] = False

    def free(self, rows):
        return rows[self.free_rows[rows]]

    def neighbourhood(self, x, split, scores):
        """X's candidates (same split, other artists, unused): the union of each model's top NEIGHBOURS."""
        candidates = self.free(self.side[split])
        candidates = candidates[(candidates != x) & (self.pool.artists[candidates] != self.pool.artists[x])]
        if len(candidates) < 2:
            return {}, candidates
        top = self.committee.top_k(scores, candidates, NEIGHBOURS)
        return top, np.unique(np.concatenate(list(top.values())))

    def pairs(self, x, union, n):
        """Up to n random (a, b) rows from the union, different artists, not seen with this X."""
        ids, out = self.pool.ids, []
        for _ in range(n * 5):
            if len(out) == n or len(union) < 2:
                break
            a, b = (int(v) for v in self.rng.choice(union, 2, replace=False))
            if self.pool.artists[a] != self.pool.artists[b] and seen_key(ids[x], ids[a], ids[b]) not in self.seen:
                out.append((a, b))
        return out

    def take(self, x, a, b):
        triplet = self.pool.ids[x], self.pool.ids[a], self.pool.ids[b]
        self.exclude(triplet)
        self.seen.add(seen_key(*triplet))
        return triplet

    def neighbourhood_triplets(self, split, n):
        out = []
        for _ in range(n * 20):
            anchors = self.free(self.side[split])
            if len(out) == n or not len(anchors):
                break
            x = int(self.rng.choice(anchors))
            top, union = self.neighbourhood(x, split, self.committee.scores(self.pool, x))
            for a, b in self.pairs(x, union, 1):
                in_top = {"a": _models_with(top, a), "b": _models_with(top, b)}
                selection = {"rule": ratings.NEIGHBOURHOOD, "k": NEIGHBOURS, "in_top": in_top}
                out.append((*self.take(x, a, b), selection))
        return out

    def contested_triplets(self, split, n):
        candidates = []  # (disagreement, x, a, b, share, families, j): families[f][j] is this triplet's
        anchors = self.free(self.side[split])
        for x in self.rng.choice(anchors, min(CONTESTED_ANCHORS, len(anchors)), replace=False):
            x = int(x)
            scores = self.committee.scores(self.pool, x)
            _, union = self.neighbourhood(x, split, scores)
            pairs = self.pairs(x, union, CONTESTED_PAIRS)
            if not pairs:
                continue
            a, b = np.array(pairs).T
            share, families = self.committee.split(scores, a, b)
            candidates += [
                (d, x, int(a[j]), int(b[j]), float(share[j]), families, j) for j, d in enumerate(disagreement(share))
            ]
        out = []
        for d, x, a, b, share, families, j in sorted(candidates, key=lambda c: -c[0]):
            if len(out) == n:
                break
            if self.free_rows[[x, a, b]].all():
                out.append((*self.take(x, a, b), _contested_selection(d, share, families, j)))
        logger.info(
            "contested: %d candidates, picked disagreement %s", len(candidates), [o[3]["disagreement"] for o in out]
        )
        return out


def _models_with(top, row):
    """The models that have `row` among their top NEIGHBOURS."""
    return [model for model, rows in top.items() if row in rows]


def _contested_selection(d, share, families, j):
    return {
        "rule": ratings.CONTESTED,
        "disagreement": round(float(d), 3),
        "share_a": round(share, 3),
        "families": {f: None if np.isnan(v[j]) else round(float(v[j]), 3) for f, v in families.items()},
    }


def pick_shared(candidates, split_of, used, seen, rng):
    """One public-dataset triplet: (x, a, b, selection), or None. `candidates` are
    (dataset, judgment id, anchor, closer, farther, agreement, votes)."""
    usable = [
        c
        for c in candidates
        if all(split_of(t) == ratings.FIT and t not in used for t in c[2:5]) and seen_key(*c[2:5]) not in seen
    ]
    if not usable:
        return None
    dataset, judgment_id, x, closer, farther, agreement, votes = rng.choice(usable)
    a, b = (closer, farther) if rng.random() < 0.5 else (farther, closer)
    selection = {
        "rule": ratings.SHARED,
        "source": dataset,
        "judgment_id": judgment_id,
        "crowd_closer": ratings.A if a == closer else ratings.B,
        "agreement": agreement,
        "votes": votes,
    }
    return x, a, b, selection


def pick_repeat(rated, rng):
    """One of a rater's earlier session ratings, to repeat with A/B swapped, or None."""
    repeated = {r["repeat_of"] for r in rated}
    usable = [
        r
        for r in rated
        if r["session"] != config.PILOT
        and r["design"] != ratings.REPEAT
        and r["choice"] != ratings.BAD_CLIP
        and r["id"] not in repeated
    ]
    return rng.choice(usable) if usable else None


def repeat_row(original):
    swap = {"a": "b", "b": "a", "x": "x"}
    return {
        "design": ratings.REPEAT,
        "split": original["split"],
        "x": original["x"],
        "a": original["b"],
        "b": original["a"],
        "play": original["play"],
        "audio": {clip: original["audio"][swap[clip]] for clip in ratings.CLIPS},
        "stems": {clip: original["stems"][swap[clip]] for clip in ratings.CLIPS} if original["stems"] else None,
        "window": original["window"],
        "repeat_of": original["id"],
        "selection": {"rule": ratings.REPEAT, "repeat_of": original["id"]},
    }


def triplet_row(design, split, x, a, b, selection, audio):
    return {
        "design": design,
        "split": split,
        "x": x,
        "a": a,
        "b": b,
        "play": MIX,
        "audio": {clip: audio[t] for clip, t in zip(ratings.CLIPS, (x, a, b), strict=True)},
        "stems": None,
        "window": None,
        "repeat_of": None,
        "selection": selection,
    }


def finish(rows, session, info, rng):
    """Shuffles, numbers and labels the rows; adds the reveal info."""
    rows = list(rows)
    rng.shuffle(rows)
    for i, row in enumerate(rows, start=1):
        row["id"] = f"{session}-{i:02d}"
        row["session"] = session
        row["info"] = {clip: info.get(row[clip], {}) for clip in ratings.CLIPS}
    return rows


# --- reading the registry and the rater's history --------------------------------


def session_name(rater_id, sessions_dir=config.SESSIONS_DIR):
    n = len(list(sessions_dir.glob(f"{rater_id}-*.jsonl"))) + 1
    return f"{rater_id}-{n:03d}"


def rater_history(rater_id, ratings_path=config.RATINGS_FILE, sessions_dir=config.SESSIONS_DIR):
    """(this rater's ratings, triplets already given to them, tracks any rater flagged)."""
    all_ratings = ratings.load_jsonl(ratings_path)
    mine = [r for r in all_ratings if r["rater_id"] == rater_id]
    seen = {seen_key(r["x"], r["a"], r["b"]) for r in mine}
    for path in sessions_dir.glob(f"{rater_id}-*.jsonl"):
        seen |= {seen_key(r["x"], r["a"], r["b"]) for r in ratings.load_jsonl(path)}
    return mine, seen, set(ratings.flagged_tracks(all_ratings))


def registry_tables(registry):
    """(audio {track: {source, ref}}, info {track: {title, artist}}, catalog track ids with
    audio + embeddings, catalog ids of pilot songs)."""
    audio = {
        r["track_id"]: {"source": r["audio_source"], "ref": r["artifact"].replace("\\", "/")}
        for r in registry.conn.execute(
            "SELECT t.track_id, t.audio_source, s.artifact FROM tracks t JOIN stage_status s USING (track_id)"
            " WHERE s.stage = ? AND s.state = ?",
            (config.AUDIO_STAGE, DONE),
        )
    }
    info = {
        r["track_id"]: {"title": r["title"], "artist": r["artist"], "year": None}
        for r in registry.conn.execute("SELECT track_id, title, artist FROM tracks")
    }
    catalog = [
        r["track_id"]
        for r in registry.conn.execute(
            "SELECT m.track_id FROM memberships m JOIN stage_status e ON e.track_id = m.track_id"
            " WHERE m.dataset = ? AND e.stage = ? AND e.state = ? ORDER BY m.track_id",
            (SpotifyCatalog.name, config.EMBED_STAGE, DONE),
        )
        if r["track_id"] in audio
    ]
    pilot = {
        json.loads(r["meta"]).get("catalog_track")
        for r in registry.conn.execute("SELECT meta FROM tracks WHERE audio_source = ?", (config.PILOT,))
    }
    return audio, info, catalog, pilot - {None}


def shared_candidates(registry, audio):
    """Public-dataset triplets whose three tracks all have audio. DIM-SIM: the clean subset only."""
    out = []
    for dataset in SHARED_SOURCES:
        for r in registry.conn.execute(
            "SELECT id, a, b, c, value, weight, meta FROM judgments WHERE dataset = ? AND kind = ?",
            (dataset, TRIPLET),
        ):
            if dataset == "dim_sim" and not json.loads(r["meta"]).get("clean"):
                continue
            if all(t in audio for t in (r["a"], r["b"], r["c"])):
                out.append((dataset, r["id"], r["a"], r["b"], r["c"], r["value"], r["weight"]))
    return out


def build(registry, rater_id, seed=None, ratings_path=config.RATINGS_FILE, sessions_dir=config.SESSIONS_DIR):
    """(session name, rows) for the next session of `rater_id`."""
    from ..experiments.stem_weights import Vectors
    from . import committee as committee_mod

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rated, seen, flagged = rater_history(rater_id, ratings_path, sessions_dir)
    audio, info, catalog, pilot = registry_tables(registry)

    def split_of(track_id):
        return ratings.song_split(track_id, always_validation=pilot)

    vectors = Vectors(registry)
    committee = committee_mod.fit(registry, vectors, np_rng)
    pool = committee_mod.load_pool(registry, catalog, vectors)
    del vectors  # the pool has its own copy; keeping the cache would double the memory
    picker = Picker(pool, committee, split_of, np_rng, seen, flagged)
    logger.info("pool: %d catalog songs (%d validation)", len(pool.ids), len(picker.side[ratings.VALIDATION]))

    rows = []
    original = pick_repeat(rated, rng)
    if original is not None:
        rows.append(repeat_row(original))
        picker.exclude((original["x"], original["a"], original["b"]))
    shared = pick_shared(shared_candidates(registry, audio), split_of, picker.used, picker.seen, rng)
    if shared is not None:
        x, a, b, selection = shared
        rows.append(triplet_row(ratings.SHARED, ratings.FIT, x, a, b, selection, audio))
        picker.exclude((x, a, b))
    for design, split, n in SESSION_PLAN:
        if design == ratings.CONTESTED:
            picked = picker.contested_triplets(split, n)
        elif design == ratings.NEIGHBOURHOOD:
            picked = picker.neighbourhood_triplets(split, n)
        else:
            continue
        rows += [triplet_row(design, split, x, a, b, sel, audio) for x, a, b, sel in picked]
        if len(picked) < n:
            logger.warning("%s/%s: only %d of %d triplets found", design, split, len(picked), n)
    name = session_name(rater_id, sessions_dir)
    return name, finish(rows, name, info, rng)


def write(rows, name, sessions_dir=config.SESSIONS_DIR):
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return path


def summary(rows):
    lines = []
    for row in rows:
        sel = row["selection"]
        extra = {ratings.CONTESTED: f"disagreement {sel.get('disagreement')}", ratings.SHARED: sel.get("source", "")}
        names = " / ".join(f"{row['info'][c].get('artist')} - {row['info'][c].get('title')}" for c in ratings.CLIPS)
        lines.append(f"{row['id']}  {row['design']:<13} {row['split']:<10} {extra.get(sel['rule'], ''):<22} {names}")
    return "\n".join(lines)
