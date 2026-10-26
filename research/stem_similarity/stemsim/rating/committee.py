"""Committee scoring: how evenly do plausible similarity models split on
a triplet? The session builder's "contested" triplets are the most even splits,
where an answer teaches the most.

Each member scores a triplet (X, A, B) as sim(X, A) - sim(X, B) and votes A when
that's positive (a tie is half a vote). Members come in four families of equal
total weight, so 100 bootstrap refits can't outvote the rest:
  posterior  stem weights refit on bootstrap resamples of the fit sets: the
             weights' own uncertainty
  datasets   weights fit on DIM-SIM alone and on MagnaTagATune alone
  baselines  mix only, and equal weights
  app        the app's current model; abstains when a song isn't in its catalog
split = weighted share of votes for A; disagreement = 1 - |2 split - 1|
(1 = an even split, 0 = unanimous).

Stem similarities are stem_weights.py's: per-channel cosine of full-preview
MuQ-MuLan embeddings, z-scored per channel, silence-gated. Weights are fit on
public datasets only, never on a rater's ratings, so validation ratings can't
leak into selection.

The Pool is in memory (~200 MB for the 20k-song catalog; a 1M-song catalog
needs a precomputed catalog index). X is scored against every row in one batched matmul
(~6 ms for 16k rows) and callers index the result: indexing the embeddings
first would copy ~150 MB per anchor.
"""

import logging
from dataclasses import dataclass

import numpy as np

from overtone.ml import dedup

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 100
POSTERIOR, DATASETS, BASELINES, APP = "posterior", "datasets", "baselines", "app"
STEM_FAMILIES = (POSTERIOR, DATASETS, BASELINES)
FAMILIES = (*STEM_FAMILIES, APP)


@dataclass
class Pool:
    """Candidate songs as arrays: row i is ids[i]."""

    ids: list
    emb: np.ndarray  # (channels, N, d), unit length; channel-major for the batched matmul
    audible: np.ndarray  # (N, channels) bool
    artists: np.ndarray  # (N,) normalized artist names
    app: np.ndarray | None = None  # (N, d') unit app-model vectors, NaN rows where the app lacks the song

    def __post_init__(self):
        self.row = {track_id: i for i, track_id in enumerate(self.ids)}

    def zsim(self, x, stats):
        """(N, channels): z-scored, silence-gated similarity of row x to every row."""
        mean, std = stats
        cos = np.matmul(self.emb, self.emb[:, x, :, None])[..., 0].T
        both = self.audible[x] & self.audible
        one = self.audible[x] ^ self.audible
        return np.where(both, (cos - mean) / std, np.where(one, -1.0, 0.0))

    def app_sim(self, x):
        """(N,): app-model cosine of row x to every row; NaN where either song is missing."""
        if self.app is None:
            return np.full(len(self.ids), np.nan)
        return self.app @ self.app[x]


@dataclass
class Committee:
    names: list  # one per stem-weight member
    families: list  # family of each stem-weight member
    weights: np.ndarray  # (members, channels)
    fitted: np.ndarray  # (channels,) weights fit on all fit-set data
    stats: tuple  # (mean, std) per channel, as stem_weights computes them

    def __post_init__(self):
        families = np.asarray(self.families)
        self._members = {f: np.flatnonzero(families == f) for f in STEM_FAMILIES}

    def scores(self, pool, x):
        """X's similarity to every pool row: (stem z-sims (N, channels), app cosine (N,))."""
        return pool.zsim(x, self.stats), pool.app_sim(x)

    def split(self, scores, a, b):
        """For candidate rows a[j], b[j]: (share voting A, {family: share})."""
        z, app = scores
        votes = _votes((z[a] - z[b]) @ self.weights.T)  # (n, members)
        by_family = {f: votes[:, rows].mean(1) for f, rows in self._members.items()}
        app_diff = app[a] - app[b]
        by_family[APP] = np.where(np.isnan(app_diff), np.nan, _votes(np.nan_to_num(app_diff)))
        stacked = np.column_stack([by_family[f] for f in FAMILIES])
        return np.nanmean(stacked, axis=1), by_family  # a missing app abstains

    def top_k(self, scores, candidates, k):
        """{model: rows}: X's k nearest among `candidates` under mix only, fitted stems and the app."""
        z, app = scores
        candidates = np.asarray(candidates)
        z = z[candidates]
        out = {"mix": _top(z[:, 0], candidates, k), "fitted": _top(z @ self.fitted, candidates, k)}
        app = app[candidates]
        known = ~np.isnan(app)
        if known.any():
            out[APP] = _top(app[known], candidates[known], k)
        return out


def disagreement(share):
    return 1 - np.abs(2 * np.asarray(share) - 1)


def _votes(scores):
    return (scores > 0) + 0.5 * (scores == 0)


def _top(values, rows, k):
    """The rows with the k largest values, in no particular order."""
    return rows[np.argpartition(-values, k - 1)[:k]] if len(rows) > k else rows


def fit(registry, vectors, rng, n_bootstrap=N_BOOTSTRAP):
    """The committee, fit on stem_weights' fit sets (loads their vectors into `vectors`)."""
    from ..experiments import stem_weights as sw  # imports mlflow; only fitting needs it

    data = {name: sw.load_judgments(registry, name) for name in sw.FIT_SETS}
    for rows, windows in data.values():
        for row in rows:
            for t in (row["a"], row["b"], row["c"]):
                if t:
                    vectors.get(t, windows.get(t))
    stats = sw.channel_stats(vectors, rng)
    blocks, all_rows, names = [], [], []
    for name, (rows, windows) in data.items():
        D, kept = sw.features(vectors, [r for r in rows if r["kind"] == "triplet"], windows, stats)
        blocks.append(D)
        all_rows += kept
        names += [name] * len(kept)
    D, names = np.vstack(blocks), np.array(names)
    groups = sw.song_groups(all_rows)
    boot = sw.bootstrap_fits(D, groups, sw.L2, rng, n_bootstrap)
    per_dataset = np.array([sw.fit_weights(D[names == name], sw.L2) for name in sw.FIT_SETS])
    channels = len(sw.CHANNELS)
    baselines = np.array([np.eye(channels)[0], np.full(channels, 1 / channels)])
    logger.info("committee: fit on %d triplets (%s)", len(D), ", ".join(sw.FIT_SETS))
    return Committee(
        names=[f"bootstrap {i}" for i in range(len(boot))] + [f"{n} only" for n in sw.FIT_SETS] + ["mix", "equal"],
        families=[POSTERIOR] * len(boot) + [DATASETS] * len(per_dataset) + [BASELINES] * 2,
        weights=np.vstack([boot, per_dataset, baselines]),
        fitted=sw.fit_weights(D, sw.L2),
        stats=stats,
    )


def load_pool(registry, track_ids, vectors, with_app=True):
    """Pool of the given tracks that have embeddings (full-preview window)."""
    meta = {
        r["track_id"]: r
        for r in registry.conn.execute(
            "SELECT track_id, artist, json_extract(meta, '$.spotify_track_id') AS spotify FROM tracks"
        )
    }
    ids, audible, emb = [], [], None
    for track_id in track_ids:
        v = vectors.get(track_id) if track_id in meta else None
        if v is None:
            continue
        if emb is None:  # sized for every track; the few without embeddings leave unused rows
            emb = np.empty((v[0].shape[0], len(track_ids), v[0].shape[1]), np.float32)
        emb[:, len(ids)] = v[0]
        ids.append(track_id)
        audible.append(v[1])
    if emb is None:
        raise ValueError("none of the tracks has an embedding")
    pool = Pool(
        ids,
        emb[:, : len(ids)],
        np.array(audible),
        np.array([dedup.normalize_artist(meta[t]["artist"] or "") for t in ids]),
    )
    if with_app:
        pool.app = _app_vectors([meta[t]["spotify"] for t in ids])
    return pool


def _app_vectors(spotify_ids):
    """The app model's genre embedding per song, unit length; NaN rows where it doesn't know the song."""
    from overtone.ml.similarity_model import SimilarityModel

    model = SimilarityModel()
    Z = model.Z / np.linalg.norm(model.Z, axis=1, keepdims=True)
    out = np.full((len(spotify_ids), Z.shape[1]), np.nan, np.float32)
    for i, spotify_id in enumerate(spotify_ids):
        row = model.row_of(spotify_id) if spotify_id else None
        if row is not None:
            out[i] = Z[row]
    return out
