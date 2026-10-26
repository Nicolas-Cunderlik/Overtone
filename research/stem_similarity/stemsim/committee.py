"""Committee scoring (SSM-8): how evenly do plausible similarity models split
on a triplet? The rating app's "contested" triplets are the most even splits -
where an answer teaches the most.

Members each score a triplet (X, A, B) as sim(X, A) - sim(X, B) and vote A
when it's positive (a tie is half a vote). They come in four families of equal
total weight, so 100 bootstrap refits can't outvote everything else:
  posterior  stem weights refit on bootstrap resamples of the fit sets - the
             spread is the weights' own uncertainty (drums/bass/vocals are
             each somewhere in 0.0-0.2 as of the last stem_weights run)
  datasets   weights fit on DIM-SIM alone and on MagnaTagATune alone
  baselines  mix only, and equal weights
  app        the app's current model (Spotify features + genre embedding);
             it abstains when a song isn't in its catalog
split = weighted share of votes for A; disagreement = 1 - |2 split - 1|
(1 = an even split, 0 = unanimous).

Stem similarities are the same as stem_weights.py's: per-channel cosine of
full-preview MuQ-MuLan embeddings, z-scored per channel (stats over the fit
sets' songs), with silence gating. Weights are fit only on public datasets,
never on a rater's own ratings, so validation ratings can't leak into
selection.

Everything here works on in-memory arrays (Pool). Fine for the 20k-song
catalog (~200 MB); a 1M-song catalog needs the consolidated index of SSM-3.
"""

import logging
from dataclasses import dataclass

import numpy as np

from overtone.ml import dedup

logger = logging.getLogger(__name__)

N_BOOTSTRAP = 100
L2 = 1.0  # stem_weights' default
FAMILIES = ("posterior", "datasets", "baselines", "app")


@dataclass
class Pool:
    """Candidate songs as arrays: row i is ids[i]."""

    ids: list
    emb: np.ndarray  # (N, channels, d), unit length per channel
    audible: np.ndarray  # (N, channels) bool
    artists: np.ndarray  # (N,) normalized artist names
    app: np.ndarray | None = None  # (N, d') unit app-model vectors, NaN rows where the app lacks the song

    def __post_init__(self):
        self.row = {track_id: i for i, track_id in enumerate(self.ids)}

    def zsim(self, x, others, stats):
        """(len(others), channels) z-scored, silence-gated similarity of row x to rows `others`."""
        mean, std = stats
        cos = np.einsum("cd,ncd->nc", self.emb[x], self.emb[others])
        z = (cos - mean) / std
        both = self.audible[x] & self.audible[others]
        one = self.audible[x] ^ self.audible[others]
        return np.where(both, z, np.where(one, -1.0, 0.0))

    def app_sim(self, x, others):
        """(len(others),) app-model cosine; NaN where either song is missing."""
        if self.app is None:
            return np.full(len(others), np.nan)
        return self.app[others] @ self.app[x]


@dataclass
class Committee:
    names: list  # one per stem-weight member
    families: list  # family of each stem-weight member
    weights: np.ndarray  # (members, channels)
    fitted: np.ndarray  # (channels,) weights fit on all fit-set data - the "fitted" model
    stats: tuple  # (mean, std) per channel, as stem_weights computes them

    def split(self, pool, x, a, b):
        """For anchor row x and candidate rows a[j], b[j]: (share voting A, {family: share})."""
        a, b = np.asarray(a), np.asarray(b)
        diff = pool.zsim(x, a, self.stats) - pool.zsim(x, b, self.stats)
        votes = _votes(diff @ self.weights.T)  # (n, members)
        by_family = {f: votes[:, [i for i, g in enumerate(self.families) if g == f]].mean(1) for f in FAMILIES[:3]}
        app = pool.app_sim(x, a) - pool.app_sim(x, b)
        by_family["app"] = np.where(np.isnan(app), np.nan, _votes(np.nan_to_num(app)))
        stacked = np.column_stack([by_family[f] for f in FAMILIES])
        return np.nanmean(stacked, axis=1), by_family  # equal family weights; a missing app abstains

    def top_k(self, pool, x, candidates, k):
        """{model: rows} - x's k nearest among `candidates` under mix only, fitted stems and the app."""
        candidates = np.asarray(candidates)
        z = pool.zsim(x, candidates, self.stats)
        out = {"mix": candidates[np.argsort(-z[:, 0])[:k]], "fitted": candidates[np.argsort(-(z @ self.fitted))[:k]]}
        app = pool.app_sim(x, candidates)
        if not np.isnan(app).all():
            known = ~np.isnan(app)
            out["app"] = candidates[known][np.argsort(-app[known])[:k]]
        return out


def disagreement(share):
    return 1 - np.abs(2 * np.asarray(share) - 1)


def _votes(scores):
    return (scores > 0) + 0.5 * (scores == 0)


def fit(registry, vectors, rng, n_bootstrap=N_BOOTSTRAP, l2=L2):
    """The committee, fit on stem_weights' fit sets (loads their vectors into `vectors`)."""
    from .experiments import stem_weights as sw  # imports mlflow; only fitting needs it

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
    groups = sw.triple_groups(all_rows)
    boot = sw.bootstrap_fits(D, groups, l2, rng, n_bootstrap)
    per_dataset = np.array([sw.fit_weights(D[names == name], l2) for name in sw.FIT_SETS])
    channels = len(sw.CHANNELS)
    baselines = np.array([np.eye(channels)[0], np.full(channels, 1 / channels)])
    logger.info("committee: fit on %d triplets (%s)", len(D), ", ".join(sw.FIT_SETS))
    return Committee(
        names=[f"bootstrap {i}" for i in range(len(boot))] + [f"{n} only" for n in sw.FIT_SETS] + ["mix", "equal"],
        families=["posterior"] * len(boot) + ["datasets"] * len(per_dataset) + ["baselines"] * 2,
        weights=np.vstack([boot, per_dataset, baselines]),
        fitted=sw.fit_weights(D, l2),
        stats=stats,
    )


def load_pool(registry, track_ids, vectors, with_app=True):
    """Pool of the given tracks that have embeddings (full-preview window)."""
    ids, emb, audible, artists, spotify = [], [], [], [], []
    rows = {
        r["track_id"]: r
        for r in registry.conn.execute(
            "SELECT track_id, artist, json_extract(meta, '$.spotify_track_id') AS spotify FROM tracks"
        )
    }
    for track_id in track_ids:
        v = vectors.get(track_id)
        if v is None or track_id not in rows:
            continue
        ids.append(track_id)
        emb.append(v[0])
        audible.append(v[1])
        artists.append(dedup.normalize_artist(rows[track_id]["artist"] or ""))
        spotify.append(rows[track_id]["spotify"])
    pool = Pool(ids, np.array(emb, np.float32), np.array(audible), np.array(artists))
    if with_app:
        pool.app = _app_vectors(spotify)
    return pool


def _app_vectors(spotify_ids):
    """The app model's genre embedding per song, unit length; NaN rows where it doesn't know the song."""
    from overtone.ml.similarity_model import SimilarityModel

    model = SimilarityModel()
    Z = model.Z / np.linalg.norm(model.Z, axis=1, keepdims=True)
    out = np.full((len(spotify_ids), Z.shape[1]), np.nan, np.float32)
    for i, spotify_id in enumerate(spotify_ids):
        row = model._row_by_id.get(spotify_id) if spotify_id else None
        if row is not None:
            out[i] = Z[row]
    return out
