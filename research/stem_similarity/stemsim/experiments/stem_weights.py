"""Experiment: do per-stem similarities, weighted by human judgments, agree
with people more than the mix alone - or the app's current model?

    python -m stemsim.experiments.stem_weights [--run-name NAME] [--l2 1.0] [--folds 5] [--repeats 100]

Read-only: uses the registry and embeddings, writes only an MLflow run.

Setup
- Fit sets (repeated grouped k-fold CV): dim_sim, magnatagatune. Triplets
  that share a song, directly or through a chain, form one group, and folds
  split by group, so no song is on both sides of a fold. Groups are small
  (largest 5 triplets; 22% of DIM-SIM triplets share a song, as of
  2026-10-25), so grouping costs no data.
- The split is repeated (default 100, one seed each) and each triplet's
  out-of-fold agreement is averaged, so results don't hinge on one
  partition. The spread across repeats is logged as split noise. Repeats
  don't narrow the CIs below: those depend on the number of judgments.
- Held out entirely (never fitted on): listener_abx - overall and per-stem
  facets - and ccmsim (pairs, Spearman). The final weights are fit on all
  fit-set data and applied to these. Fit-set triplets that contain a
  held-out song are dropped (6 of ~4.4k as of 2026-10-25), and the z-score
  stats below use only the remaining fit-set songs, so held-out songs never
  shape the weights.
- Splits are logged, not stored: groups.csv (each judgment's role and group)
  plus the seed and repeat count rebuild every fold exactly.
- Each song is embedded on the window its dataset's raters heard (membership
  clip) when there is one, else the full audio.
- Per channel (mix, drums, bass, vocals, other): cosine similarity, z-scored
  per channel over random audible pairs so channels share a scale. Silence
  gating: a stem silent in exactly one song scores -1 (one has bass, the other
  doesn't); silent in both scores 0 (no evidence either way).
- A triplet (anchor a, positive b, negative c) becomes d = zsim(a,b) - zsim(a,c)
  per channel; a model agrees when its score w.d > 0 (ties count half).
- Weights: non-negative logistic regression, L2, no intercept.

Baselines: random (0.5), mix only, equal weights, each stem alone, and the
app's current model (Spotify features + genre embedding) - on listener_abx
only, the one dataset whose songs have Spotify features.

Uncertainty (bootstrap, 95%)
- Every metric gets a CI, and every model a *paired* difference vs mix only
  (and vs the app model on listener_abx): all models are scored on the same
  resamples, so a difference's CI is far tighter than comparing two CIs.
  With n = 1794 DIM-SIM triplets, single-model CIs are about +-2 points -
  as wide as the stems-vs-mix gap itself - so only the paired CI answers
  "do stems help?".
- Resampling draws whole song groups (the same groups as the folds), not
  rows: triplets sharing a song (e.g. MagnaTagATune's two orderings of one
  triple) aren't independent evidence.
- Weights: refit on group resamples. The fold std understates their
  uncertainty because folds share 80% of their data; between v0 (660 DIM-SIM
  triplets) and v1 (1794) bass moved 0.04 -> 0.11, far outside +-0.02.
"""

import argparse
import json
import logging
import time

import mlflow
import numpy as np
import pandas as pd
from overtone_core.ml.stems import CHANNELS
from scipy.optimize import minimize
from scipy.stats import spearmanr

from .. import config, tracking
from ..registry import Registry
from . import report

logger = logging.getLogger(__name__)

SILENT_DB = -45.0
FIT_SETS = ("dim_sim", "magnatagatune")
HELD_OUT_TRIPLETS = ("listener_abx",)
HELD_OUT_PAIRS = ("ccmsim",)
Z_SAMPLE_PAIRS = 20_000
BOOTSTRAP = 2_000
WEIGHT_BOOTSTRAP = 500  # each resample is a full refit
L2 = 1.0
FOLDS = 5
CV_REPEATS = 100
SEED = 0
BASELINE = "mix only"
APP_MODEL = "app model (current)"

# --- run report (report.py): the experiment-level sections
SUCCESS = (
    "Stems beat the mix alone - and the current app model - on held-out human judgments, with the paired "
    "95% CI of the difference excluding zero. At minimum: the best configuration beats the current app "
    "model on the held-out sets (or we learn why not)."
)
CAVEATS = """\
- listener_abx, the target that matters most, is small (see n in Results): its CIs are wide.
- DIM-SIM raters judged 3 s moments of a different provider's previews, not our Deezer audio.
- MagnaTagATune audio is 32 kbps MP3 - a domain shift from the catalog."""
RELATED = ("SSM-19", "SSM-22", "SSM-10", "SSM-11")
REPORT_MODELS = ("mix only", "equal weights", "fitted (out-of-fold)", "fitted", APP_MODEL)
REPORT_MIN_N = 5  # smaller slices (e.g. single listener_abx facet ratings) stay in the CSVs


class Vectors:
    """Unit-normalized embeddings + audibility per (track, window), loaded lazily."""

    def __init__(self, registry):
        self.registry = registry
        self._files = {}

    def get(self, track_id, window=None):
        if track_id not in self._files:
            out = self.registry.output(track_id, config.EMBED_STAGE)
            self._files[track_id] = None if out is None else dict(np.load(out[0]))
        data = self._files[track_id]
        if data is None:
            return None
        index = 0  # full audio
        if window is not None:
            for k, (start, _end) in enumerate(data["windows"]):
                if k and abs(start - window[0]) < 0.05:
                    index = k
        emb = data["emb"][index]
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
        return emb, data["rms_db"][index] > SILENT_DB


def channel_stats(vectors, track_ids, rng):
    """Mean/std of cosine per channel over random pairs of audible channels,
    drawn from `track_ids` (full audio). Sorted so the sample doesn't depend
    on set order."""
    loaded = [v for v in (vectors.get(t) for t in sorted(track_ids)) if v is not None]
    i, j = rng.integers(0, len(loaded), (2, Z_SAMPLE_PAIRS))
    mean, std = np.zeros(len(CHANNELS)), np.ones(len(CHANNELS))
    for c in range(len(CHANNELS)):
        sims = [
            float(loaded[x][0][c] @ loaded[y][0][c])
            for x, y in zip(i, j, strict=True)
            if x != y and loaded[x][1][c] and loaded[y][1][c]
        ]
        mean[c], std[c] = np.mean(sims), np.std(sims) + 1e-9
    return mean, std


def zsim(x, y, mean, std):
    (ex, ax), (ey, ay) = x, y
    z = (np.einsum("cd,cd->c", ex, ey) - mean) / std
    return np.where(ax & ay, z, np.where(ax ^ ay, -1.0, 0.0))


def load_judgments(registry, dataset):
    windows = {
        row["track_id"]: (row["clip_start"], row["clip_end"])
        for row in registry.conn.execute(
            "SELECT track_id, clip_start, clip_end FROM memberships WHERE dataset = ? AND clip_start IS NOT NULL",
            (dataset,),
        )
    }
    rows = registry.conn.execute(  # ORDER BY: group ids (and so folds) follow row order
        "SELECT id, kind, a, b, c, value, weight, facet, meta FROM judgments WHERE dataset = ? ORDER BY id",
        (dataset,),
    ).fetchall()
    return rows, windows


def features(vectors, rows, windows, stats):
    """Triplet rows -> (D, kept rows). Rows missing any embedding are dropped."""
    D, kept = [], []
    for row in rows:
        vs = [vectors.get(t, windows.get(t)) for t in (row["a"], row["b"], row["c"])]
        if any(v is None for v in vs):
            continue
        D.append(zsim(vs[0], vs[1], *stats) - zsim(vs[0], vs[2], *stats))
        kept.append(row)
    return np.array(D).reshape(-1, len(CHANNELS)), kept


def fit_weights(D, l2):
    """Non-negative w maximizing mean log sigmoid(D.w) - every row is oriented
    so the human-chosen song is the positive."""

    def loss(w):
        m = D @ w
        return np.logaddexp(0, -m).mean() + l2 * (w @ w) / len(D)

    def grad(w):
        m = D @ w
        return -(D * (1 / (1 + np.exp(m)))[:, None]).mean(0) + 2 * l2 * w / len(D)

    return minimize(loss, np.full(D.shape[1], 0.2), jac=grad, method="L-BFGS-B", bounds=[(0, None)] * D.shape[1]).x


def correctness(scores):
    """Per row: 1 agrees, 0.5 tie, 0 disagrees; NaN where the model has no score."""
    scores = np.asarray(scores, float)
    return np.where(np.isnan(scores), np.nan, (scores > 0) + 0.5 * (scores == 0))


def row_correctness(scores):
    """correctness() per row; (repeats, rows) scores from repeated CV are averaged over the repeats."""
    c = correctness(scores)
    return c.mean(0) if c.ndim == 2 else c


def song_groups(rows):
    """Group id per row: rows linked by shared songs, directly or through a
    chain, share one - so splitting by group never puts a song on both sides."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        songs = [t for t in (r["a"], r["b"], r["c"]) if t]
        for t in songs[1:]:
            parent[find(t)] = find(songs[0])
    return pd.factorize(pd.Series([find(r["a"]) for r in rows], dtype=object))[0]


def repeated_folds(groups, k, repeats, seed):
    """Yield (repeat, train, test) row indices: k folds of whole groups,
    reshuffled each repeat. A repeat's split depends only on (seed, repeat)
    and the group ids, so groups.csv + the seed rebuild it exactly."""
    n_groups = int(groups.max()) + 1
    for r in range(repeats):
        fold = (np.random.default_rng([seed, r]).permutation(n_groups) % k)[groups]
        for f in range(k):
            yield r, np.flatnonzero(fold != f), np.flatnonzero(fold == f)


def songs(rows):
    """Every track id in `rows`."""
    return {t for row in rows for t in (row["a"], row["b"], row["c"]) if t}


def held_out_songs(registry, rows):
    """songs(rows), plus the catalog song id behind each pilot track (pilot
    audio is its own "pilot:" track; the shared "song:" id is in its meta)."""
    ids = songs(rows)
    pilot = registry.conn.execute("SELECT track_id, meta FROM tracks WHERE audio_source = ?", (config.PILOT,))
    ids |= {json.loads(r["meta"]).get("catalog_track") for r in pilot if r["track_id"] in ids}
    return ids - {None}


def touches(row, songs):
    return any(t in songs for t in (row["a"], row["b"], row["c"]) if t)


def group_resamples(groups, rng, n):
    """(ids, counts): dense group id per row, and (n, G) how often each group
    is drawn in each resample (G groups drawn with replacement)."""
    ids = pd.factorize(pd.Series(groups))[0]
    G = int(ids.max()) + 1
    draws = rng.integers(0, G, (n, G)) + np.arange(n)[:, None] * G
    return ids, np.bincount(draws.ravel(), minlength=n * G).reshape(n, G).astype(float)


def _boot_mean(values, ids, counts):
    """Point mean of per-row values (NaN rows skipped) and its 95% bootstrap CI."""
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan, np.nan, np.nan, 0
    G = counts.shape[1]
    sums = np.bincount(ids, weights=np.where(valid, values, 0.0), minlength=G)
    sizes = np.bincount(ids, weights=valid.astype(float), minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = (counts @ sums) / (counts @ sizes)
    lo, hi = np.nanpercentile(resampled, [2.5, 97.5])
    return float(sums.sum() / sizes.sum()), float(lo), float(hi), int(valid.sum())


def bootstrap_agreement(scores, groups, rng, baselines=(BASELINE,)):
    """scores: {model: per-row scores, or (repeats, rows) from repeated CV},
    all aligned on the same rows.
    Returns ({model: (n, agreement, lo, hi)}, {(model, baseline): (n, diff, lo, hi)}).
    One set of resamples for everything, so differences are paired; a
    difference uses only the rows both models scored."""
    ids, counts = group_resamples(groups, rng, BOOTSTRAP)
    correct = {model: row_correctness(s) for model, s in scores.items()}
    metrics = {}
    for model, c in correct.items():
        point, lo, hi, n = _boot_mean(c, ids, counts)
        metrics[model] = (n, point, lo, hi)
    diffs = {}
    for base in baselines:
        if base not in correct:
            continue
        for model, c in correct.items():
            if model != base:
                point, lo, hi, n = _boot_mean(c - correct[base], ids, counts)
                diffs[(model, base)] = (n, point, lo, hi)
    return metrics, diffs


def bootstrap_spearman(scores, human, rng, baseline=BASELINE):
    """Like bootstrap_agreement, for pair datasets scored by Spearman correlation."""
    n = len(human)
    idx = rng.integers(0, n, (BOOTSTRAP, n))

    def rho(s, i):
        return spearmanr(s[i], human[i])[0]

    boot = {model: np.array([rho(s, i) for i in idx]) for model, s in scores.items()}
    metrics, diffs = {}, {}
    for model, s in scores.items():
        lo, hi = np.nanpercentile(boot[model], [2.5, 97.5])
        metrics[model] = (n, float(rho(s, slice(None))), float(lo), float(hi))
        if model != baseline:
            d = boot[model] - boot[baseline]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            diffs[(model, baseline)] = (n, metrics[model][1] - float(rho(scores[baseline], slice(None))), lo, hi)
    return metrics, diffs


def bootstrap_fits(D, groups, l2, rng, n=WEIGHT_BOOTSTRAP):
    """(n, channels) weights, each refit on one group resample."""
    ids, counts = group_resamples(groups, rng, n)
    return np.array([fit_weights(D[np.repeat(np.arange(len(D)), c[ids].astype(int))], l2) for c in counts])


def bootstrap_weights(D, groups, l2, rng):
    """95% CI per channel weight, refitting on group resamples."""
    return np.percentile(bootstrap_fits(D, groups, l2, rng), [2.5, 97.5], axis=0)


def fixed_models(D):
    out = {"mix only": D[:, 0], "equal weights": D.mean(1)}
    out.update({f"{c} only": D[:, k] for k, c in enumerate(CHANNELS) if c != "mix"})
    return out


def app_model_scores(rows):
    """Current app similarity (Spotify features + genre embedding) for pilot tracks."""
    from overtone_core.ml.similarity_model import SimilarityModel

    model = SimilarityModel()
    Z = model.Z / np.linalg.norm(model.Z, axis=1, keepdims=True)
    scores = []
    for row in rows:
        ids = [model.row_of(t.split(":", 1)[1]) for t in (row["a"], row["b"], row["c"])]
        scores.append(np.nan if None in ids else float(Z[ids[0]] @ Z[ids[1]] - Z[ids[0]] @ Z[ids[2]]))
    return np.array(scores)


def _metric_key(*parts):
    return ".".join(parts).replace(" ", "_").replace("(", "").replace(")", "")


def _cell(point, lo, hi, signed=False):
    if pd.isna(point):
        return "-"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f.format(point) if pd.isna(lo) else f"{f.format(point)} ({f.format(lo)} to {f.format(hi)})"


def _verdict(lo, hi):
    if pd.isna(lo):
        return "no CI"
    return "higher" if lo > 0 else "lower" if hi < 0 else "no difference shown"


def results_markdown(table, diff_table=None, weights=None, split_noise=None):
    """The Results section: main models per dataset, paired differences, split noise, weights."""
    t = table[table["model"].isin(REPORT_MODELS) & (table["n"] >= REPORT_MIN_N)]
    lines = [
        "| Dataset | Subset | n | Mix only | Equal weights | Fitted | App model |",
        "|---|---|---|---|---|---|---|",
    ]
    for (dataset, subset), g in t.groupby(["dataset", "subset"], sort=False):
        cell = {r.model: _cell(r.agreement, r.ci_low, r.ci_high) for r in g.itertuples()}
        fitted = cell.get("fitted (out-of-fold)", cell.get("fitted", "-"))
        lines.append(
            f"| {dataset} | {subset} | {g['n'].max()} | {cell.get(BASELINE, '-')} | "
            f"{cell.get('equal weights', '-')} | {fitted} | {cell.get(APP_MODEL, '-')} |"
        )
    parts = [
        "Agreement = share of human judgments the model agrees with (ties count half), 95% bootstrap CI in "
        "brackets; CCMSim is a Spearman correlation. Fit sets are scored out-of-fold.",
        "\n".join(lines),
    ]

    if diff_table is not None:
        d = diff_table[
            diff_table["model"].isin(REPORT_MODELS)
            & (diff_table["model"] != APP_MODEL)
            & (diff_table["n"] >= REPORT_MIN_N)
        ]
        rows = [
            f"| {r.dataset} | {r.subset} | {r.model} vs {r.baseline} | {r.n} | "
            f"{_cell(r.difference, r.ci_low, r.ci_high, signed=True)} | {_verdict(r.ci_low, r.ci_high)} |"
            for r in d.itertuples()
        ]
        header = ["| Dataset | Subset | Comparison | n | Difference (95% CI) | Verdict |", "|---|---|---|---|---|---|"]
        parts += ["### Paired differences", "\n".join([*header, *rows])]
    else:
        parts.append("_No paired differences in this run (added in v2-paired-cis)._")

    if split_noise is not None:
        rows = [
            f"| {r.dataset} | {r.subset} | {r.n} | {r.mean:.3f} | {r.std:.4f} | {r.min:.3f} to {r.max:.3f} |"
            for r in split_noise.itertuples()
        ]
        header = ["| Dataset | Subset | n | Mean | Std | Range |", "|---|---|---|---|---|---|"]
        parts += [
            "### Split noise",
            f"Fitted out-of-fold agreement per CV repeat ({split_noise['repeats'].iloc[0]} repeats): how much "
            "the result depends on the partition alone. The Fitted column above averages over the repeats.",
            "\n".join([*header, *rows]),
        ]

    if weights:
        final = weights["final"]
        if "ci_low" in weights:
            spread = {c: f"{weights['ci_low'][c]:.2f} to {weights['ci_high'][c]:.2f}" for c in final}
            header = "95% CI (refit on resamples)"
        else:
            spread = {c: f"{weights['fold_mean'][c]:.2f} ± {weights['fold_std'][c]:.2f}" for c in final}
            header = "Fold mean ± std (understates uncertainty)"
        rows = [f"| {c} | {w:.3f} | {spread[c]} |" for c, w in final.items()]
        parts += ["### Fitted weights", "\n".join([f"| Channel | Weight | {header} |", "|---|---|---|", *rows])]

    parts.append("Single-stem models and slices with n < 5 are in `results.csv` / `differences.csv`.")
    return "\n\n".join(parts)


def write_report(run, table, diff_table, weights, split_noise=None, runtime_s=None):
    params = run.data.params
    command = (
        f".venv/Scripts/python -m stemsim.experiments.stem_weights --run-name {run.info.run_name} "
        f"--l2 {params.get('l2', L2)} --folds {params.get('folds', FOLDS)} --repeats {params.get('cv_repeats', 1)}"
    )
    text = report.render(
        run,
        __doc__,
        results=results_markdown(table, diff_table, weights, split_noise),
        success=SUCCESS,
        caveats=CAVEATS,
        related=RELATED,
        command=command,
        runtime_s=runtime_s,
    )
    report.log(run.info.run_id, text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-name", default="stem-weights")
    parser.add_argument("--l2", type=float, default=L2)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--repeats", type=int, default=CV_REPEATS, help="CV repeats, each a fresh seeded split")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()

    registry = Registry()
    rng = np.random.default_rng(SEED)
    vectors = Vectors(registry)
    data = {name: load_judgments(registry, name) for name in (*FIT_SETS, *HELD_OUT_TRIPLETS, *HELD_OUT_PAIRS)}
    held_out = held_out_songs(registry, (r for name in (*HELD_OUT_TRIPLETS, *HELD_OUT_PAIRS) for r in data[name][0]))
    fit_songs = songs(
        r for name in FIT_SETS for r in data[name][0] if r["kind"] == "triplet" and not touches(r, held_out)
    )
    stats = channel_stats(vectors, fit_songs, rng)  # fit-set songs only: held-out songs never shape the weights

    results = []  # (dataset, subset, model, n, agreement, lo, hi)
    differences = []  # (dataset, subset, model, baseline, n, difference, lo, hi)

    def record(dataset, subset, metrics, diffs):
        for model, (n, point, lo, hi) in metrics.items():
            results.append((dataset, subset, model, n, point, lo, hi))
        for (model, base), (n, point, lo, hi) in diffs.items():
            differences.append((dataset, subset, model, base, n, point, lo, hi))

    split_rows = []  # groups.csv: every judgment's role in this run and its group

    # --- fit sets: repeated out-of-fold scores
    fit_D, fit_rows, fit_names = [], [], []
    excluded = {}
    for name in FIT_SETS:
        rows, windows = data[name]
        triplets = [r for r in rows if r["kind"] == "triplet"]
        overlap = [r for r in triplets if touches(r, held_out)]
        D, kept = features(vectors, [r for r in triplets if not touches(r, held_out)], windows, stats)
        fit_D.append(D)
        fit_rows += kept
        fit_names += [name] * len(kept)
        excluded[name] = len(overlap)
        kept_ids = {r["id"] for r in kept}
        overlap_ids = {r["id"] for r in overlap}
        for r in triplets:
            if r["id"] not in kept_ids:
                role = "excluded: held-out song" if r["id"] in overlap_ids else "excluded: not embedded"
                split_rows.append((r["id"], name, role, None))
        logger.info(
            "%s: %d of %d triplets used (%d dropped: share a song with a held-out set)",
            name,
            len(kept),
            len(triplets),
            len(overlap),
        )
    D = np.vstack(fit_D)
    names = np.array(fit_names)
    groups = song_groups(fit_rows)
    split_rows += [(r["id"], n, "fit", int(g)) for r, n, g in zip(fit_rows, names, groups, strict=True)]
    oof = np.full((args.repeats, len(D)), np.nan)
    fold_weights = []
    for repeat, train, test in repeated_folds(groups, args.folds, args.repeats, SEED):
        w = fit_weights(D[train], args.l2)
        fold_weights.append(w)
        oof[repeat, test] = D[test] @ w
    clean = np.array([json.loads(r["meta"]).get("clean", False) for r in fit_rows])
    split_noise = []  # (dataset, subset, n, repeats, mean, std, min, max) of per-repeat fitted agreement
    for name in FIT_SETS:
        for subset, mask in (("all", names == name), ("clean", (names == name) & clean)):
            if subset == "clean" and name != "dim_sim":
                continue
            scored = {"fitted (out-of-fold)": oof[:, mask], **fixed_models(D[mask])}
            record(name, subset, *bootstrap_agreement(scored, groups[mask], rng))
            per_repeat = correctness(oof[:, mask]).mean(1)
            split_noise.append(
                (
                    name,
                    subset,
                    int(mask.sum()),
                    args.repeats,
                    *(float(f(per_repeat)) for f in (np.mean, np.std, np.min, np.max)),
                )
            )

    # --- final weights, applied to held-out sets
    w_final = fit_weights(D, args.l2)
    w_ci = bootstrap_weights(D, groups, args.l2, rng)
    for name in HELD_OUT_TRIPLETS:
        rows, windows = data[name]
        H, kept = features(vectors, rows, windows, stats)
        facets = np.array([r["facet"] for r in kept])
        app = app_model_scores(kept)
        held_groups = song_groups(kept)
        split_rows += [(r["id"], name, "held out", int(g)) for r, g in zip(kept, held_groups, strict=True)]
        for facet in sorted(set(facets)):
            mask = facets == facet
            scored = {"fitted": H[mask] @ w_final, **fixed_models(H[mask])}
            baselines = (BASELINE,)
            if facet == "overall":
                scored[APP_MODEL] = app[mask]  # NaN where a song lacks Spotify features
                baselines = (BASELINE, APP_MODEL)
            record(name, facet, *bootstrap_agreement(scored, held_groups[mask], rng, baselines))
    for name in HELD_OUT_PAIRS:
        rows, windows = data[name]
        pairs = [
            (vectors.get(r["a"], windows.get(r["a"])), vectors.get(r["b"], windows.get(r["b"])), r["value"])
            for r in rows
            if r["kind"] == "pair"
        ]
        split_rows += [
            (r["id"], name, "held out", None)
            for r in rows
            if r["kind"] == "pair"
            and vectors.get(r["a"], windows.get(r["a"])) is not None
            and vectors.get(r["b"], windows.get(r["b"])) is not None
        ]
        pairs = [(x, y, v) for x, y, v in pairs if x is not None and y is not None]
        if len(pairs) >= 3:
            S = np.array([zsim(x, y, *stats) for x, y, _ in pairs])
            human = np.array([v for _, _, v in pairs])
            scored = {"fitted": S @ w_final, BASELINE: S[:, 0], "equal weights": S.mean(1)}
            record(name, "pairs (spearman)", *bootstrap_spearman(scored, human, rng))

    table = pd.DataFrame(results, columns=["dataset", "subset", "model", "n", "agreement", "ci_low", "ci_high"])
    diff_table = pd.DataFrame(
        differences, columns=["dataset", "subset", "model", "baseline", "n", "difference", "ci_low", "ci_high"]
    )
    noise_table = pd.DataFrame(split_noise, columns=["dataset", "subset", "n", "repeats", "mean", "std", "min", "max"])
    groups_table = pd.DataFrame(split_rows, columns=["judgment_id", "dataset", "role", "group"]).astype(
        {"group": "Int64"}
    )
    weights = {
        "final": dict(zip(CHANNELS, map(float, w_final), strict=True)),
        "ci_low": dict(zip(CHANNELS, map(float, w_ci[0]), strict=True)),
        "ci_high": dict(zip(CHANNELS, map(float, w_ci[1]), strict=True)),
        "fold_mean": dict(zip(CHANNELS, map(float, np.mean(fold_weights, 0)), strict=True)),
        "fold_std": dict(zip(CHANNELS, map(float, np.std(fold_weights, 0)), strict=True)),
        "z_mean": dict(zip(CHANNELS, map(float, stats[0]), strict=True)),
        "z_std": dict(zip(CHANNELS, map(float, stats[1]), strict=True)),
    }

    params = {
        "l2": args.l2,
        "folds": args.folds,
        "cv_repeats": args.repeats,
        "seed": SEED,
        "split_groups": "song",  # triplets linked by shared songs (v0-v2: "triple")
        "excluded_held_out_overlap": sum(excluded.values()),
        "silent_db": SILENT_DB,
        "embedding": config.EMBED_STAGE,
        "fit_sets": ",".join(FIT_SETS),
        "held_out": ",".join((*HELD_OUT_TRIPLETS, *HELD_OUT_PAIRS)),
    }
    with tracking.run("stem-weights", args.run_name, registry, config.EMBED_STAGE, params) as active:
        for row in table.itertuples():
            if not np.isnan(row.agreement):
                mlflow.log_metric(_metric_key(row.dataset, row.subset, row.model), row.agreement)
        for row in diff_table.itertuples():
            if not np.isnan(row.difference):
                key = _metric_key("diff", row.dataset, row.subset, f"{row.model} vs {row.baseline}")
                mlflow.log_metric(key, row.difference)
        for channel, value in weights["final"].items():
            mlflow.log_metric(f"weight.{channel}", value)
        for row in noise_table.itertuples():
            mlflow.log_metric(_metric_key("split_std", row.dataset, row.subset), row.std)
        mlflow.log_text(table.to_csv(index=False), "results.csv")
        mlflow.log_text(diff_table.to_csv(index=False), "differences.csv")
        mlflow.log_text(noise_table.to_csv(index=False), "split_noise.csv")
        mlflow.log_text(groups_table.to_csv(index=False), "groups.csv")
        mlflow.log_dict(weights, "weights.json")
    write_report(
        mlflow.get_run(active.info.run_id), table, diff_table, weights, noise_table, time.perf_counter() - started
    )

    pd.set_option("display.width", 160)
    fmt = {"float_format": lambda x: f"{x:.3f}", "index": False}
    print(table.to_string(**fmt))
    print("\npaired differences (model - baseline, same resamples):")
    print(diff_table.to_string(**fmt))
    print(f"\nsplit noise (fitted out-of-fold agreement across {args.repeats} CV repeats):")
    print(noise_table.to_string(**fmt))
    print("\nweights (final):", {k: round(v, 3) for k, v in weights["final"].items()})
    print("weights (95% CI):", {c: f"{weights['ci_low'][c]:.2f}-{weights['ci_high'][c]:.2f}" for c in CHANNELS})
    print(
        "weights (fold mean ± std):",
        {c: f"{weights['fold_mean'][c]:.2f}±{weights['fold_std'][c]:.2f}" for c in CHANNELS},
    )


if __name__ == "__main__":
    main()
