"""Pilot step 7: fit stem weights to the listener's ABX answers and compare
similarity measures by how often they agree with the listener (plan
section 5).

For a full-mix triplet (X, A, B) every measure produces a score
sim(X,A) - sim(X,B); it "agrees" when the sign matches the listener's pick.
Ties ("can't tell") are reported but not scored.

Per-stem similarity is the cosine of MuQ-MuLan embeddings, z-scored per stem
over all song pairs so stems are on one scale. Silence gating: if a stem is
silent in exactly one of the two songs the pair scores -1 (one has bass, the
other doesn't); silent in both scores 0 (no information either way).

Run: research/stem_similarity/.venv/Scripts/python research/stem_similarity/analyze.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import triplets
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent.parent / "models" / "similarity"  # the app's trained model (overtone.ml.paths)
N_SPLITS = 100
TEST_FRACTION = 0.3
L2 = 1.0
SEED = 0


def zscored_stem_sims(emb, rms):
    """(stems, n, n) gated, z-scored cosine similarity."""
    sims = triplets.cosine_matrices(emb)
    audible = rms > triplets.SILENT_DB
    out = np.empty_like(sims)
    iu = np.triu_indices(sims.shape[1], k=1)
    for s in range(sims.shape[0]):
        both = audible[:, s][:, None] & audible[:, s][None, :]
        vals = sims[s][iu][both[iu]]
        z = (sims[s] - vals.mean()) / (vals.std() + 1e-9)
        one = audible[:, s][:, None] ^ audible[:, s][None, :]
        out[s] = np.where(both, z, np.where(one, -1.0, 0.0))
    return out


def fit_weights(D, y, l2=L2):
    """Non-negative logistic regression without intercept: P(A) = sigmoid(D @ w)."""
    sign = 2 * y - 1

    def loss(w):
        m = sign * (D @ w)
        return np.logaddexp(0, -m).mean() + l2 * (w @ w) / len(y)

    def grad(w):
        m = sign * (D @ w)
        g = -(sign / (1 + np.exp(m)))[:, None] * D
        return g.mean(0) + 2 * l2 * w / len(y)

    res = minimize(loss, np.full(D.shape[1], 0.5), jac=grad, method="L-BFGS-B", bounds=[(0, None)] * D.shape[1])
    return res.x


def agreement(score, y):
    decided = score != 0
    return float(np.mean((score[decided] > 0) == (y[decided] == 1))) if decided.any() else float("nan")


def spotify_similarity(track_ids):
    """Cosine similarity in the shipped model's embedding and its raw features."""
    meta = pd.read_csv(ARTIFACTS / "track_meta.csv")
    row = {}
    for i, (tid, aliases) in enumerate(zip(meta.track_id, meta.alias_track_ids, strict=True)):
        row[tid] = i
        for a in str(aliases).split(";"):
            row.setdefault(a, i)
    idx = [row[t] for t in track_ids]
    out = {}
    for name, file in (
        ("spotify features (raw kNN)", "features.joblib"),
        ("shipped model (features + genre)", "embedding.joblib"),
    ):
        Z = joblib.load(ARTIFACTS / file)[idx]
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        out[name] = Z @ Z.T
    return out


def main():
    songs = pd.read_csv(HERE / "pilot_songs.csv")
    npz = np.load(HERE / "embeddings" / "pilot_embeddings.npz", allow_pickle=True)
    stems = list(npz["stems"])
    pos = {t: i for i, t in enumerate(npz["track_ids"])}
    ratings = triplets.load_jsonl(HERE / "ratings.jsonl")
    if not ratings:
        sys.exit("No ratings yet - run rate_server.py first.")
    r = pd.DataFrame(ratings)
    report = {"n_ratings": len(r), "tie_rate": float((r.choice == "tie").mean())}
    print(f"{len(r)} ratings, {report['tie_rate']:.0%} 'can't tell'")

    # Self-consistency (noise ceiling): repeats swap A/B, so consistent = flipped letter
    queue = {t["id"]: t for t in triplets.load_jsonl(HERE / "triplet_queue.jsonl")}
    by_id = r.set_index("id")
    flip = {"a": "b", "b": "a", "tie": "tie"}
    reps = [
        (by_id.loc[i].choice, by_id.loc[queue[i]["repeat_of"]].choice)
        for i in r.id
        if queue[i]["kind"] == "repeat" and queue[i]["repeat_of"] in by_id.index
    ]
    if reps:
        report["self_consistency"] = float(np.mean([flip[a] == b for a, b in reps]))
        print(f"Self-consistency on {len(reps)} repeats: {report['self_consistency']:.0%} (noise ceiling)")

    S_ex = zscored_stem_sims(npz["emb_excerpt"], npz["rms_excerpt"])
    S_full = zscored_stem_sims(npz["emb_full"], npz["rms_full"])
    mfcc = npz["mfcc_excerpt"]  # standardized per stem and coefficient, so MFCC 0 (loudness) can't dominate
    S_mfcc = zscored_stem_sims((mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-9), npz["rms_excerpt"])
    spotify = spotify_similarity(list(npz["track_ids"]))
    years = songs.set_index("track_id").reindex(npz["track_ids"]).release_year.to_numpy()
    subgenre = songs.set_index("track_id").reindex(npz["track_ids"]).playlist_subgenre.to_numpy()

    mix = r[(r.play == "mix") & (r.choice != "tie")].reset_index(drop=True)
    x, a, b = (mix[c].map(pos).to_numpy() for c in ("x", "a", "b"))
    y = (mix.choice == "a").astype(int).to_numpy()
    D = (S_ex[:, x, a] - S_ex[:, x, b]).T  # (n, stems)
    year_term = -(np.abs(years[x] - years[a]) - np.abs(years[x] - years[b])) / 10
    year_term = np.nan_to_num(year_term)
    print(f"\n{len(mix)} decisive full-mix judgments used for scoring\n")

    fixed = {
        "subgenre label match": (subgenre[x] == subgenre[a]).astype(float) - (subgenre[x] == subgenre[b]),
        **{name: M[x, a] - M[x, b] for name, M in spotify.items()},
        "MFCC mix only": S_mfcc[0, x, a] - S_mfcc[0, x, b],
        "MFCC stems, equal weights": (S_mfcc[:, x, a] - S_mfcc[:, x, b]).mean(0),
        "MuQ mix only (full 30 s)": S_full[0, x, a] - S_full[0, x, b],
        "MuQ mix only (rated excerpt)": D[:, 0],
        "MuQ stems, equal weights": D.mean(1),
    }
    for s, name in enumerate(stems[1:], start=1):
        fixed[f"MuQ {name} alone"] = D[:, s]

    rng = np.random.default_rng(SEED)
    n_test = max(1, int(len(y) * TEST_FRACTION))
    results = {name: [] for name in fixed}
    results["MuQ stems, fitted weights"] = []
    results["MuQ stems fitted + release year"] = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(y))
        te, tr = perm[:n_test], perm[n_test:]
        for name, score in fixed.items():
            results[name].append(agreement(score[te], y[te]))
        w = fit_weights(D[tr], y[tr])
        results["MuQ stems, fitted weights"].append(agreement(D[te] @ w, y[te]))
        Dy = np.c_[D, year_term]
        wy = fit_weights(Dy[tr], y[tr])
        results["MuQ stems fitted + release year"].append(agreement(Dy[te] @ wy, y[te]))

    print(f"{'measure':40s} agreement with you (held-out, {N_SPLITS} splits)")
    print("-" * 78)
    report["agreement"] = {}
    for name, vals in sorted(results.items(), key=lambda kv: -np.nanmean(kv[1])):
        m, s = np.nanmean(vals), np.nanstd(vals)
        report["agreement"][name] = [float(m), float(s)]
        print(f"{name:40s} {m:6.1%} ± {s:4.1%}")

    boot = np.array([fit_weights(D[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(300))])
    full_w = fit_weights(D, y)
    print("\nFitted stem weights (all data; 95% bootstrap interval)")
    report["weights"] = {}
    for s, name in enumerate(stems):
        lo, hi = np.percentile(boot[:, s], [2.5, 97.5])
        report["weights"][name] = [float(full_w[s]), float(lo), float(hi)]
        print(f"  {name:8s} {full_w[s]:5.2f}  [{lo:4.2f}, {hi:4.2f}]")

    stem_only = r[(r.kind == "stem_only") & (r.choice != "tie")]
    if len(stem_only):
        print("\nStem reliability (isolated-stem triplets: you agreed with the stem embedding)")
        report["stem_reliability"] = {}
        for stem, g in stem_only.groupby("play"):
            acc = float((g.choice == g.near).mean())
            report["stem_reliability"][stem] = [acc, len(g)]
            print(f"  {stem:8s} {acc:5.0%}  (n={len(g)})")

    tagged = r[r.tags.map(len) > 0]
    if len(tagged):
        counts = pd.Series([t for ts in tagged.tags for t in ts]).value_counts()
        report["tags"] = counts.to_dict()
        print(f"\nReasons you tagged ({len(tagged)} judgments with tags):")
        print(counts.to_string())
    notes = r[r.note.str.len() > 0]
    report["n_notes"] = len(notes)

    out = HERE / "pilot_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
