"""Feature engineering for the song-similarity model.

Raw audio-feature columns come from the public TidyTuesday Spotify dataset
(see data/README.md). `key` is a pitch class (0-11) and is circular - key 11
(B) is musically adjacent to key 0 (C), so encoding it as a single ordinal
column makes a cosine/Euclidean model treat them as maximally distant
instead. It's encoded as (sin, cos) of its position on the 12-tone circle.

Key is a real but weak genre signal - on this catalog key+mode explains ~1.5%
of genre uncertainty vs. ~8.6% for tempo (rock's E/A major bias is the
clearest case). Two standardized columns would give it double the weight of
any other feature in the distance, so the default variant scales the pair
down via VARIANT_WEIGHTS. Measured with evaluate.py over 20 splits: full
weight costs ~1.4 points of genre precision@3, dropping key loses ~0.1, and a
total weight of 0.25 beats both. Circle-of-fifths and key-signature encodings
were also tried - no better than chromatic at low weight.

Feature sets are named variants rather than a fixed count - which columns
actually help is an empirical question, answered by evaluate.py.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import dedup

RAW_AUDIO_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

FEATURE_VARIANTS = {
    "raw_key_with_duration": [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ],
    "circular_key_with_duration": [
        "danceability",
        "energy",
        "key_sin",
        "key_cos",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ],
    "circular_key_no_duration": [
        "danceability",
        "energy",
        "key_sin",
        "key_cos",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ],
    "raw_key_no_duration": [
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ],
    "circular_key_light": [
        "danceability",
        "energy",
        "key_sin",
        "key_cos",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ],
}

# Per-variant multipliers applied after standardization (default 1). The key
# pair is split so its combined weight is KEY_WEIGHT features' worth:
# sqrt(sin_w^2 + cos_w^2) = KEY_WEIGHT.
KEY_WEIGHT = 0.25
VARIANT_WEIGHTS = {
    "circular_key_light": {"key_sin": KEY_WEIGHT / np.sqrt(2), "key_cos": KEY_WEIGHT / np.sqrt(2)},
}

DEFAULT_VARIANT = "circular_key_light"


def load_catalog(csv_path, dedupe=True):
    """Load the raw playlist-track CSV and collapse it to one row per song.

    The source CSV is a playlist/track join table, so the same track appears
    once per playlist it's on - those rows are first collapsed per track_id.
    Audio features are intrinsic to the track, so the first occurrence is kept.

    The same song can also appear under several track_ids (single vs. album,
    remasters, "feat." credits - see dedup.py). With `dedupe`, those copies are
    collapsed too: the most popular copy becomes the canonical row and every
    copy's id is kept in `alias_track_ids` (";"-joined) so a lookup by any of
    them still resolves. Genre labels (used only for evaluation) are the most
    frequent tag across all of a song's playlist rows.

    The dedup statistics are attached as `catalog.attrs["dedup"]`.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["track_name", "track_artist"])

    agg = {col: "first" for col in RAW_AUDIO_COLUMNS}
    agg.update(track_name="first", track_artist="first", track_popularity="max")
    tracks = df.groupby("track_id", sort=False).agg(agg).reset_index()

    rule_counts = {}
    if dedupe:
        # Scaler here is only for dedup's audio-distance check; it's refit on
        # the deduplicated catalog (or a train split of it) for the model itself
        # Pinned variant: dedup's AUDIO_DUPLICATE_DISTANCE was measured in this space
        X, _ = build_feature_matrix(tracks, variant="circular_key_with_duration")
        groups, rule_counts = dedup.find_duplicate_groups(
            tracks["track_name"].tolist(), tracks["track_artist"].tolist(), X
        )
    else:
        groups = np.arange(len(tracks))
    tracks["song_group"] = groups

    # Canonical copy: most popular, then the plainest (shortest) title
    tracks["_title_len"] = tracks["track_name"].str.len()
    tracks = tracks.sort_values(["song_group", "track_popularity", "_title_len"], ascending=[True, False, True])
    aliases = tracks.groupby("song_group")["track_id"].agg(";".join)

    df["song_group"] = df["track_id"].map(dict(zip(tracks["track_id"], tracks["song_group"], strict=True)))
    labels = df.groupby("song_group").agg(
        playlist_genre=("playlist_genre", lambda s: s.mode().iat[0]),
        playlist_subgenre=("playlist_subgenre", lambda s: s.mode().iat[0]),
    )

    catalog = (
        tracks.drop_duplicates("song_group")
        .drop(columns="_title_len")
        .set_index("song_group")
        .join(labels)
        .assign(alias_track_ids=aliases)
        .reset_index(drop=True)
    )
    catalog.attrs["dedup"] = {"track_ids": len(tracks), "songs": len(catalog), **rule_counts}
    return catalog


def add_derived_columns(catalog):
    """Return a copy of `catalog` with the circular key encoding columns added.

    Spotify's legacy format reports key -1 when no key was detected; that maps
    to the circle's centre (0, 0) - equally far from every key - instead of
    landing on B."""
    catalog = catalog.copy()
    key = catalog["key"].astype(float)
    radians = key * (2 * np.pi / 12)
    detected = key >= 0
    catalog["key_sin"] = np.where(detected, np.sin(radians), 0.0)
    catalog["key_cos"] = np.where(detected, np.cos(radians), 0.0)
    return catalog


def build_feature_matrix(catalog, variant=DEFAULT_VARIANT, scaler=None):
    """Build a standardized feature matrix for the given variant.

    If `scaler` is None, a new StandardScaler is fit on `catalog` and
    returned. Otherwise the provided (already-fit) scaler is reused, e.g.
    fit on a train split then applied to a held-out test split to avoid
    leaking test statistics into the normalization.
    """
    catalog = add_derived_columns(catalog)
    columns = FEATURE_VARIANTS[variant]
    raw = catalog[columns].to_numpy(dtype=float)

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(raw)
    else:
        X = scaler.transform(raw)

    weights = VARIANT_WEIGHTS.get(variant, {})
    if weights:
        X = X * np.array([weights.get(col, 1.0) for col in columns])

    return X, scaler
