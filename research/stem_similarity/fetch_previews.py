"""Pilot steps 1-2: sample catalog songs across subgenres and download their
Deezer 30 s previews.

Subgenre stratification is only for diversity - it is not used as ground
truth. A match must be by the same artist, have a similar title (dedup.py's
rules) and a duration within DURATION_TOLERANCE_S of the catalog's, so a
live version or remix isn't silently swapped in.

Deezer preview URLs are signed and expire, so the MP3s are cached under
audio/ (gitignored, never redistributed).

Run with the app venv (overtone is installed there):
  .venv/Scripts/python research/stem_similarity/fetch_previews.py
"""

import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from overtone.ml import dedup, features
from overtone.ml.paths import CSV_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "audio")
OUT_PATH = os.path.join(HERE, "pilot_songs.csv")

N_SONGS = 300
OVERSAMPLE = 2.0  # candidates per slot, to absorb Deezer misses
DURATION_TOLERANCE_S = 12
REQUEST_INTERVAL_S = 0.15  # Deezer allows ~50 requests / 5 s
SEED = 0


def deezer_get(url):
    time.sleep(REQUEST_INTERVAL_S)
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.load(resp)


def find_preview(title, artist, duration_s):
    query = f"{artist} {dedup.base_title(title)}"
    url = "https://api.deezer.com/search?" + urllib.parse.urlencode({"q": query, "limit": 10})
    want_artist = dedup.normalize_artist(artist)
    for hit in deezer_get(url).get("data", []):
        if not hit.get("preview"):
            continue
        if dedup.normalize_artist(hit["artist"]["name"]) != want_artist:
            continue
        if not dedup.titles_similar(title, hit["title"]):
            continue
        if abs(hit["duration"] - duration_s) > DURATION_TOLERANCE_S:
            continue
        return hit
    return None


def sample_candidates(catalog):
    per_subgenre = int(N_SONGS / catalog["playlist_subgenre"].nunique() * OVERSAMPLE) + 1
    return (
        catalog.groupby("playlist_subgenre").sample(n=per_subgenre, random_state=SEED).sample(frac=1, random_state=SEED)
    )


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    catalog = features.load_catalog(CSV_PATH)
    dates = (
        pd.read_csv(CSV_PATH, usecols=["track_id", "track_album_release_date"])
        .drop_duplicates("track_id")
        .set_index("track_id")["track_album_release_date"]
    )
    catalog["release_year"] = pd.to_numeric(catalog["track_id"].map(dates).str[:4], errors="coerce")

    quota = {s: N_SONGS // catalog["playlist_subgenre"].nunique() + 1 for s in catalog["playlist_subgenre"].unique()}
    rows, misses = [], 0
    for _, song in sample_candidates(catalog).iterrows():
        if len(rows) >= N_SONGS:
            break
        if quota[song.playlist_subgenre] == 0:
            continue
        try:
            hit = find_preview(song.track_name, song.track_artist, song.duration_ms / 1000)
        except Exception as exc:  # network hiccup - skip this candidate
            print(f"  error for {song.track_name!r}: {exc}")
            hit = None
        if hit is None:
            misses += 1
            continue
        path = os.path.join(AUDIO_DIR, f"{song.track_id}.mp3")
        if not os.path.exists(path):
            urllib.request.urlretrieve(hit["preview"], path)
        quota[song.playlist_subgenre] -= 1
        rows.append(
            {
                "track_id": song.track_id,
                "track_name": song.track_name,
                "track_artist": song.track_artist,
                "playlist_genre": song.playlist_genre,
                "playlist_subgenre": song.playlist_subgenre,
                "release_year": song.release_year,
                "deezer_id": hit["id"],
                "deezer_title": hit["title"],
                "isrc": hit.get("isrc", ""),
            }
        )
        if len(rows) % 25 == 0:
            print(f"{len(rows)} matched, {misses} misses")

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Done: {len(rows)} songs with previews ({misses} candidates had no match) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
