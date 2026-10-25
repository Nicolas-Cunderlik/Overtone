"""The app's song catalog (TidyTuesday Spotify songs, deduplicated by
overtone.ml.features.load_catalog) - ~25.6k songs with no judgments.

It's the seed pool: Last.fm similarity is crawled from these songs, and its
tracks are the candidates the app recommends from, so every model is
ultimately evaluated on them. Audio comes from Deezer previews by name.
"""

import os

from overtone.ml import features
from overtone.ml.paths import CSV_PATH

from ..schema import SEEDS, Membership, NormalizedDataset, song_track
from .base import DatasetAdapter


class SpotifyCatalog(DatasetAdapter):
    name = "spotify_catalog"
    target = SEEDS
    metric = None
    license = "CC0 (TidyTuesday 2020-01-21)"
    url = "https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21"

    def normalize(self, registry):
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"{CSV_PATH} missing - run `python -m overtone.ml.download_dataset` first")
        catalog = features.load_catalog(CSV_PATH)
        out = NormalizedDataset()
        for row in catalog.itertuples():
            track = song_track(
                row.track_name,
                row.track_artist,
                spotify_track_id=row.track_id,
                duration_s=row.duration_ms / 1000,
            )
            out.tracks.append(track)
            out.members.append(Membership(track.track_id, row.track_id))
        return out
