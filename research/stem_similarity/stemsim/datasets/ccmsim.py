"""CCMSim - cross-cultural music similarity ratings (Xaris et al., TISMIR).

Listeners rated 20 s excerpt pairs for overall musical similarity (also
cultural and "would recommend" similarity, kept for later). The published
scores are already normalized to 0-1. Most pairs have a single rater.

Default subset: Western music only (MagnaTagATune + FMA), the part closest to
a Spotify catalog - 46 pairs / 55 ratings, so a sanity check, not a training
target. `CCMSim(cultures=None)` keeps all nine source datasets (1,130 pairs).

Audio: MagnaTagATune excerpts reuse the MagnaTagATune clip tracks (shared
audio and embeddings); the listeners heard seconds survey_start-survey_end of
each clip, recorded per membership. FMA tracks need the FMA archive, which
has no audio source yet, so their pairs stay unscored until one is added.
"""

import csv

from overtone_core.clients.http import download

from ..schema import PAIR, PERCEPTUAL, SPEARMAN, Judgment, Membership, NormalizedDataset, Track
from .base import DatasetAdapter
from .magnatagatune import clip_track

BASE_URL = "https://raw.githubusercontent.com/pxaris/CCMSim/HEAD/data/annotations"
RATINGS = "human_similarity.csv"
MAPPING = "audio_annotations_mapping.csv"
WESTERN = ("magnatagatune", "fma")


def _track(dataset, audio_path):
    if dataset == "magnatagatune":
        return clip_track(audio_path)
    return Track(f"{dataset}:{audio_path}", dataset, audio_ref=audio_path)


class CCMSim(DatasetAdapter):
    name = "ccmsim"
    target = PERCEPTUAL
    metric = SPEARMAN
    license = "CC-BY-4.0 (annotations); audio under each source dataset's license"
    url = "https://github.com/pxaris/CCMSim"

    def __init__(self, cultures=WESTERN):
        self.cultures = set(cultures) if cultures else None

    def fetch(self, registry, **_):
        raw = registry.raw_dir(self.name)
        for name in (RATINGS, MAPPING):
            download(f"{BASE_URL}/{name}", raw / name)

    def normalize(self, registry):
        raw = registry.raw_dir(self.name)
        # The mapping file has a few bytes that aren't valid UTF-8 (in paths we don't use)
        with open(raw / MAPPING, newline="", encoding="utf-8", errors="replace") as f:
            clips = {row["survey_audio_id"]: row for row in csv.DictReader(f)}
        with open(raw / RATINGS, newline="", encoding="utf-8") as f:
            ratings = list(csv.DictReader(f))

        out = NormalizedDataset()
        members = {}
        for row in ratings:
            pair = [clips[clip_id] for clip_id in row["audio_pair"].split("-")]
            if self.cultures and not all(clip["dataset"] in self.cultures for clip in pair):
                continue
            tracks = [_track(clip["dataset"], clip["audio_file_path"]) for clip in pair]
            out.judgments.append(
                Judgment(
                    PAIR,
                    tracks[0].track_id,
                    tracks[1].track_id,
                    value=float(row["overall_music_similarity"]),
                    weight=float(row["num_annotators"]),
                )
            )
            for track, clip in zip(tracks, pair, strict=True):
                members[track.track_id] = (track, clip)

        for track_id, (track, clip) in sorted(members.items()):
            out.tracks.append(track)
            out.members.append(
                Membership(
                    track_id,
                    clip["survey_audio_id"],
                    clip_start=float(clip["survey_start_second"]),
                    clip_end=float(clip["survey_end_second"]),
                )
            )
        return out
