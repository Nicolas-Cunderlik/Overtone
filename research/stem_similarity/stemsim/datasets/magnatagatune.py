"""MagnaTagATune "odd one out" similarity votes (TagATune bonus game).

Players heard three ~29 s clips and picked the one least like the others.
comparisons_final.csv holds 533 rows - 346 unique clip triples, some listed
in several orders - with each clip's vote count.

Normalization: votes for the same three clips are summed across orderings.
The clip with the most votes is the outlier; the other two are each other's
closer match, giving two triplets: (p, q, outlier) and (q, p, outlier).
value = the outlier's share of the votes (how clear the call was), weight =
total votes. Triples with no votes, or tied for most votes, say nothing
reliable and are skipped.

Audio: three split-zip parts (3.3 GB, 32 kbps MP3), joined into one archive.
Only `fetch(with_audio=True)` downloads them; the audio stage extracts clips.
"""

import csv
import zipfile
from collections import defaultdict

from ..clients.http import download
from ..schema import PERCEPTUAL, TRIPLET, TRIPLET_AGREEMENT, Judgment, Membership, NormalizedDataset, Track
from .base import DatasetAdapter

BASE_URL = "https://mirg.city.ac.uk/datasets/magnatagatune"
COMPARISONS = "comparisons_final.csv"
AUDIO_PARTS = ["mp3.zip.001", "mp3.zip.002", "mp3.zip.003"]
ARCHIVE = "mp3.zip"


def clip_track(mp3_path, clip_id=None):
    """The shared track for an MTAT clip - other datasets (CCMSim) reuse it."""
    stem = mp3_path.rsplit("/", 1)[-1].removesuffix(".mp3")
    parts = stem.split("-")  # artist-album-tracknum-title-start-end, words joined by "_"
    pretty = [p.replace("_", " ") for p in parts]
    return Track(
        f"mtat:{mp3_path}",
        "mtat",
        audio_ref=mp3_path,
        title=pretty[3] if len(parts) == 6 else None,
        artist=pretty[0],
        meta={"mtat_clip_id": clip_id} if clip_id else {},
    )


def archive_path(registry):
    return registry.raw_dir(MagnaTagATune.name) / ARCHIVE


class MagnaTagATune(DatasetAdapter):
    name = "magnatagatune"
    target = PERCEPTUAL
    metric = TRIPLET_AGREEMENT
    license = "Research use; cite Law et al. ISMIR 2009 (no explicit license on the dataset page)"
    url = f"{BASE_URL}/index1.html"

    def fetch(self, registry, with_audio=False, progress=None, **_):
        raw = registry.raw_dir(self.name)
        download(f"{BASE_URL}/{COMPARISONS}", raw / COMPARISONS)
        if with_audio and not (raw / ARCHIVE).exists():
            for part in AUDIO_PARTS:
                download(f"{BASE_URL}/{part}", raw / part, progress=progress)
            # The parts are one zip split by byte ranges; joined, it's a normal archive
            joined = raw / (ARCHIVE + ".part")
            with open(joined, "wb") as out:
                for part in AUDIO_PARTS:
                    with open(raw / part, "rb") as src:
                        while chunk := src.read(1 << 24):
                            out.write(chunk)
            with zipfile.ZipFile(joined) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError("MagnaTagATune archive is corrupt; delete the parts and re-fetch")
            joined.replace(raw / ARCHIVE)
            for part in AUDIO_PARTS:
                (raw / part).unlink()

    def normalize(self, registry):
        with open(registry.raw_dir(self.name) / COMPARISONS, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        votes = defaultdict(lambda: defaultdict(int))  # frozenset of clip ids -> clip id -> votes
        paths = {}
        for row in rows:
            clips = [row[f"clip{i}_id"] for i in (1, 2, 3)]
            for i, clip in enumerate(clips, start=1):
                votes[frozenset(clips)][clip] += int(row[f"clip{i}_numvotes"])
                paths[clip] = row[f"clip{i}_mp3_path"]

        out = NormalizedDataset()
        used = set()
        for clip_votes in votes.values():
            total = sum(clip_votes.values())
            ranked = sorted(clip_votes.items(), key=lambda kv: kv[1], reverse=True)
            if total == 0 or ranked[0][1] == ranked[1][1]:
                continue
            outlier = ranked[0][0]
            p, q = sorted(clip for clip, _ in ranked[1:])
            ids = {clip: f"mtat:{paths[clip]}" for clip in (p, q, outlier)}
            share = clip_votes[outlier] / total
            out.judgments += [
                Judgment(TRIPLET, ids[p], ids[q], ids[outlier], value=share, weight=total),
                Judgment(TRIPLET, ids[q], ids[p], ids[outlier], value=share, weight=total),
            ]
            used |= {p, q, outlier}

        for clip in sorted(used):
            out.tracks.append(clip_track(paths[clip], clip))
            out.members.append(Membership(f"mtat:{paths[clip]}", clip))
        return out
