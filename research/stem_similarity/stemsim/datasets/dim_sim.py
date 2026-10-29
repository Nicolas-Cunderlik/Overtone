"""DIM-SIM (Lee et al., ICASSP 2020) - direct triplet votes: "which of song 1
and song 2 sounds more similar to the anchor?", on 3 s clips of Million Song
Dataset tracks. 3,990 triplets, each rated by 5-12 people (39,440 votes); the
authors' "clean" subset is the 879 with >= 90% agreement.

Normalization: the raw votes. The majority pick is the positive; value = its
share of the votes (agreement), weight = total votes; ties are skipped. The
clean subset is flagged in meta (meta["clean"]), so either set can be scored.

Audio: tracks are MSD ids, mapped to artist + title through MSD's
unique_tracks.txt and fetched as Deezer previews by name (`song:` tracks, so
any that are also catalog songs merge with them). Caveat: listeners heard
specific 3 s moments of 7digital previews (start times kept in meta), which no
longer exist; a Deezer preview covers a different 30 s of the song, so this
is song-level evidence, not moment-level.

License: CC BY-NC 4.0 (non-commercial research use).
"""

import csv
import tarfile

from overtone_core.clients.http import download

from ..schema import PERCEPTUAL, TRIPLET, TRIPLET_AGREEMENT, Judgment, Membership, NormalizedDataset, song_track
from .base import DatasetAdapter

ARCHIVE_URL = "https://zenodo.org/records/3889149/files/dim-sim.tar.gz?download=1"
MSD_TRACKS_URL = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt"
RAW_CSV = "dim-sim/raw-dim-sim.csv"
CLEAN_CSV = "dim-sim/clean-dim-sim.csv"
MSD_TRACKS = "msd_unique_tracks.txt"
ROLES = ("anchor", "song1", "song2")


def msd_titles(path, wanted):
    """{msd track id: (artist, title)} for the ids in `wanted`, from unique_tracks.txt
    (one "track_id<SEP>song_id<SEP>artist<SEP>title" line per MSD track)."""
    found = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            track_id, _song_id, artist, title = line.rstrip("\n").split("<SEP>", 3)
            if track_id in wanted:
                found[track_id] = (artist, title)
    return found


class DimSim(DatasetAdapter):
    name = "dim_sim"
    target = PERCEPTUAL
    metric = TRIPLET_AGREEMENT
    license = "CC BY-NC 4.0"
    url = "https://zenodo.org/records/3889149"

    def fetch(self, registry, **_):
        raw = registry.raw_dir(self.name)
        archive = download(ARCHIVE_URL, raw / "dim-sim.tar.gz")
        if not (raw / RAW_CSV).exists():
            with tarfile.open(archive) as tar:
                tar.extractall(raw, filter="data")
        download(MSD_TRACKS_URL, raw / MSD_TRACKS)

    def normalize(self, registry):
        raw = registry.raw_dir(self.name)
        with open(raw / RAW_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(raw / CLEAN_CSV, newline="", encoding="utf-8") as f:
            clean = {row["triplet_id"] for row in csv.DictReader(f)}
        names = msd_titles(raw / MSD_TRACKS, {row[f"{role}_id"] for row in rows for role in ROLES})

        out = NormalizedDataset()
        tracks = {}
        for row in rows:
            votes = int(row["song1_vote"]), int(row["song2_vote"])
            if votes[0] == votes[1] or not all(row[f"{role}_id"] in names for role in ROLES):
                continue
            ids = {}
            for role in ROLES:
                artist, title = names[row[f"{role}_id"]]
                track = song_track(title, artist, msd_track_id=row[f"{role}_id"])
                tracks.setdefault(track.track_id, track)
                ids[role] = track.track_id
            if len(set(ids.values())) < 3:
                continue  # two roles are the same song - says nothing about similarity
            closer, farther = ("song1", "song2") if votes[0] > votes[1] else ("song2", "song1")
            out.judgments.append(
                Judgment(
                    TRIPLET,
                    ids["anchor"],
                    ids[closer],
                    ids[farther],
                    value=max(votes) / sum(votes),
                    weight=sum(votes),
                    meta={
                        "triplet_id": row["triplet_id"],
                        "clean": row["triplet_id"] in clean,
                        "clip_seconds": 3,
                        "msd_starts": {role: float(row[f"{role}_start_seconds"]) for role in ROLES},
                    },
                )
            )
        used = {j.a for j in out.judgments} | {j.b for j in out.judgments} | {j.c for j in out.judgments}
        out.tracks = [tracks[track_id] for track_id in sorted(used)]
        out.members = [Membership(t.track_id, t.meta["msd_track_id"]) for t in out.tracks]
        return out
