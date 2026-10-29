"""MATCHA (Batlle-Roca et al. 2026) - attribute-level triplets judged by 83
music experts: for a reference excerpt and samples A and B, which is closer
on melody, harmony, rhythm, voice and timbre (or "Neither"). 300 cases: 150
human (plagiarism pairs, cover versions), 150 AI-generated.

Normalization: one triplet per case and attribute with a clear A/B majority -
facet = the attribute, value = agreement share, weight = raters. "Neither"
decisions are skipped (no ordering); ties are already "Neither" upstream.
Control cases are kept but flagged in meta.

Audio: NOT AVAILABLE YET. Human cases point at YouTube excerpts (link +
timestamp, kept per membership); AI cases are unpublished. The authors will
release materials on acceptance - until then these tracks resolve to
"unavailable" and their judgments simply aren't scored. Annotations: MIT.
"""

import csv

from overtone_core.clients.http import download

from ..schema import PERCEPTUAL, TRIPLET, TRIPLET_AGREEMENT, Judgment, Membership, NormalizedDataset, Track
from .base import DatasetAdapter

BASE_URL = "https://raw.githubusercontent.com/roserbatlleroca/matcha/HEAD"
CASES = "raw_cases.csv"
DECISIONS = "annotations_per_case.csv"
ATTRIBUTES = ("melody", "harmony", "rhythm", "voice", "timbre")
ROLES = {"ref": "ref", "A": "sampleA", "B": "sampleB"}


def _seconds(stamp):
    minutes, seconds = stamp.split(":")
    return int(minutes) * 60 + float(seconds)


def _window(timestamp):
    """'00:11-00:19' -> (11.0, 19.0); None when absent or unparseable."""
    try:
        start, end = timestamp.split("-")
        return _seconds(start), _seconds(end)
    except (AttributeError, ValueError):
        return None, None


class Matcha(DatasetAdapter):
    name = "matcha"
    target = PERCEPTUAL
    metric = TRIPLET_AGREEMENT
    license = "MIT (annotations); audio not yet released"
    url = "https://github.com/roserbatlleroca/matcha"

    def fetch(self, registry, **_):
        raw = registry.raw_dir(self.name)
        download(f"{BASE_URL}/metadata/{CASES}", raw / CASES)
        download(f"{BASE_URL}/annotations/{DECISIONS}", raw / DECISIONS)

    def normalize(self, registry):
        raw = registry.raw_dir(self.name)
        with open(raw / CASES, newline="", encoding="utf-8") as f:
            cases = {row["case_idraw_"]: row for row in csv.DictReader(f)}
        with open(raw / DECISIONS, newline="", encoding="utf-8") as f:
            decisions = list(csv.DictReader(f))

        out = NormalizedDataset()
        members = {}
        for decision in decisions:
            case = cases.get(decision["case_id"])
            if case is None:
                continue
            ids = {}
            for role, prefix in ROLES.items():
                track_id = f"matcha:{case[f'{prefix}_id']}"
                ids[role] = track_id
                start, end = _window(case[f"{prefix}_timestamp"])
                track = Track(
                    track_id,
                    "youtube" if case["case_origin"] == "human" else "matcha_ai",
                    audio_ref=case[f"{prefix}_link"],
                    title=case[f"{prefix}_title"],
                    meta={"section": case["section"]},
                )
                members[track_id] = (track, Membership(track_id, case[f"{prefix}_id"], start, end))
            for attribute in ATTRIBUTES:
                choice = decision[f"{attribute}_decision"]
                if choice not in ("A", "B"):
                    continue
                other = "B" if choice == "A" else "A"
                out.judgments.append(
                    Judgment(
                        TRIPLET,
                        ids["ref"],
                        ids[choice],
                        ids[other],
                        value=float(decision[f"{attribute}_agreement"]) / 100,
                        weight=float(decision[f"{attribute}_n"]),
                        facet=attribute,
                        meta={
                            "case": decision["case_id"],
                            "section": case["section"],
                            "control": case["is_control"] == "1",
                        },
                    )
                )
        for track, membership in members.values():
            out.tracks.append(track)
            out.members.append(membership)
        return out
