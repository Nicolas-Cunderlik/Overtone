"""The common model every dataset is normalized into.

Track       One piece of audio, identified by where that audio comes from - not
            by which dataset mentions it - so datasets that share a recording
            share its audio, stems and embeddings:
              "<source>:<locator>"  audio the dataset ships (e.g. "mtat:8/x.mp3")
              "song:<artist|title>" a song we fetch a preview for by name
                                    (catalog, Last.fm), normalized with the app's
                                    dedup rules so "feat."/remaster variants merge
Membership  A dataset's use of a track: its local id and the excerpt the
            dataset's listeners actually heard (clip_start/clip_end, seconds).
Judgment    One ground-truth statement, of one of two kinds:
              triplet  sim(a, b) > sim(a, c) - anchor a, positive b, negative c
              pair     sim(a, b) = value, on a 0-1 scale
            `weight` is how much evidence backs it (votes, raters); `group`
            ties judgments that are scored together (e.g. one Last.fm seed's
            ranked list); `facet` says what was compared - the whole song
            ("overall") or one part of it (a stem: "drums", "bass", "vocals",
            "other") or one musical attribute ("melody", "harmony", "rhythm",
            "voice", "timbre"); `meta` keeps whatever else the rater said (reason tags,
            free-text notes) for analysis.

Adapters also declare a target (what kind of similarity the people behind the
data were expressing) and a metric (how a model is scored against it), so
results are always reported per dataset and never averaged across targets.
"""

from dataclasses import dataclass, field

from overtone.ml import dedup

TRIPLET = "triplet"
PAIR = "pair"
JUDGMENT_KINDS = (TRIPLET, PAIR)

# Targets
PERCEPTUAL = "perceptual"  # people listened and judged how alike the audio sounds
BEHAVIORAL = "behavioral"  # derived from listening behaviour (co-listening, playlists)
SEEDS = "seeds"  # no judgments - a pool of tracks other datasets build on

# Metrics
TRIPLET_AGREEMENT = "triplet_agreement"  # share of triplets the model orders the same way
SPEARMAN = "spearman"  # rank correlation of model similarity with pair scores
NDCG = "ndcg"  # ranking quality per group, pair values as graded relevance

# Facets - what a judgment compared
OVERALL = "overall"
STEMS = ("drums", "bass", "vocals", "other")  # Demucs htdemucs stems
# Embedding channels; fitted stem weights are positional, so every user takes this order
MIX = "mix"
CHANNELS = (MIX, *STEMS)
ATTRIBUTE_FACETS = ("melody", "harmony", "rhythm", "voice", "timbre")  # MATCHA's musical attributes


@dataclass(frozen=True)
class Track:
    track_id: str
    audio_source: str  # which audio resolver fetches it: "deezer", "mtat", "fma", ...
    audio_ref: str = ""  # source-specific locator ("" = look up by title/artist)
    title: str | None = None
    artist: str | None = None
    meta: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Membership:
    track_id: str
    local_id: str
    clip_start: float | None = None
    clip_end: float | None = None


@dataclass(frozen=True)
class Judgment:
    kind: str
    a: str
    b: str
    c: str | None = None
    value: float = 1.0
    weight: float = 1.0
    group: str | None = None
    facet: str = OVERALL
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.kind not in JUDGMENT_KINDS:
            raise ValueError(f"unknown judgment kind {self.kind!r}")
        if (self.kind == TRIPLET) != (self.c is not None):
            raise ValueError("a triplet needs c (the negative); a pair must not have one")


@dataclass
class NormalizedDataset:
    tracks: list[Track] = field(default_factory=list)
    members: list[Membership] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)


def song_track_id(title, artist):
    """Id for a song identified by name, shared across datasets."""
    artist_key, title_key = dedup.song_key(title, artist)
    return f"song:{artist_key}|{title_key}"


def song_track(title, artist, **meta):
    """A song whose audio is found by name (a Deezer preview)."""
    return Track(song_track_id(title, artist), "deezer", title=title, artist=artist, meta=meta)
