"""Each adapter's normalize() on small fixture files - no network."""

import json

import pytest
from stemsim.datasets.ccmsim import MAPPING, RATINGS, CCMSim
from stemsim.datasets.lastfm import LastFm, _cache_name
from stemsim.datasets.magnatagatune import COMPARISONS, MagnaTagATune
from stemsim.schema import PAIR, TRIPLET, song_track_id

MTAT_HEADER = (
    "clip1_id\tclip2_id\tclip3_id\tclip1_numvotes\tclip2_numvotes\tclip3_numvotes\t"
    "clip1_mp3_path\tclip2_mp3_path\tclip3_mp3_path\n"
)


def mtat_row(ids, votes):
    return "\t".join([*ids, *map(str, votes), *(f"0/{i}.mp3" for i in ids)]) + "\n"


def test_magnatagatune_sums_votes_across_orderings_and_skips_unclear_triples(registry):
    (registry.raw_dir("magnatagatune") / COMPARISONS).write_text(
        MTAT_HEADER
        + mtat_row(["a", "b", "c"], [3, 1, 6])
        + mtat_row(["b", "c", "a"], [0, 2, 1])  # same three clips, listed in another order
        + mtat_row(["d", "e", "f"], [2, 2, 0])  # tied for most votes - no clear outlier
        + mtat_row(["g", "h", "i"], [0, 0, 0]),  # nobody voted
        encoding="utf-8",
    )
    out = MagnaTagATune().normalize(registry)

    a, b, c = (f"mtat:0/{x}.mp3" for x in "abc")
    # c got 8 of 13 votes as the odd one out, so a and b are each other's closer match
    assert {(j.kind, j.a, j.b, j.c) for j in out.judgments} == {(TRIPLET, a, b, c), (TRIPLET, b, a, c)}
    assert all(j.value == pytest.approx(8 / 13) and j.weight == 13 for j in out.judgments)
    assert sorted(t.track_id for t in out.tracks) == [a, b, c]


def write_ccmsim(registry):
    raw = registry.raw_dir("ccmsim")
    (raw / MAPPING).write_text(
        "survey_audio_id,dataset,audio_file_path,survey_start_second,survey_end_second\n"
        "1_1,magnatagatune,0/x.mp3,4,24\n"
        "1_2,magnatagatune,0/y.mp3,4,24\n"
        "2_1,fma,030/030074.mp3,10,30\n"
        "3_1,carnatic,c.mp3,0,20\n",
        encoding="utf-8",
    )
    (raw / RATINGS).write_text(
        "audio_pair,overall_music_similarity,num_annotators\n1_1-1_2,0.5,2\n1_1-2_1,0.25,1\n1_1-3_1,0.9,1\n",
        encoding="utf-8",
    )


def test_ccmsim_keeps_western_pairs_and_shares_magnatagatune_tracks(registry):
    write_ccmsim(registry)
    out = CCMSim().normalize(registry)

    assert [(j.kind, j.a, j.b, j.value, j.weight) for j in out.judgments] == [
        (PAIR, "mtat:0/x.mp3", "mtat:0/y.mp3", 0.5, 2.0),
        (PAIR, "mtat:0/x.mp3", "fma:030/030074.mp3", 0.25, 1.0),
    ]
    # Same id as the MagnaTagATune adapter gives the clip, so audio is shared;
    # the excerpt the listeners heard is kept per membership
    clip = next(m for m in out.members if m.track_id == "mtat:0/x.mp3")
    assert (clip.clip_start, clip.clip_end) == (4.0, 24.0)


def test_ccmsim_all_cultures(registry):
    write_ccmsim(registry)
    assert len(CCMSim(cultures=None).normalize(registry).judgments) == 3


def test_lastfm_collapses_versions_and_skips_the_seed_itself(registry):
    seed = song_track_id("Karma Police", "Radiohead")
    similar = [
        {"name": "No Surprises", "artist": {"name": "Radiohead"}, "match": "0.8"},
        {"name": "No Surprises - Remastered 2009", "artist": {"name": "Radiohead"}, "match": "0.6"},
        {"name": "Karma Police - Live", "artist": {"name": "Radiohead"}, "match": "0.9"},  # a different recording
        {"name": "Karma Police - Remastered", "artist": {"name": "Radiohead"}, "match": "0.95"},  # the seed itself
        {"name": "Untitled", "artist": {}, "match": "0.5"},  # no artist - unusable
    ]
    raw = registry.raw_dir("lastfm")
    (raw / _cache_name(seed)).write_text(json.dumps({"seed": seed, "similar": similar}), encoding="utf-8")
    missing = song_track_id("Obscure", "Nobody")
    (raw / _cache_name(missing)).write_text(json.dumps({"seed": missing, "similar": []}), encoding="utf-8")

    out = LastFm().normalize(registry)

    pairs = {(j.b, j.value) for j in out.judgments}
    assert pairs == {
        (song_track_id("No Surprises", "Radiohead"), 0.8),
        (song_track_id("Karma Police - Live", "Radiohead"), 0.9),
    }
    assert all(j.kind == PAIR and j.a == seed and j.group == seed for j in out.judgments)
    assert missing not in {m.track_id for m in out.members}


def write_listener_abx(registry, rows):
    from stemsim.datasets.listener_abx import PILOT_SONGS, RATINGS

    raw = registry.raw_dir("listener_abx")
    (raw / PILOT_SONGS).write_text(
        "track_id,track_name,track_artist,deezer_title\n"
        "x,Song X,Artist,Song X\ny,Song Y,Artist,Song Y (Some Remix)\nz,Song Z,Artist,Song Z\n",
        encoding="utf-8",
    )
    (raw / RATINGS).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def rating(rating_id, x="song:x", a="song:a", b="song:b", source="deezer", window=None, **changes):
    audio = {clip: {"source": source, "ref": f"audio/{track.split(':')[1]}.mp3"} for clip, track in
             (("x", x), ("a", a), ("b", b))}  # fmt: skip
    row = {
        "v": 2, "id": rating_id, "rater_id": "rater2", "session": "s1", "design": "neighbourhood",
        "split": "fit", "x": x, "a": a, "b": b, "play": "mix", "audio": audio,
        "window": window, "choice": "b", "strength": 1, "strength_source": "rater", "tags": ["bass"],
        "note": "", "bad_clip": None, "timing": {"total_ms": 40000},
    }  # fmt: skip
    return row | changes


def pilot_rating(rating_id, **changes):
    return rating(
        rating_id, x="pilot:x", a="pilot:y", b="pilot:z", source="pilot", window=[10.0, 20.0],
        session="pilot", rater_id="nicolas", split="validation", design="near_far", **changes,
    )  # fmt: skip


def test_listener_abx_orients_triplets_by_choice_and_keeps_notes(registry):
    from stemsim.datasets.listener_abx import ListenerAbx

    write_listener_abx(
        registry,
        [
            pilot_rating("pilot-01", note="same sort of bass"),
            pilot_rating("pilot-02", play="drums", choice="a", strength=None, strength_source=None),
            pilot_rating("pilot-03", choice="tie", strength=None, strength_source=None),
        ],
    )

    out = ListenerAbx().normalize(registry)

    assert [(j.a, j.b, j.c, j.facet) for j in out.judgments] == [
        ("pilot:x", "pilot:z", "pilot:y", "overall"),  # chose B -> X is closer to B
        ("pilot:x", "pilot:y", "pilot:z", "drums"),  # isolated-stem rating -> that stem's facet
    ]
    assert out.judgments[0].meta["note"] == "same sort of bass"
    assert out.judgments[1].meta["strength"] is None  # a pilot rating with no note to infer it from
    assert {(m.local_id, m.clip_start) for m in out.members} == {("x", 10.0), ("y", 10.0), ("z", 10.0)}
    y = next(t for t in out.tracks if t.track_id == "pilot:y")
    assert y.audio_source == "pilot" and y.audio_ref == "audio/y.mp3"
    assert y.meta["version_differs"]  # the pilot heard a remix, not the catalog song


def test_listener_abx_registry_tracks_keep_rater_split_strength(registry):
    from stemsim.datasets.listener_abx import ListenerAbx

    bad = {"clips": ["a"], "reason": "silence"}
    write_listener_abx(
        registry,
        [rating("s1-01"), rating("s1-02", choice="bad_clip", strength=None, strength_source=None, bad_clip=bad)],
    )

    out = ListenerAbx().normalize(registry)

    (j,) = out.judgments  # the bad clip records nothing
    assert (j.a, j.b, j.c) == ("song:x", "song:b", "song:a")
    assert {"rater_id": "rater2", "split": "fit", "strength": 1, "strength_source": "rater"}.items() <= j.meta.items()
    assert j.meta["listen_ms"] == 40000
    assert {(m.track_id, m.clip_start) for m in out.members} == {("song:x", None), ("song:a", None), ("song:b", None)}
    assert {t.audio_source for t in out.tracks} == {"deezer"}


def test_listener_abx_refuses_two_windows_for_one_track(registry):
    from stemsim.datasets.listener_abx import ListenerAbx

    write_listener_abx(registry, [rating("s1-01"), rating("s1-02", window=[10, 20])])
    with pytest.raises(ValueError, match="one window per track"):
        ListenerAbx().normalize(registry)


def test_dim_sim_majority_vote_names_and_clean_flag(registry):
    from stemsim.datasets.dim_sim import CLEAN_CSV, MSD_TRACKS, RAW_CSV, DimSim

    raw = registry.raw_dir("dim_sim")
    (raw / "dim-sim").mkdir()
    header = (
        "triplet_id,anchor_id,anchor_start_seconds,anchor_start_samples,song1_id,song1_start_seconds,"
        "song1_start_samples,song2_id,song2_start_seconds,song2_start_samples,sampling_rate,"
        "clip_lengths_seconds,clip_lengths_samples,song1_vote,song2_vote\n"
    )
    (raw / RAW_CSV).write_text(
        header
        + "0001,TRA,1.0,0,TRB,2.0,0,TRC,3.0,0,22050,3,66150,2,9\n"  # song2 wins 9 of 11
        + "0002,TRA,1.0,0,TRB,2.0,0,TRC,3.0,0,22050,3,66150,4,4\n"  # tie - skipped
        + "0003,TRA,1.0,0,TRB,2.0,0,TRX,3.0,0,22050,3,66150,5,1\n",  # TRX unknown to MSD - skipped
        encoding="utf-8",
    )
    (raw / CLEAN_CSV).write_text(header + "0001,TRA,1.0,0,TRB,2.0,0,TRC,3.0,0,22050,3,66150,0,1\n", encoding="utf-8")
    (raw / MSD_TRACKS).write_text(
        "TRA<SEP>S1<SEP>Artist A<SEP>Anchor\nTRB<SEP>S2<SEP>Artist B<SEP>One\nTRC<SEP>S3<SEP>Artist C<SEP>Two\n",
        encoding="utf-8",
    )

    out = DimSim().normalize(registry)

    [j] = out.judgments
    assert (j.a, j.b, j.c) == (
        song_track_id("Anchor", "Artist A"),
        song_track_id("Two", "Artist C"),
        song_track_id("One", "Artist B"),
    )
    assert j.value == pytest.approx(9 / 11) and j.weight == 11 and j.meta["clean"]
    assert {t.audio_source for t in out.tracks} == {"deezer"}


def test_matcha_one_triplet_per_decided_attribute(registry):
    from stemsim.datasets.matcha import CASES, DECISIONS, Matcha

    raw = registry.raw_dir("matcha")
    roles = ["ref", "sampleA", "sampleB"]
    cols = [f"{r}_{c}" for r in roles for c in ("id", "title", "link", "timestamp")]
    (raw / CASES).write_text(
        "case_idraw_,case_origin,section,is_control," + ",".join(cols) + "\n"
        "c_001,human,HP,0,r1,Ref,http://y/1,00:11-00:19,a1,A,http://y/2,01:00-01:08,b1,B,http://y/3,bad\n",
        encoding="utf-8",
    )
    attrs = ["melody", "harmony", "rhythm", "voice", "timbre"]
    (raw / DECISIONS).write_text(
        "case_id," + ",".join(f"{a}_decision,{a}_agreement,{a}_n" for a in attrs) + "\n"
        "c_001,B,83.3,6,A,50.0,6,Neither,50.0,6,B,100.0,6,A,75.0,4\n",
        encoding="utf-8",
    )
    out = Matcha().normalize(registry)

    got = {(j.facet, j.b, j.c) for j in out.judgments}
    assert got == {
        ("melody", "matcha:b1", "matcha:a1"),
        ("harmony", "matcha:a1", "matcha:b1"),
        ("voice", "matcha:b1", "matcha:a1"),
        ("timbre", "matcha:a1", "matcha:b1"),
    }  # rhythm was "Neither" - no ordering
    windows = {m.track_id: (m.clip_start, m.clip_end) for m in out.members}
    assert windows["matcha:r1"] == (11.0, 19.0) and windows["matcha:b1"] == (None, None)
