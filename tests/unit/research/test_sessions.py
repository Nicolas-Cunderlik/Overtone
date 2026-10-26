"""stemsim.sessions' selection rules on a synthetic pool - no registry, no embeddings on disk."""

import random

import numpy as np
import pytest
from stemsim import committee, ratings, sessions

CHANNELS, D, N = 5, 8, 60


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(N, CHANNELS, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    artists = np.array([f"artist {i // 2}" for i in range(N)])  # two songs per artist
    return committee.Pool([f"song:{i}" for i in range(N)], emb, np.ones((N, CHANNELS), bool), artists)


@pytest.fixture
def comm():
    eye = np.eye(CHANNELS)
    return committee.Committee(
        names=["p0", "p1", "d0", "d1", "mix", "equal"],
        families=["posterior"] * 2 + ["datasets"] * 2 + ["baselines"] * 2,
        weights=np.vstack([eye[0], eye[1], eye[2], eye[3], eye[0], np.full(CHANNELS, 0.2)]),
        fitted=np.full(CHANNELS, 0.2),
        stats=(np.zeros(CHANNELS), np.ones(CHANNELS)),
    )


def split_of(track_id):  # the first 40 songs are fit, the rest validation
    return ratings.FIT if int(track_id.split(":")[1]) < 40 else ratings.VALIDATION


def check_rules(picker, triplets, split):
    songs = [t for trip in triplets for t in trip[:3]]
    assert len(songs) == len(set(songs))  # no song twice in a session
    artist = dict(zip(picker.pool.ids, picker.pool.artists, strict=True))
    for x, a, b, _ in triplets:
        assert {split_of(t) for t in (x, a, b)} == {split}
        assert artist[x] not in (artist[a], artist[b]) and artist[a] != artist[b]


def test_neighbourhood_triplets_follow_the_rules(pool, comm):
    picker = sessions.Picker(pool, comm, split_of, np.random.default_rng(1))
    fit = picker.neighbourhood_triplets(ratings.FIT, 3)
    val = picker.neighbourhood_triplets(ratings.VALIDATION, 3)
    assert len(fit) == len(val) == 3
    check_rules(picker, fit, ratings.FIT)
    check_rules(picker, val, ratings.VALIDATION)
    assert fit[0][3]["rule"] == "neighbourhood" and set(fit[0][3]["in_top"]) == {"a", "b"}


def test_contested_triplets_are_the_most_even_splits(pool, comm):
    picker = sessions.Picker(pool, comm, split_of, np.random.default_rng(1))
    picked = picker.contested_triplets(ratings.FIT, 3)
    check_rules(picker, picked, ratings.FIT)
    scores = [p[3]["disagreement"] for p in picked]
    assert scores == sorted(scores, reverse=True)
    for x, a, b, sel in picked:  # the recorded split is the committee's
        row = pool.row
        share, _ = comm.split(pool, row[x], [row[a]], [row[b]])
        assert sel["share_a"] == pytest.approx(share[0], abs=1e-3)


def test_seen_triplets_and_flagged_songs_are_never_picked(pool, comm):
    flagged = {f"song:{i}" for i in range(0, 40, 3)}
    first = sessions.Picker(pool, comm, split_of, np.random.default_rng(1)).neighbourhood_triplets(ratings.FIT, 3)
    seen = {sessions.seen_key(*t[:3]) for t in first}
    again = sessions.Picker(pool, comm, split_of, np.random.default_rng(1), seen=seen, flagged=flagged)
    picked = again.neighbourhood_triplets(ratings.FIT, 3)
    assert not {sessions.seen_key(*t[:3]) for t in picked} & seen
    assert not {s for t in picked for s in t[:3]} & flagged


def test_pick_shared_keeps_fit_songs_and_hides_the_crowd_answer_in_selection():
    candidates = [
        ("dim_sim", 1, "song:1", "song:2", "song:3", 0.9, 10),
        ("dim_sim", 2, "song:1", "song:2", "song:50", 0.9, 10),  # a validation song
    ]
    for seed in range(10):
        x, a, b, sel = sessions.pick_shared(candidates, split_of, set(), set(), random.Random(seed))
        assert sel["judgment_id"] == 1
        assert (a, b)[sel["crowd_closer"] == "b"] == "song:2"  # the crowd's pick, wherever it landed
    assert sessions.pick_shared(candidates, split_of, {"song:3"}, set(), random.Random(0)) is None


def rated(rating_id, **changes):
    row = {
        "id": rating_id, "session": "nicolas-001", "design": "neighbourhood", "split": "fit", "choice": "a",
        "x": "song:x", "a": "song:a", "b": "song:b", "play": "mix", "window": None, "stems": None,
        "audio": {k: {"source": "deezer", "ref": f"audio/{k}.mp3"} for k in "xab"},
    }  # fmt: skip
    return row | changes


def test_repeats_swap_a_and_b_and_skip_what_cant_be_repeated():
    rows = [
        rated("pilot-01", session="pilot"),
        rated("s-01", choice="bad_clip"),
        rated("s-02"),
        rated("s-03", design="repeat", repeat_of="s-02"),
    ]
    assert sessions.pick_repeat(rows, random.Random(0)) is None  # s-02 is already repeated
    original = rated("s-04")
    row = sessions.repeat_row(original)
    assert (row["a"], row["b"], row["repeat_of"]) == ("song:b", "song:a", "s-04")
    assert row["audio"]["a"] == original["audio"]["b"]
    assert sessions.pick_repeat([*rows, original], random.Random(0))["id"] == "s-04"


def test_finished_rows_are_valid_once_answered(pool, comm):
    picker = sessions.Picker(pool, comm, split_of, np.random.default_rng(1))
    audio = {t: {"source": "deezer", "ref": f"audio/{t}.mp3"} for t in pool.ids}
    rows = [
        sessions.triplet_row(ratings.NEIGHBOURHOOD, ratings.FIT, x, a, b, sel, audio)
        for x, a, b, sel in picker.neighbourhood_triplets(ratings.FIT, 2)
    ]
    rows = sessions.finish(rows, "nicolas-001", {}, random.Random(0))
    assert [r["id"] for r in rows] == ["nicolas-001-01", "nicolas-001-02"]
    answer = {"choice": "b", "strength": 1, "tags": [], "note": ""}
    assert all(ratings.check(ratings.from_session(r, answer, "nicolas", "now")) == [] for r in rows)


def test_session_names_count_up_per_rater(tmp_path):
    assert sessions.session_name("nicolas", tmp_path) == "nicolas-001"
    (tmp_path / "nicolas-001.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "rater2-001.jsonl").write_text("", encoding="utf-8")
    assert sessions.session_name("nicolas", tmp_path) == "nicolas-002"
