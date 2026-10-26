"""stemsim.rating.committee on small synthetic pools - no registry, no embeddings on disk."""

import numpy as np
import pytest
from stemsim.rating import committee

CHANNELS = 5  # mix, drums, bass, vocals, other
D = 4


def unit(*vectors):
    v = np.array(vectors, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def song(mix, stems):
    """One song's (channels, d) embedding: a mix direction and one direction shared by all stems."""
    return unit(mix, stems, stems, stems, stems)


@pytest.fixture
def pool():
    # x; a matches x's mix but not its stems; b matches x's stems but not its mix; c matches both; e neither
    emb = np.stack(
        [
            song([1, 0, 0, 0], [0, 0, 1, 0]),  # x
            song([1, 0, 0, 0], [0, 0, 0, 1]),  # a
            song([0, 1, 0, 0], [0, 0, 1, 0]),  # b
            song([1, 0, 0, 0], [0, 0, 1, 0]),  # c
            song([0, 1, 0, 0], [0, 0, 0, 1]),  # e
        ]
    )
    return committee.Pool(
        ids=["x", "a", "b", "c", "e"],
        emb=emb.transpose(1, 0, 2),  # channel-major
        audible=np.ones((5, CHANNELS), bool),
        artists=np.array(["p", "q", "r", "s", "t"]),
    )


def make_committee(posterior_weights):
    """Posterior members as given, plus mix-only and stems-only stand-ins for the other families."""
    w = np.array(posterior_weights, float)
    mix, stems = np.eye(CHANNELS)[0], np.r_[0, np.full(CHANNELS - 1, 0.25)]
    return committee.Committee(
        names=[f"p{i}" for i in range(len(w))] + ["d1", "d2", "mix", "equal"],
        families=["posterior"] * len(w) + ["datasets"] * 2 + ["baselines"] * 2,
        weights=np.vstack([w, stems, stems, mix, mix]),
        fitted=stems,
        stats=(np.zeros(CHANNELS), np.ones(CHANNELS)),
    )


def test_zsim_gates_silent_stems(pool):
    pool.audible[1, 1] = False  # a has no drums
    pool.audible[1:3, 2] = False  # neither a nor b has bass; x does
    pool.audible[0, 3] = pool.audible[1, 3] = False  # neither x nor a has vocals
    z = pool.zsim(0, (np.zeros(CHANNELS), np.ones(CHANNELS)))
    assert z.shape == (5, CHANNELS)  # x against every row
    assert z[1, 1] == -1.0  # one song has drums, the other doesn't
    assert z[1, 2] == -1.0
    assert z[1, 3] == 0.0  # silent in both: no evidence
    assert z[1, 0] == pytest.approx(1.0)  # same mix direction


def test_a_unanimous_committee_has_no_disagreement(pool):
    comm = make_committee([[1, 0, 0, 0, 0]])
    share, families = comm.split(comm.scores(pool, 0), [3], [4])  # c matches x, e doesn't
    assert share[0] == 1.0
    assert committee.disagreement(share)[0] == 0.0
    assert np.isnan(families["app"][0])  # no app vectors: the app abstains


def test_mix_vs_stems_splits_the_committee(pool):
    # posterior half mix-led, half stem-led; datasets stem-led; baselines mix-led -> an even split
    comm = make_committee([[1, 0, 0, 0, 0], [0, 1, 1, 1, 1]])
    share, families = comm.split(comm.scores(pool, 0), [1], [2])
    assert (families["posterior"][0], families["datasets"][0], families["baselines"][0]) == (0.5, 0.0, 1.0)
    assert share[0] == pytest.approx(0.5)
    assert committee.disagreement(share)[0] == pytest.approx(1.0)


def test_app_votes_when_it_knows_all_three(pool):
    pool.app = unit([1, 0], [1, 0], [0, 1], [1, 0], [0, 1])
    comm = make_committee([[1, 0, 0, 0, 0]])
    share, families = comm.split(comm.scores(pool, 0), [1], [2])
    assert families["app"][0] == 1.0  # a is identical to x in app space
    pool.app[2] = np.nan
    _, families = comm.split(comm.scores(pool, 0), [1], [2])
    assert np.isnan(families["app"][0])


def test_top_k_per_model(pool):
    comm = make_committee([[1, 0, 0, 0, 0]])
    top = comm.top_k(comm.scores(pool, 0), [1, 2, 3], k=2)
    assert set(top["mix"]) == {1, 3}  # a and c share x's mix
    assert set(top["fitted"]) == {2, 3}  # b and c share x's stems
    assert "app" not in top
