import numpy as np
import pytest

pytest.importorskip("mlflow")  # the experiment logs to MLflow; CI's app venv doesn't have it

from stemsim.experiments import stem_weights as sw  # noqa: E402


def test_correctness_scores_ties_half_and_keeps_missing():
    out = sw.correctness([1.0, -2.0, 0.0, np.nan])
    assert out[:3].tolist() == [1.0, 0.0, 0.5]
    assert np.isnan(out[3])


def test_triple_groups_ignore_song_order():
    rows = [{"a": "x", "b": "y", "c": "z"}, {"a": "x", "b": "z", "c": "y"}, {"a": "x", "b": "y", "c": "w"}]
    groups = sw.triple_groups(rows)
    assert groups[0] == groups[1] != groups[2]


def test_group_resamples_draw_whole_groups():
    ids, counts = sw.group_resamples(np.array([5, 5, 7, 9]), np.random.default_rng(0), n=100)
    assert ids.tolist() == [0, 0, 1, 2]
    assert counts.shape == (100, 3)
    assert (counts.sum(1) == 3).all()  # G groups drawn per resample, not one per row


def test_bootstrap_agreement_point_estimates_and_paired_differences():
    rng = np.random.default_rng(0)
    n = 400
    mix = rng.normal(0.3, 1.0, n)
    better = mix + 2.0  # agrees wherever mix does, and more
    scores = {sw.BASELINE: mix, "same": mix.copy(), "better": better}
    metrics, diffs = sw.bootstrap_agreement(scores, np.arange(n), rng)

    count, point, lo, hi = metrics[sw.BASELINE]
    assert count == n
    assert point == pytest.approx(np.mean(mix > 0))
    assert lo < point < hi
    assert diffs[("same", sw.BASELINE)][1:] == (0.0, 0.0, 0.0)  # paired: identical models never differ
    _, d, d_lo, _ = diffs[("better", sw.BASELINE)]
    assert d > 0 and d_lo > 0


def test_bootstrap_agreement_skips_rows_a_model_cannot_score():
    scores = {sw.BASELINE: np.array([1.0, -1.0, 1.0, 1.0]), "partial": np.array([1.0, np.nan, np.nan, -1.0])}
    metrics, diffs = sw.bootstrap_agreement(scores, np.arange(4), np.random.default_rng(0))
    assert metrics["partial"][:2] == (2, 0.5)
    assert diffs[("partial", sw.BASELINE)][:2] == (2, -0.5)  # compared on the 2 shared rows only


def test_bootstrap_spearman_perfect_ranking():
    human = np.arange(10, dtype=float)
    scores = {sw.BASELINE: human * 2, "reversed": -human}
    metrics, diffs = sw.bootstrap_spearman(scores, human, np.random.default_rng(0))
    assert metrics[sw.BASELINE][1] == pytest.approx(1.0)
    assert diffs[("reversed", sw.BASELINE)][1] == pytest.approx(-2.0)
