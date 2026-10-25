from types import SimpleNamespace

import pytest

pytest.importorskip("mlflow")  # CI's app venv doesn't have it

from stemsim.experiments import report  # noqa: E402

DOC = """Experiment: does it work?

    python -m something

Setup
- step one
"""


def fake_run(dirty="False", end_time=1_700_000_090_000):
    info = SimpleNamespace(
        run_name="v9", run_id="abc123", experiment_id="1", start_time=1_700_000_000_000, end_time=end_time
    )
    data = SimpleNamespace(
        tags={"git_commit": "2aad339", "git_dirty": dirty},
        params={"l2": "1.0", "data.dim_sim.judgments": "3781", "data.dim_sim.embedded": "8279"},
    )
    return SimpleNamespace(info=info, data=data)


def render(**overrides):
    kwargs = {
        "results": "| a | b |",
        "success": "beats the mix",
        "command": "python -m x",
        "experiment": "stem-weights",
        "artifacts": ["results.csv"],
    }
    return report.render(fake_run(), DOC, **{**kwargs, **overrides})


def test_render_fills_every_section_and_marks_missing_ones():
    text = render()
    assert "$" not in text  # no placeholder left unfilled
    assert text.startswith("# stem-weights: v9")
    assert "| Status | draft |" in text
    assert "## Question\n\nDoes it work?\n" in text
    assert "## Approach\n\n    python -m something" in text  # rest of the docstring, indent kept (code block)
    assert "- step one" in text
    assert "| dim_sim | 3781 | 8279 |" in text  # data snapshot, not mixed into params
    assert "| l2 | 1.0 |" in text and "| data.dim_sim" not in text
    assert "## Findings\n\n" + report.NOT_WRITTEN in text
    assert "took" not in text  # no runtime given: no misleading duration
    assert "(took 1 min 30 s)" in render(runtime_s=90.2)
    assert "- `report.md`" in text and "- `results.csv`" in text


def test_dirty_runs_are_flagged():
    text = report.render(fake_run(dirty="True"), DOC, results="", success="", command="x", experiment="e", artifacts=[])
    assert "uncommitted changes" in text


def test_edit_replaces_only_the_named_sections_and_status():
    text = render()
    edited = report.edit(text, {"Findings": "Stems help.\n", "Caveats": "Small n."}, status="reviewed")
    assert "## Findings\n\nStems help.\n\n## Caveats\n\nSmall n.\n\n## Decisions" in edited
    assert "| Status | reviewed |" in edited
    assert edited.replace("Stems help.", report.NOT_WRITTEN, 1).count("## ") == text.count("## ")
    # editing the last section keeps the file well-formed
    assert report.edit(edited, {"Artifacts": "- `x`"}).endswith("## Artifacts\n\n- `x`\n")


def test_edit_rejects_unknown_sections():
    with pytest.raises(KeyError):
        report.edit(render(), {"Nonsense": "x"})
