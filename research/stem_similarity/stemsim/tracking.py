"""Experiment tracking with a local MLflow store (inside the gitignored data dir).

Every run records, besides its own params/metrics/artifacts:
  git_commit / git_dirty  the exact code (dirty = uncommitted changes existed)
  data.*                  how many judgments and embedded tracks each dataset
                          had at the time - results change as data lands, and
                          this is what makes two runs comparable
  report.md               purpose, approach, results and findings in prose
                          (experiments/report.py; findings are added later)
Browse runs:  python scripts/launch_mlflow.py (from the repo root), which opens
http://127.0.0.1:5000
"""

import subprocess
from contextlib import contextmanager

import mlflow

from . import config


def tracking_uri():
    return f"sqlite:///{(config.DATA_DIR / 'mlflow.db').as_posix()}"


def _git(*args):
    try:
        command = ["git", *args]  # noqa: S607 - git from PATH, fixed arguments
        return subprocess.run(command, capture_output=True, text=True, cwd=config.ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_snapshot(registry, embed_stage):
    """{dataset: (judgments, tracks with embeddings)} at run time."""
    rows = registry.conn.execute(
        """SELECT d.name, d.n_judgments,
                  (SELECT COUNT(*) FROM memberships m JOIN stage_status s ON s.track_id = m.track_id
                   WHERE m.dataset = d.name AND s.stage = ? AND s.state = 'done')
           FROM datasets d ORDER BY d.name""",
        (embed_stage,),
    ).fetchall()
    return {name: (judgments, embedded) for name, judgments, embedded in rows}


@contextmanager
def run(experiment, run_name, registry, embed_stage, params):
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = (config.DATA_DIR / "mlartifacts" / experiment).as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifacts)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags(
            {"git_commit": _git("rev-parse", "--short", "HEAD"), "git_dirty": str(bool(_git("status", "--porcelain")))}
        )
        for dataset, (judgments, embedded) in data_snapshot(registry, embed_stage).items():
            mlflow.log_params({f"data.{dataset}.judgments": judgments, f"data.{dataset}.embedded": embedded})
        mlflow.log_params(params)
        yield active
