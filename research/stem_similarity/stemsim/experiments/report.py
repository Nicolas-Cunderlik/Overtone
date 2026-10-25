"""Run reports: every MLflow run carries a report.md saying why it ran, how,
and what it found - readable months later without the code or the chat.

Template: report_template.md (stdlib string.Template, $name placeholders;
KAN-29 weighs moving templates to Jinja). Sections come from three places:
  the run         metadata, git commit, data snapshot (data.* params),
                  parameters, artifacts, reproduce command
  the experiment  question + approach (its module docstring, so the
                  explanation has one source of truth), success criteria,
                  standing caveats, results summary
  a person        findings, decisions / next steps, run-specific caveats,
                  change since the previous run - written after looking at
                  the results, added to the run later with this CLI

Unfilled sections say "Not written yet" instead of disappearing, so a report
shows what is still missing.

    python -m stemsim.experiments.report RUN [--findings FILE] [--caveats FILE]
        [--next-steps FILE] [--previous FILE] [--status reviewed]

RUN is a run name or id; each FILE is Markdown that replaces that section
(use ### or lower for headings inside it - ## starts the next section).
"""

import argparse
import re
import string
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from .. import tracking

TEMPLATE = Path(__file__).with_name("report_template.md")
ARTIFACT = "report.md"
NOT_WRITTEN = "_Not written yet._"
EDITABLE = {  # CLI option -> section heading
    "findings": "Findings",
    "caveats": "Caveats",
    "next_steps": "Decisions and next steps",
    "previous": "Change since previous run",
}


def render(
    run,
    docstring,
    *,
    results,
    success,
    command,
    caveats=None,
    related=(),
    findings=None,
    next_steps=None,
    previous=None,
    status="draft",
    experiment=None,
    artifacts=None,
    runtime_s=None,
):
    """The report for an MLflow Run. `docstring` is the experiment module's:
    its first paragraph is the question, the rest the approach. `runtime_s`
    is the experiment's own wall time - the MLflow run itself is usually only
    open while results are logged, so its duration would mislead."""
    info, data = run.info, run.data
    question, _, approach = docstring.strip("\n").partition("\n\n")
    question = " ".join(question.removeprefix("Experiment:").split())
    experiment = experiment or MlflowClient().get_experiment(info.experiment_id).name
    if artifacts is None:
        artifacts = [f.path for f in MlflowClient().list_artifacts(info.run_id)]
    fields = {
        "title": f"{experiment}: {info.run_name}",
        "experiment": experiment,
        "run_name": info.run_name,
        "run_id": info.run_id,
        "status": status,
        "date": _time(info.start_time) + (f" (took {_duration(runtime_s)})" if runtime_s else ""),
        "code": _code(data.tags),
        "related": ", ".join(related) or "none",
        "question": question[:1].upper() + question[1:],
        "success": success or NOT_WRITTEN,
        "approach": approach.strip("\n").rstrip() or NOT_WRITTEN,  # keep indents: code blocks
        "data": _data_table(data.params),
        "params": _params_table(data.params),
        "results": results or NOT_WRITTEN,
        "findings": findings or NOT_WRITTEN,
        "caveats": caveats or NOT_WRITTEN,
        "next_steps": next_steps or NOT_WRITTEN,
        "previous": previous or NOT_WRITTEN,
        "reproduce": _reproduce(data.tags, command),
        "artifacts": "\n".join(f"- `{name}`" for name in sorted({*artifacts, ARTIFACT})),
    }
    return string.Template(TEMPLATE.read_text(encoding="utf-8")).substitute(fields)


def edit(text, sections=None, status=None):
    """Replace whole sections ({heading: markdown}) and/or the status line."""
    for heading, body in (sections or {}).items():
        pattern = re.compile(rf"(^## {re.escape(heading)}\n\n)(.*?)(?=\n## |\Z)", re.S | re.M)
        if not pattern.search(text):
            raise KeyError(f"no section {heading!r} in the report")
        text = pattern.sub(lambda m, body=body: m.group(1) + body.strip() + "\n", text, count=1)
    if status:
        text = re.sub(r"^\| Status \| .* \|$", f"| Status | {status} |", text, count=1, flags=re.M)
    return text


def log(run_id, text):
    MlflowClient().log_text(run_id, text, ARTIFACT)


def update(run_id, sections=None, status=None):
    text = mlflow.artifacts.load_text(f"runs:/{run_id}/{ARTIFACT}")
    log(run_id, edit(text, sections, status))


def resolve_run(name_or_id):
    """A Run by id, or by run name (must be unique across experiments)."""
    client = MlflowClient()
    try:
        return client.get_run(name_or_id)
    except mlflow.exceptions.MlflowException:
        pass
    experiments = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiments, filter_string=f"tags.mlflow.runName = '{name_or_id}'")
    if len(runs) != 1:
        raise SystemExit(f"{len(runs)} runs named {name_or_id!r} - pass the run id instead")
    return runs[0]


def _time(ms):
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def _duration(seconds):
    seconds = round(seconds)
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"


def _code(tags):
    commit = tags.get("git_commit", "unknown")
    if tags.get("git_dirty") == "True":
        return f"`{commit}` + uncommitted changes (this exact code is not in git)"
    return f"`{commit}`"


def _data_table(params):
    datasets = sorted({key.split(".")[1] for key in params if key.startswith("data.")})
    if not datasets:
        return "_No data snapshot recorded._"
    rows = [
        f"| {d} | {params.get(f'data.{d}.judgments', '-')} | {params.get(f'data.{d}.embedded', '-')} |"
        for d in datasets
    ]
    return "\n".join(["| Dataset | Judgments | Tracks with embeddings |", "|---|---|---|", *rows])


def _params_table(params):
    rows = [f"| {key} | {value} |" for key, value in sorted(params.items()) if not key.startswith("data.")]
    return "\n".join(["| Parameter | Value |", "|---|---|", *rows]) if rows else "_None._"


def _reproduce(tags, command):
    lines = ["```", "cd research/stem_similarity", f"git checkout {tags.get('git_commit', '<commit>')}", command, "```"]
    if tags.get("git_dirty") == "True":
        lines.append("The run had uncommitted changes, so this checkout may not reproduce it exactly.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Add or replace sections of a run's report.md.")
    parser.add_argument("run", help="run name or id")
    for option in EDITABLE:
        parser.add_argument(f"--{option.replace('_', '-')}", type=Path, metavar="FILE")
    parser.add_argument("--status", help="e.g. reviewed")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(tracking.tracking_uri())
    run = resolve_run(args.run)
    sections = {
        heading: getattr(args, option).read_text(encoding="utf-8")
        for option, heading in EDITABLE.items()
        if getattr(args, option)
    }
    if not sections and not args.status:
        parser.error("nothing to change: pass a section file and/or --status")
    update(run.info.run_id, sections, args.status)
    print(f"updated {ARTIFACT} of {run.info.run_name}: {', '.join([*sections, *(['status'] * bool(args.status))])}")


if __name__ == "__main__":
    main()
