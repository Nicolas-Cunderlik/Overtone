"""Starts MLflow's MCP server over the research runs, for Claude Code (.mcp.json).

    python scripts/launch_mlflow_mcp.py

Reads the same store stemsim.tracking writes to: <STEMSIM_DATA_DIR>/mlflow.db,
with stemsim's default of research/stem_similarity/data. A relative
STEMSIM_DATA_DIR resolves from research/stem_similarity, where stemsim runs.
Only the experiment and run tools are exposed, and .claude/settings.json denies
the ones that write. MCP talks over stdin/stdout, so this script never prints
to stdout.
"""

import os
import sys

from _common import RESEARCH, RESEARCH_VENV, run, venv_python


def main():
    store = RESEARCH / os.environ.get("STEMSIM_DATA_DIR", "data") / "mlflow.db"
    if not store.exists():
        # Without this check MLflow would quietly create an empty store and report no runs
        print(f"No MLflow store at {store} - run an experiment first", file=sys.stderr)
        return 1
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"sqlite:///{store.as_posix()}",
        "MLFLOW_MCP_TOOLS": "experiments,runs",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    return run([venv_python(RESEARCH_VENV), "-m", "mlflow", "mcp", "run"], cwd=RESEARCH, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
