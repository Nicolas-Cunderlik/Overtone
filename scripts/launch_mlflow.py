"""Opens the MLflow UI over the research experiment runs.

    python scripts/launch_mlflow.py [--port 5000] [--no-browser]

Runs `mlflow ui` from the repo venv against research/stem_similarity/data/mlflow.db
(the store stemsim.tracking writes to). Ctrl+C stops it.
"""

import argparse
import os
import threading
import webbrowser

from _common import RESEARCH, RESEARCH_DATA, run, venv_python


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    store = RESEARCH_DATA / "mlflow.db"
    if not store.exists():
        print(
            f"No runs yet ({store} doesn't exist) - run an experiment first, e.g.\n"
            "  python -m stemsim.experiments.stem_weights   (from research/stem_similarity)"
        )
        return 1
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(3.0, webbrowser.open, [url]).start()
    print(f"MLflow UI on {url} (Ctrl+C to stop)")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    command = [
        venv_python(),
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        f"sqlite:///{store.as_posix()}",
        "--port",
        str(args.port),
    ]
    return run(command, cwd=RESEARCH, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
