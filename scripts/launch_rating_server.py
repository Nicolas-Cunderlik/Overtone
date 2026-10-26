"""Starts the ABX rating app (research/stem_similarity/rate_server.py) and opens it.

    python scripts/launch_rating_server.py [--no-browser] [rate_server options]

Serves the newest session in research/stem_similarity/data/sessions unless
--session NAME_OR_PATH is given (see rate_server.py for --rater, --ratings).
Ratings are appended to research/stem_similarity/ratings.jsonl after every
answer; feed them to the pipeline with `python -m stemsim ingest listener_abx`.
Ctrl+C stops the server.
"""

import argparse
import threading
import webbrowser

from _common import RESEARCH, RESEARCH_VENV, run, venv_python

URL = "http://localhost:8765"  # rate_server.PORT


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-browser", action="store_true")
    args, server_args = parser.parse_known_args()
    if not args.no_browser:
        threading.Timer(3.0, webbrowser.open, [URL]).start()
    print(f"Rating app on {URL} (Ctrl+C to stop)")
    return run([venv_python(RESEARCH_VENV), "rate_server.py", *server_args], cwd=RESEARCH)


if __name__ == "__main__":
    raise SystemExit(main())
