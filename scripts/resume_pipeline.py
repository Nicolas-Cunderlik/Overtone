"""Restarts the long-running pipeline stages in the background.

    python scripts/resume_pipeline.py            downloads + GPU stages
    python scripts/resume_pipeline.py --gpu      only separate -> embed
    python scripts/resume_pipeline.py --audio    only ISRC lookups -> audio downloads
    python scripts/resume_pipeline.py --dry-run  show what would start

Every stage resumes where it stopped and holds a lock, so this is safe to run
at any time: stages already running are skipped, never duplicated. Output
goes to research/stem_similarity/data/logs/<job>.log. Jobs keep running after
this script exits (they pause while the PC sleeps).
"""

import argparse
import subprocess
import sys

from _common import LOG_DIR, RESEARCH, running_stages, venv_python

# job name -> (lock names it would hold, stemsim commands run one after another)
JOBS = {
    # Deezer audio waits for each track's ISRC lookup, so the two run in order
    "audio": (["isrc", "audio-deezer"], [["isrc"], ["audio", "--source", "deezer"]]),
    "gpu": (["separate", "embed-muq-mulan-large"], [["separate"], ["embed"]]),
}


def run_chain(commands):
    """Foreground: runs stemsim commands in order (used by the detached job)."""
    python = venv_python()
    for command in commands:
        code = subprocess.call([python, "-u", "-m", "stemsim", *command], cwd=RESEARCH)  # noqa: S603
        if code:
            return code
    return 0


def start_detached(name, commands):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{name}.log", "a", encoding="utf-8")  # noqa: SIM115 - handed to the child process
    chain = [sys.executable, __file__, "--chain", *["+".join(c) for c in commands]]
    options = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": RESEARCH}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    subprocess.Popen(chain, **options)  # noqa: S603 - our own script, fixed arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chain", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.chain:
        return run_chain([step.split("+") for step in args.chain])

    wanted = [name for name, flag in (("audio", args.audio), ("gpu", args.gpu)) if flag] or list(JOBS)
    held = set(running_stages())
    for name in wanted:
        locks, commands = JOBS[name]
        if held & set(locks):
            print(f"{name:6s} already running ({', '.join(sorted(held & set(locks)))}) - skipped")
            continue
        plan = " -> ".join("stemsim " + " ".join(c) for c in commands)
        if args.dry_run:
            print(f"{name:6s} would start: {plan}")
            continue
        start_detached(name, commands)
        print(f"{name:6s} started: {plan}  (log: data/logs/{name}.log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
