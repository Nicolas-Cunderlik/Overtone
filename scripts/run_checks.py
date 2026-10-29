"""Runs locally what CI runs on every PR and push to main: pre-commit
(ruff + hygiene hooks) over all files, pyright over the app and core
packages, then the unit tests.

    python scripts/run_checks.py
"""

from _common import REPO, run, venv_python


def main():
    python = venv_python()
    steps = [
        ("pre-commit", [python, "-m", "pre_commit", "run", "--all-files"]),
        # --pythonpath: resolve imports against the repo venv, whatever python is on PATH
        ("pyright", [python, "-m", "pyright", "--pythonpath", python]),
        ("tests", [python, "-m", "pytest", "-q", "tests/unit"]),
    ]
    failed = []
    for name, command in steps:
        print(f"\n=== {name}")
        if run(command, cwd=REPO):
            failed.append(name)
    print("\n" + ("All checks passed" if not failed else f"FAILED: {', '.join(failed)}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
