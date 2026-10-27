"""PreToolUse hook: stops Claude Code from reading or editing .env files.

The deny rules in .claude/settings.json already cover the Read/Edit/Write tools.
This hook adds a check on shell commands and searches, so `cat .env` or a Grep
aimed at .env is blocked too. `.env.example` is allowed: it's the template.

It errs towards blocking: a command that only mentions `.env` (e.g. grepping
.gitignore for it) is also stopped. Claude Code passes the tool call as JSON on
stdin; exit code 2 blocks the call and shows stderr to Claude.
"""

import json
import re
import sys

# `.env` or `.env.<suffix>` as a whole path component, not `.envrc` or `prod.env`
ENV_FILE = re.compile(r"(?<![\w.-])\.env(?:\.[\w-]+)*(?![\w-])")
ALLOWED = {".env.example"}


def mentions_env_file(text):
    return any(match.group(0) not in ALLOWED for match in ENV_FILE.finditer(text))


def main():
    call = json.load(sys.stdin)
    tool_input = call.get("tool_input") or {}
    # file_path: Read/Edit/Write; notebook_path: NotebookEdit; path/glob/pattern: Grep/Glob;
    # command: Bash/PowerShell
    fields = ("file_path", "notebook_path", "path", "glob", "pattern", "command")
    text = " ".join(str(tool_input.get(field, "")) for field in fields)
    if mentions_env_file(text):
        print(
            "Blocked: .env holds secrets and is off-limits to Claude (see CLAUDE.md, 'Secrets and data'). "
            "Use .env.example for the variable names.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
