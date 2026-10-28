---
name: ticket
description: Start or finish work on a Jira ticket in the SSM project (Overtone). "start SSM-N" researches the ticket, makes the branch and proposes a plan; "finish" runs checks and reviews and opens the PR. Use when the user names an SSM ticket or asks to open a PR.
---

# Ticket workflow (SSM project)

Jira site: nicolascunderlik.atlassian.net, project `SSM`. The owner steers: propose, explain, wait (CLAUDE.md, "Working agreement"). Each step that changes something outside the working tree needs the owner's go-ahead unless they already gave it.

## Start: `start SSM-N`
1. Read the ticket (`getJiraIssue`, view "evidence"). Note its acceptance criteria. If it has none, propose some.
2. Research read-only. Search the code yourself first; spawn an `Explore` agent only if the change spans many unfamiliar modules.
3. Propose a plan: the files to change, the approach, any trade-offs, and anything on the CLAUDE.md "ask first" list. **Stop and wait for approval.**
4. Once approved: branch from an up-to-date `main` as `ssm-N/<short-slug>`, and move the ticket to In Progress.

## Finish: `finish`
1. `python scripts/run_checks.py`, then `.venv\Scripts\pyright src`. Fix only failures this branch introduced; report the rest.
2. Clean up, then review, in this order, each on demand (they cost tokens). Commit the branch's work first, so the cleanup agents' edits show up as their own diff.
   - The `test-reviewer` agent restructures and prunes the diff's tests, and reports missing tests and bugs.
   - The `docs-sanitizer` agent rewrites the diff's comments and docstrings, including any test-reviewer wrote.
   The owner reviews these edits (`git diff`) before they are committed.
   - `/code-review` at the level the owner prefers (default: low) for correctness.
   - The `conventions-reviewer` agent for project rules.
   Show the findings to the owner; fix only the ones they accept.
3. Commit messages: `ssm-N: <imperative summary>`. No AI attribution trailers or footers (owner's rule).
4. Open the PR against `main` with `gh pr create`. Title: same as the commit summary. Body: `## Summary` (why, plus the result or numbers if it's research), `## Changes` (bulleted), `## Testing` (what was run), and a link to the ticket. No AI footer.
5. Comment the PR link on the ticket. Leave the status for the owner to change: they move it to Done after merging.

CI (`lint`, `unit-tests`) must pass before GitHub allows the merge. CodeRabbit also comments on the PR, so point out its findings rather than repeating them.
