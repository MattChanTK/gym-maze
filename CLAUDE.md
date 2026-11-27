# gym-maze

Gymnasium 2D maze environment (Python 3.8+, gymnasium>=1.0, pygame, numpy). Main branch: `master`.
Shared tooling: `kit/` = dev-kit at the tag in `kit/VERSION` (see Kit below). Settings the tools read: `kit.env`. Secrets: git-ignored `.env.local` and `CLAUDE.local.md`, never committed.
File map: docs/CODE_OVERVIEW.md and docs/code/ (keep updated when files change).
Words with a set meaning: docs/GLOSSARY.md. Read before naming things or writing copy; add a row when you coin a term.

## Talk short (caveman mode)
Applies to every coding agent and subagent (Claude, cloud agents, bots).
- Agent to agent (subagents, handoffs, briefs, reports, PR notes for bots): caveman mode. As short as possible. Facts, SHAs, file:line, asks. No filler.
- To Matthew: almost caveman, a bit more readable. Short plain sentences, key facts and asks only. No recaps.

## Keep sessions short (tokens)
Every call re-reads the whole conversation, so long sessions cost the most. Auto-compact at 200k (`.claude/settings.json`).
- One ticket or feature per session. New unrelated request: offer a fresh session.
- At a natural stop (commit, merge, finished ticket), check context; past ~150k, suggest a fresh session with a short brief.
- Code research: start with docs/CODE_OVERVIEW.md and docs/code/, then open only the lines you need. Doc disagrees with code: trust code, fix the doc.
- Read only the part of a file you need.

## One worktree per session
Several sessions run at once; each works apart.
- **Start:** before changing files, make a worktree on a new branch from main: `kit/tools/worktree.sh create <name>`. Read-only work needs none.
- **Setup:** `kit/tools/lib.sh` copies `CLAUDE.local.md` and `.env.local` from the main checkout into a worktree. `kit/tools/scan_secrets.py` fails if `CLAUDE.local.md` is missing.
- **Finish:** `kit/tools/check.sh`, commit on the branch (`kit/tools/commit.sh`), push (ask first). Do not merge.
- **Clean up:** `kit/tools/worktree.sh prune` removes merged, clean worktrees. `remove <name>` refuses a dirty tree without `--force`.
- **Conflicts:** in CHANGELOG and other newest-first docs, keep both entries. Unclear code conflict or someone else's uncommitted work: stop and ask; never discard it.
- Tests, QA and uploads refuse to run in the protected checkout (`KIT_PROTECTED_CHECKOUT`).

## Merge gate
No merge to main, no deploy, no release upload, without Matthew's yes. Docs-only changes may go straight to main.
Never merge with required checks red or missing. Required check: `tests`.
Ask first: push, deploy, anything that calls a paid API, anything that changes live data.

## Docs (keep them in this shape)
Log notable decisions as a short dated entry in `docs/company/MINUTES.md` (newest first).
- `README.md`: what it does, install, setup. Update when a feature changes.
- `docs/CHANGELOG.md`: newest first; date, what was built, decisions, lessons, commit hashes. Add an entry with each change. Over 80 KB: `python3 kit/tools/roll_changelog.py` (check.sh fails until you do). Read only the top or grep.
- `docs/BACKLOG.md`: unbuilt work in priority order: a table (code, item, owner, impact, effort), each row linking `docs/backlog/<KIND>/<CODE>.md`. Code = kind + next number, never reused (e.g. APP-007). Done or dropped: delete file and row, add a row to ARCHIVE, name the code in CHANGELOG. No changelog entry for just filing a ticket.
- `docs/ARCHIVE.md`: finished and dropped items, newest first.
- `docs/FEEDBACK.md`: reported problems, newest first, FB code, resolving ticket, state. Triage first; suggest, don't fix unasked.
- `docs/GLOSSARY.md`, `docs/CODE_OVERVIEW.md`, `docs/code/`, `docs/TESTING.md`.

## Shortcuts
- `kit/tools/check.sh`: every fast test (kit tools + `KIT_CHECKS`). Runs freely.
- `kit/tools/commit.sh "message" file…`: stages only the named files, secret-scans, commits with the attribution line.
- `kit/tools/push.sh`: lists outgoing commits, scans, pushes. Ask first.
- `kit/tools/worktree.sh create|list|remove|prune`.
- `kit/tools/lane.sh heavy|light -- cmd`: run-queue slot so parallel sessions don't overload the machine (heavy 1, light 3).
- `python3 kit/tools/scrub_report.py`: scrub logs before sharing. Never print keys, tokens, device IDs, team IDs, or emails.

## Secrets
Never print or commit keys. Never copy `.env.local`, `CLAUDE.local.md`, or key files (*.p8, *.pem) anywhere tracked.

## Kit (shared dev-kit, versioned)
`kit/` is a copy of github.com/MattChanTK/dev-kit at the tag in `kit/VERSION`. Other projects use it too.
- Don't change kit/ for this repo only. Repo-specific tools go in `tools/`.
- A kit fix that helps every project: edit kit/, test, then `kit/tools/kit.sh contribute <branch>` (pushes a dev-kit branch; ask first) and open a PR on dev-kit. Keep the edit here until a new tag ships it.
- Upgrade on purpose: `kit/tools/kit.sh status`, then `kit/tools/kit.sh update vX.Y.Z`, run `kit/tools/check.sh`, commit kit/ alone.
