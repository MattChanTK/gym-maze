# Changelog

Newest first. Each entry: date, what was built, decisions, lessons, commit hashes. Older entries: `docs/changelog/`.

## 2026-11-28 CI on self-hosted Mac
- `tests.yml` from kit v0.4.0 template (Matthew asked): runs on self-hosted Mac (label `mac`), waits while it sleeps; PR label `ci-on-github` runs on ubuntu-latest. No `git config --global` (GIT_AUTHOR_*/GIT_COMMITTER_* env). Deps in a venv under RUNNER_TEMP so the Mac's Python stays clean. Mac uses its own python3; GitHub-hosted keeps setup-python 3.13 (3.14 has no pygame wheel).
- Needs a runner for this repo on the Mac: `kit/tools/mac_runner.sh register MattChanTK/gym-maze mac`.

## 2026-11-28 kit v0.4.2
- kit/ updated v0.3.2 -> v0.4.2 (daily bot): kit_env.py (Python tools load kit.env themselves), mac_runner.sh, lane owner + multi-repo QA dashboard. check.sh pass (13 tests).
- Not taken: v0.4.0 self-hosted Mac `tests.yml`. gym-maze keeps ubuntu-latest CI; `.github/` edits need Matthew.

## 2026-11-27 kit v0.3.2, no iOS add-on
- kit/ updated -> v0.3.2 (cf27276) and kit/ios removed: gym-maze is not an iOS app. From v0.3.2 kit.sh update keeps kit/ios out of projects without it. check.sh pass.

## 2026-11-27 kit v0.3.1
- kit/ updated v0.2.1 -> v0.3.1 (a004a2d): qa.sh, QA dashboard, layout_rules, iOS layout/release tools, shared lanes; template drops docs/company. check.sh pass (13 tests).

## 2026-11-27 Remove docs/company
- Deleted docs/company/ (MINUTES.md, STRUCTURE.md) and the MINUTES line in CLAUDE.md: not needed (Matthew). Decisions live in this changelog. Same change in dev-kit PR #4 (v0.3.1).

## 2026-11-27 Daily maintenance bot
- Added docs/AUTOMATION.md: runbook and guardrails for a daily scheduled Claude session (health check, kit updates, issue/PR triage, `needs-approval` label, pinned "Bot daily log" issue).

## 2026-11-27 v1.0: move to gymnasium, add LICENSE
- Old `gym` is unmaintained and `numpy<2` has no Python 3.13 wheels, so v0.5 pins could not install cleanly on current Python. Moved to `gymnasium>=1.0`, any numpy, Python 3.8+.
- API now gymnasium style: `reset(seed=, options=)` returns `(obs, info)`; `step` returns `(obs, reward, terminated, truncated, info)`; `render_mode` in the constructor, `render()` no args. `enable_render` kept as alias. Default is now headless (no window unless `render_mode` set).
- `reset(options={"start": (x, y)})` (#12). Obs dtype int64 to match observation_space. gymnasium `check_env` added to tests.
- Added MIT LICENSE (#17). setup.py version 1.0. CI on Python 3.13.
- Tested: gymnasium 1.0.0 and 1.4.0, numpy 2.5, Python 3.13, 12 tests pass with UserWarning as error.

## 2026-11-27 Open PR cleanup + dev-kit v0.2.0
- dev-kit v0.2.0 installed (kit/, CLAUDE.md, docs/, tests workflow).
- From PR #22 (lcdbezerra): README uses real env ids; numpy int actions accepted; obs returned as copies. Dropped its pickle maze loader and exact pins; pins now `gym<0.26`, `numpy<2`.
- From PR #15 (Kuzat): `maze-random-custom-v0` / `MazeEnvRandomCustom`, `zero_step_rewards`, `env.restore()`, int obs. Dropped its setup.py url/version change and README CRLF rewrite.
- Fixes #20 (replaces PR #21, JupiLogy): `enable_render=False` no longer crashes in `render()`; returns None.
- PR #18 closed (IDE files, unrelated scripts).
- Added tests/test_maze_env.py. setup.py version 0.5.
