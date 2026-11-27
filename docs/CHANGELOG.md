# Changelog

Newest first. Each entry: date, what was built, decisions, lessons, commit hashes. Older entries: `docs/changelog/`.

## 2026-11-27 Open PR cleanup + dev-kit v0.2.0
- dev-kit v0.2.0 installed (kit/, CLAUDE.md, docs/, tests workflow).
- From PR #22 (lcdbezerra): README uses real env ids; numpy int actions accepted; obs returned as copies. Dropped its pickle maze loader and exact pins; pins now `gym<0.26`, `numpy<2`.
- From PR #15 (Kuzat): `maze-random-custom-v0` / `MazeEnvRandomCustom`, `zero_step_rewards`, `env.restore()`, int obs. Dropped its setup.py url/version change and README CRLF rewrite.
- Fixes #20 (replaces PR #21, JupiLogy): `enable_render=False` no longer crashes in `render()`; returns None.
- PR #18 closed (IDE files, unrelated scripts).
- Added tests/test_maze_env.py. setup.py version 0.5.
