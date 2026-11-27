# Minutes

Notable decisions and session summaries, newest first. One short dated entry each.

## 2026-11-27
- Daily maintenance bot switched on (Matthew OK): scheduled Claude session follows docs/AUTOMATION.md. Auto-merges only low-risk changes with `tests` green; API, deps, security, others' PRs and rule edits get label `needs-approval`. Until Actions billing is fixed nothing auto-merges.
- Moved to gymnasium>=1.0 (v1.0): old gym dead, numpy<2 won't install on Python 3.13. Breaking API change accepted; old-API users stay on pre-1.0. Added MIT LICENSE (#17). #10 closed as stale.
- Open PRs #15, #18, #21, #22 resolved via one combined PR; good parts kept with author credit. Pickle loading rejected (runs code from file). Dependency pins kept loose (`gym<0.26`, `numpy<2`), not exact.
