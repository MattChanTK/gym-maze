# Testing

Each suite: what it covers, when it runs, what it blocks. `kit/tools/check.sh` runs the fast ones (kit tests + `KIT_CHECKS` in kit.env).

- `python3 -m unittest discover -s tests`: env step/reset/render, headless mode, custom size, restore. Runs in `kit/tools/check.sh` and the `tests` workflow on every PR. Needs `pip install 'gym<0.26' pygame 'numpy<2'`.
