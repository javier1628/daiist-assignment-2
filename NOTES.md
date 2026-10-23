# Notes

Running log of decisions, bugs and numbers. Bullets, newest section last.

## Setup (2026-10-23)

- Environment: Claude cloud container, no GPU. `uv sync` pins the CPU build of
  PyTorch (2.11.0+cpu). Ultralytics 8.4.173, Gradio 6.29.1.
- First `uv sync` failed: the container's network policy refused
  `download-r2.pytorch.org`. Fixed by allowlisting the PyTorch, Roboflow and
  GitHub release hosts in the environment settings. Sync now succeeds.
- Roboflow key is supplied as the `ROBOFLOW_API_KEY` environment variable in the
  cloud environment settings, not via `.env`. `api_key.py` checks the variable
  first. Verified: Roboflow accepts it, workspace `javialonsov03-gmail-com`.
- A `.env` with the placeholder value exists locally and is git-ignored. The
  environment variable takes precedence, so it is inert.
- Repo ships both `train.py` and `train.ipynb`, and both `app.py` and
  `app.ipynb`. `main.py` refuses to run a stage while both exist. The notebooks
  must be deleted before either stage can run. Not yet done.
- `check.py` loads every `*.pt` under `weights/` and launches `main.py app` on
  port 7860. `weights/` currently holds only `.gitkeep`.
- Training plan: `train.py` is the real pipeline. Training itself runs on
  Google Colab (T4) because this container has no GPU. Weights are committed
  from Colab as `weights/best.pt`, then pulled here for the sweep, review band
  and dashboard.
- Branching: all work happens on `claude/funny-rubin-b746w8`. Before the final
  PR to ami232/daiist-assignment-2, merge that branch into the fork's `main`
  and open the PR from `main`. To be done at the end, on instruction.
