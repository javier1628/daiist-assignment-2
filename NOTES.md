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

## Dataset survey (2026-10-23)

- `scripts/survey_datasets.py` downloads each candidate's latest version in
  YOLOv11 format to `data/candidates/<workspace>__<project>/` and computes
  stats from the label files. Report: `data/candidates/SURVEY.md` (git-ignored
  with the rest of `data/`). Seed 42 for the preview picks. Reruns reuse
  existing downloads unless `--refresh` is passed.
- Skipped `project-33xgh/damaged-box-detection`: it is a classification
  project, not object detection.
- Five object-detection candidates surveyed. Headline facts from the labels:
  - `dms-8pnzm/package-kglu1-we3hv` v2: 1169 images, 1023/73/73 split, classes
    Damaged (1402 instances) and package (1175). Damaged boxes are tiny: median
    0.96% of image area, p10 0.23%. The exported images are grayscale, so the
    version has a grayscale preprocessing step baked in. No zero-label images.
  - `infocomm-project-fpj1z/damaged-packages-6snht` v2: 1303 images,
    1149/108/46 split. "damaged package" 3099 instances vs "undamaged" 517, so
    the negative class is the minority. Test split is small (46 images, 121
    instances, 19 undamaged). Median box area 11.7% / 7.6%.
  - `nani-tmzf6/package-detection-5ozpr` v2: 2406 images, 1679/476/251 split.
    "damaged" 2051 vs "no damaged" 386. Boxes cover most of the image (median
    72% / 58%), with roughly one box per image, so it behaves like
    classification with boxes. 9 images with zero labels.
  - `roboflow-ngkro/package-detection-e1ssd` v6: 1200 images, 840/240/120
    split, 4 classes (Box 448, Box_broken 304, Open_package 308, Package 433).
    Balanced, mid-size boxes (median 26 to 46%). Test split per class is 29 to
    42 instances. Mix of boxes and mailer envelopes.
  - `damages-shoeg/damaged_boxes-3oeks` v3: 219 images, train split only, no
    valid or test. Frames from a handful of videos of one product (Nescafe
    cartons), so images are near-duplicates. Classes buckled 8, open 10, tear
    149, well 54.
- Bug fixed during the run: preview boxes were drawn before resizing, so on
  high-resolution video frames the lines and labels vanished. Now draws after
  resizing.
- Dataset choice is the owner's. No recommendation recorded.
