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

## Dataset pinned (2026-10-23)

- Chosen by the owner: Roboflow Universe `roboflow-ngkro/package-detection-e1ssd/6`
  (workspace `roboflow-ngkro`, project `package-detection-e1ssd`, version 6,
  YOLOv11 export). Pinned in `src/data.py`; `get_dataset()` downloads to
  `data/raw/` once and returns `data.yaml`.
- The other candidate downloads were deleted from `data/candidates/`; only the
  chosen one and its preview remain there.
- Decision classes, set by the owner: positive ("compromised") = Box_broken or
  Open_package; negative ("intact") = Box or Package. An image is positive if it
  holds at least one positive box. Constants live in `src/data.py`.
- Project is public on Universe (API `public: True`), but the export's
  `data.yaml` license field says `Private`. Worth stating in the report.

## Dataset description (2026-10-23)

- `scripts/describe_dataset.py` writes `outputs/dataset_stats.json`,
  `outputs/dataset_stats.md` and `outputs/box_area_hist.png`.
- Roboflow version 6 metadata: preprocessing = auto-orient plus resize to
  640x640 stretch; augmentation field empty, so no augmentation. All 1200
  images are 640x640 on disk. No field says how the split was made; the exact
  70/20/10 ratio (840/240/120) matches Roboflow's default, so a Roboflow random
  split is the likely explanation. Inference, not fact.
- Per split: train 840 images / 1052 boxes, valid 240 / 305, test 120 / 136.
  Instances per class (train/valid/test): Box 314/103/31, Box_broken 206/64/34,
  Open_package 225/54/29, Package 307/84/42.
- Image-level balance: train 424 positive vs 416 negative, valid 118 vs 122,
  test 62 vs 58. Roughly 50/50 in every split, so the dataset base rate is not
  a production base rate. Only 5 images mix positive and negative boxes.
- Zero-label images: none. Boxes per image: mean about 1.2, max 11.
- Box area: medians Box 25.6%, Box_broken 29.6%, Open_package 38.7%, Package
  45.6% of image area. p10 is 8 to 19%, so objects are large; not a small-object
  problem.
- Near-duplicates (pHash, Hamming <= 4): 24 pairs, 14 of them exact hash
  matches. Within train 11, none within valid or test. Across splits: 13 pairs
  (train/valid 7, test/valid 4, test/train 2). So some test images have a
  near-identical twin in train or valid, which inflates test scores slightly.
  Example pairs with filenames are in `outputs/dataset_stats.md`.
- Duplicate check verified at pixel level: 18 of the 24 flagged pairs differ by
  a mean of under 5 grey levels, so they are re-uploads, not pHash
  false positives. The twins carry different label files (re-annotated copies).
  6 of the 120 test images have a twin in train or valid. Keep this in mind
  when reading test metrics; dropping them is an option the owner decides.
