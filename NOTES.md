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
- Branching: all work happens on `assignment-2`. Before the final
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

## Business figures and scoring (2026-10-23)

- `src/config.py` holds every owner-supplied figure. Costs in DOP per unit:
  FN 7000 (replacement 4400 + two-way freight 2500, claim window closed), FP 40
  (10-min unpack at RD$3/min), human review 40 (same unpack), model-FAIL
  confirmation 10 (3-min glance, photo is the claim evidence). Wage RD$3/min.
  Base rate 1%, sweep [0.5%, 1%, 3%, 5%, 10%, 15%]. Containers 20ft 200,
  40ft_low 400, 40ft_high 600 units; receiving is per container, no daily
  flow. Staff 2, 7 productive hours/day, 7-day claim window. DOP per EUR
  68.67 (poundsterlinglive.com, 2026-10-23).
- The class lists moved from `src/data.py` into `src/config.py`; `data.py`
  re-exports them so the describe script is unchanged.
- `src/scoring.py`: `image_score` = max confidence over positive-class
  detections, resolved by class name via `results.names`, 0.0 if none.
  `decide(score, t_low, t_high)`: `score >= t_high` fail, `score >= t_low`
  review, else pass. Ties go to the stricter outcome; `t_low == t_high`
  gives no review band. `image_label` reads a YOLO label file and returns
  True if any box is a positive class.
- `tests/test_scoring.py`: 14 tests, all passing. Covers the five required
  cases plus name-based resolution with a shuffled index map, bad threshold
  order, and that the config class lists are disjoint and cover the dataset.

## Training pipeline and evaluation code (2026-10-23)

- Branch renamed from `claude/funny-rubin-b746w8` to `assignment-2` locally and on
  origin. The old remote branch could not be deleted from this session (the push
  proxy returns 403 on deletions); the owner deletes it on GitHub.
- `train.ipynb` deleted. `train.py` is the pipeline: `ensure_dataset()`, finetune
  `yolo11n.pt` with imgsz 640, epochs 50, batch 16, seed 42, deterministic,
  patience 10, project `runs`, name `finetune`; copy best checkpoint to
  `weights/best.pt`; then `src.evaluate`. Device, versions, settings and
  wall-clock go to `outputs/train_log.json` and are appended to NOTES.md by
  the run itself. The log is written before evaluation so a failure there
  cannot lose it.
- `src/data.py` gained `ensure_dataset()` (old `get_dataset` kept as alias) and
  `normalise_data_yaml()`: Roboflow writes `train: ../train/images`, which
  Ultralytics resolves against its own datasets directory, not the yaml, so
  the yaml now carries an absolute `path` and split folders relative to it.
- `src/evaluate.py`: detection metrics via `model.val` twice per subset, conf
  0.001 for mAP and conf 0.25 for the confusion matrix, because in
  ultralytics 8.4 the matrix uses the validator's own conf and a 0.001 matrix
  would flag everything. Image scores at conf 0.001 via `src.scoring` to
  `outputs/test_scores.csv` with filename, score, label, main-box class and a
  `dup_in_trainval` flag. TPR/FPR at 0.25 and 0.50 with FPR split by Box vs
  Package. Everything computed on all 120 test images and on the 114 without a
  twin in train/valid, taking the pair list from `outputs/dataset_stats.json`.
  Baselines: COCO yolo11n zero-shot (class histogram at conf 0.25, every image
  scored 0.0, no class mapping invented) and trivial all-pass / all-fail, to
  `outputs/baselines.json`.
- Not yet exercised end to end: a smoke test of `evaluate.py` with a
  random-weight 4-class checkpoint was prepared but not run. First real run
  is the Colab training; `train.py` keeps the weights even if evaluation
  fails.
- `notebooks/train_colab.ipynb` (outside the repo root so `main.py` ignores
  it): GPU check, clone the fork at `assignment-2`, pip install
  ultralytics==8.4.173 and roboflow, key from Colab secrets, `python main.py
  train`, zip `weights/best.pt` + `outputs/` and download.
- `.gitignore`: root-level `*.pt` ignored (the `yolo11n.pt` Ultralytics
  downloads); `weights/best.pt` stays committed.
- Smoke test of `src/evaluate.py` done with a random-weight 4-class checkpoint
  built in the scratchpad (class head biased toward Box_broken so positive
  scores appear). No training. Ran end to end on CPU in about 70 s: all-120
  and dedup-114 paths, both confusion matrices, `test_scores.csv` with the
  right columns and 6 `dup_in_trainval` rows, and both baselines.
  Two bugs fixed: (1) ultralytics 8.4 only fills the confusion matrix when
  `plots=True`, so the matrix pass now sets it (its PNGs go to `runs/`,
  git-ignored); (2) CSV writers used CRLF line endings, now LF. The scratch
  checkpoint and every file it produced were deleted; nothing from random
  weights is committed.

## Colab run 1 failure and fixes (2026-10-25)

- Bug: `train.py` hard-coded the run directory as `runs/finetune/`, but with
  `project="runs"` Ultralytics nested it under `runs/detect/runs/finetune/`,
  so after a complete 50-epoch Colab run the copy to `weights/best.pt` failed.
  Fix: take the location from the trainer's return value
  (`results.save_dir / "weights" / "best.pt"`), with `model.trainer.save_dir`
  as fallback; the run directory is now logged in `train_log.json`.
  `tests/test_train.py` fakes a nested save_dir and checks the copy.
- Ultralytics' AMP check on GPU downloads `yolo26n.pt` into `weights/`.
  `check.py` loads every `.pt` there, so `train.py` now deletes
  `weights/yolo26n.pt` (and `yolo11n.pt`, same mechanism in older versions)
  after training, logs what it removed, and both are git-ignored.
- Evaluation confidences, stated for the report: per-class P/R/AP and mAP
  come from a validation pass at conf 0.001 (standard for mAP); the confusion
  matrix comes from a second pass at conf 0.25; image scores keep every
  detection at conf 0.001. The two passes therefore report different numbers
  by design. `test_metrics_finetuned.json` records `map_conf`,
  `confusion_matrix_conf` and a `conf_note` at top level and per subset.

## Training run (2026-10-25)

- Dataset `roboflow-ngkro/package-detection-e1ssd/6`, base weights `yolo11n.pt`.
- Device: Tesla T4 (0). ultralytics 8.4.173, torch 2.11.0+cu130, python 3.13.16.
- Settings: imgsz=640, epochs=50, batch=16, seed=42, deterministic=True, patience=10, project=runs, name=finetune, exist_ok=True.
- Epochs run: 50 of 50 (patience 10). Best epoch: 50.
- Wall-clock: training 652s, evaluation 19s, total 681s.
- Run directory: `/content/daiist-assignment-2/runs/detect/runs/finetune`. Best checkpoint copied to `weights/best.pt` (5.5 MB).

## Colab run 2: results landed and reproduced (2026-10-25)

- The Colab artifacts were uploaded through the GitHub web UI onto the fork's
  `main` (two "Add files via upload" commits: `weights/best.pt`, then
  `outputs/`), not onto `assignment-2`. Brought onto `assignment-2` with
  `git checkout origin/main -- weights/best.pt outputs/`. `main` was left as is.
- `weights/best.pt` is 5,474,010 bytes; `outputs/` holds 12 files including
  `train_log.json`. Training: Tesla T4, 50 of 50 epochs in 652 s, best epoch
  50, so early stopping never triggered and the model was still improving at
  the last epoch. Worth stating in the report; a longer run is an option the
  owner decides. The nested `runs/detect/runs/finetune` path and the
  `yolo26n.pt` removal both behaved as fixed.
- Headline test numbers (GPU run, conf 0.001 for mAP): all 120 images mAP50
  0.9678, mAP50-95 0.8884; dedup 114 images mAP50 0.965, mAP50-95 0.8836.
  Image level: TPR 0.9516 / FPR 0.069 at 0.25, TPR 0.9355 / FPR 0.0517 at
  0.50 (all); TPR 0.9483 / FPR 0.0714 at 0.25, TPR 0.931 / FPR 0.0536 at 0.50
  (dedup).
- CPU reproduction of `src.evaluate` against the committed GPU outputs:
  `test_metrics_finetuned.json` identical except the `device` field;
  `baselines.json` identical; both confusion-matrix CSVs identical;
  `test_scores.csv` identical in labels and classes, 12 of 120 scores differ
  in the sixth decimal (max difference 0.000002), no pass/fail flip at 0.25
  or 0.50; `baseline_coco_detections.csv` has the same 91 detections with
  identical confidences, only row order within an image differs. The
  committed GPU files stay as the canonical outputs.
- `check.py`: key check and weights check pass (4 classes load). Step 3 fails
  because `app.py` and `app.ipynb` both still exist and `app.py` is the
  template; expected until the dashboard is built.
