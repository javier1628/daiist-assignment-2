"""
Survey candidate Roboflow Universe datasets before choosing one.

For each candidate project, download the latest version in YOLOv11 format into
data/candidates/<slug>/ and compute, from the label files on disk and not from
the website:

- total images and train/valid/test counts
- instances per class per split
- median and 10th-percentile box area as a fraction of image area, per class
- images with zero labels, per split

Writes data/candidates/SURVEY.md and, per dataset, a 2x2 grid of random
labelled images at data/candidates/<slug>/preview.jpg.

Projects that are private, not object detection, or fail to download are
skipped and listed with the reason. The script does not recommend a dataset.

Run:

    uv run python scripts/survey_datasets.py            # reuses existing downloads
    uv run python scripts/survey_datasets.py --refresh  # re-downloads everything
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api_key import load_api_key  # noqa: E402

SEED = 42
OUT_DIR = ROOT / "data" / "candidates"
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANDIDATES = [
    "dms-8pnzm/package-kglu1-we3hv",
    "project-33xgh/damaged-box-detection",
    "infocomm-project-fpj1z/damaged-packages-6snht",
    "nani-tmzf6/package-detection-5ozpr",
    "roboflow-ngkro/package-detection-e1ssd",
    "damages-shoeg/damaged_boxes-3oeks",
]


@dataclass
class ClassStats:
    per_split: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    areas: list[float] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.per_split.values())


@dataclass
class SplitStats:
    images: int = 0
    labelled: int = 0
    zero_label: int = 0
    instances: int = 0


@dataclass
class Survey:
    slug: str
    version: int
    project_type: str
    location: Path
    names: dict[int, str]
    splits: dict[str, SplitStats]
    classes: dict[int, ClassStats]
    preview: Path | None
    license: str = ""

    @property
    def total_images(self) -> int:
        return sum(s.images for s in self.splits.values())


def slug_dir(slug: str) -> str:
    """Directory name for a workspace/project slug."""
    return slug.replace("/", "__")


def latest_version_number(project) -> int:
    versions = project.get_version_information()
    numbers = []
    for entry in versions:
        vid = str(entry.get("id", ""))
        tail = vid.rsplit("/", 1)[-1]
        if tail.isdigit():
            numbers.append(int(tail))
    if not numbers:
        raise RuntimeError("project has no versions")
    return max(numbers)


def parse_label_line(line: str) -> tuple[int, float] | None:
    """Return (class_id, box_area_fraction) for one YOLO label line.

    Handles plain boxes (cls cx cy w h) and polygon lines (cls x1 y1 x2 y2 ...)
    by taking the polygon's bounding box. Coordinates are normalised, so
    w * h is already the fraction of image area.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        w, h = vals[2], vals[3]
    else:
        xs, ys = vals[0::2], vals[1::2]
        if len(xs) != len(ys) or not xs:
            return None
        w, h = max(xs) - min(xs), max(ys) - min(ys)
    return cls, max(w, 0.0) * max(h, 0.0)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Boxes as (cls, cx, cy, w, h) in normalised coordinates, for drawing."""
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            if not xs:
                continue
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        boxes.append((cls, cx, cy, w, h))
    return boxes


def load_names(location: Path) -> dict[int, str]:
    data_yaml = location / "data.yaml"
    if not data_yaml.is_file():
        return {}
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in dict(names).items()}


def analyse(location: Path) -> tuple[dict[str, SplitStats], dict[int, ClassStats], list[Path]]:
    splits: dict[str, SplitStats] = {}
    classes: dict[int, ClassStats] = defaultdict(ClassStats)
    labelled_images: list[Path] = []

    for split in SPLITS:
        img_dir = location / split / "images"
        lbl_dir = location / split / "labels"
        if not img_dir.is_dir():
            continue
        stats = SplitStats()
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stats.images += 1
            label = lbl_dir / (img.stem + ".txt")
            found = 0
            if label.is_file():
                for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                    parsed = parse_label_line(line)
                    if parsed is None:
                        continue
                    cls, area = parsed
                    classes[cls].per_split[split] += 1
                    classes[cls].areas.append(area)
                    found += 1
            stats.instances += found
            if found:
                stats.labelled += 1
                labelled_images.append(img)
            else:
                stats.zero_label += 1
        splits[split] = stats
    return splits, dict(classes), labelled_images


def make_preview(images: list[Path], location: Path, names: dict[int, str], out: Path) -> Path | None:
    if not images:
        return None
    rng = random.Random(SEED)
    picks = rng.sample(images, k=min(4, len(images)))
    cell = 480
    tiles = []
    for img_path in picks:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        split = img_path.parents[1].name
        label = location / split / "labels" / (img_path.stem + ".txt")
        # Resize first, then draw, so line width and text are legible whatever
        # the source resolution.
        scale = cell / max(h, w)
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        rh, rw = resized.shape[:2]
        for cls, cx, cy, bw, bh in read_boxes(label):
            x0, y0 = int((cx - bw / 2) * rw), int((cy - bh / 2) * rh)
            x1, y1 = int((cx + bw / 2) * rw), int((cy + bh / 2) * rh)
            color = tuple(int(c) for c in np.random.RandomState(cls + SEED).randint(60, 255, 3))
            cv2.rectangle(resized, (x0, y0), (x1, y1), color, 2)
            ty = y0 - 6 if y0 - 6 > 14 else min(y1 - 6, rh - 6)
            cv2.putText(resized, names.get(cls, str(cls)), (max(x0, 2), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(resized, names.get(cls, str(cls)), (max(x0, 2), ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        canvas = np.full((cell, cell, 3), 30, dtype=np.uint8)
        canvas[(cell - rh) // 2:(cell - rh) // 2 + rh, (cell - rw) // 2:(cell - rw) // 2 + rw] = resized
        cv2.putText(canvas, f"{split}/{img_path.name}"[:48], (6, cell - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) < 4:
        tiles.append(np.full((cell, cell, 3), 30, dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render(surveys: list[Survey], skipped: list[tuple[str, str]]) -> str:
    lines = [
        "# Candidate dataset survey",
        "",
        "Computed from the YOLOv11 label files on disk by `scripts/survey_datasets.py`,",
        "not from the Universe website. Box area is `w * h` in normalised label",
        "coordinates, which is the fraction of image area. Images with zero labels are",
        "images whose label file is missing or holds no parseable box.",
        "",
        f"Seed {SEED}. Preview grids pick 4 random labelled images.",
        "",
    ]
    if skipped:
        lines += ["## Skipped", ""]
        for slug, reason in skipped:
            lines.append(f"- `{slug}`: {reason}")
        lines.append("")

    for s in surveys:
        lines += [f"## `{s.slug}` (version {s.version})", ""]
        lines.append(f"- Type: {s.project_type}")
        if s.license:
            lines.append(f"- License: {s.license}")
        lines.append(f"- Location: `{s.location.relative_to(ROOT)}`")
        lines.append(f"- Classes ({len(s.names)}): " + ", ".join(s.names[k] for k in sorted(s.names)))
        lines.append(f"- Total images: {s.total_images}")
        lines.append("")
        lines += ["| Split | Images | Labelled | Zero labels | Instances |", "|---|---|---|---|---|"]
        for split in SPLITS:
            st = s.splits.get(split)
            if st is None:
                lines.append(f"| {split} | 0 | 0 | 0 | 0 |")
            else:
                lines.append(f"| {split} | {st.images} | {st.labelled} | {st.zero_label} | {st.instances} |")
        lines.append("")
        lines += [
            "| Class | train | valid | test | total | median box area | p10 box area |",
            "|---|---|---|---|---|---|---|",
        ]
        for cls in sorted(set(s.names) | set(s.classes)):
            cs = s.classes.get(cls, ClassStats())
            name = s.names.get(cls, f"class {cls}")
            med = pct(statistics.median(cs.areas)) if cs.areas else "n/a"
            p10 = pct(float(np.percentile(cs.areas, 10))) if cs.areas else "n/a"
            lines.append(
                f"| {name} | {cs.per_split.get('train', 0)} | {cs.per_split.get('valid', 0)} | "
                f"{cs.per_split.get('test', 0)} | {cs.total} | {med} | {p10} |"
            )
        lines.append("")
        if s.preview:
            lines.append(f"- Preview: `{s.preview.relative_to(ROOT)}`")
        else:
            lines.append("- Preview: none (no labelled images)")
        lines.append("")
    return "\n".join(lines)


def survey_one(rf, slug: str, refresh: bool) -> Survey:
    workspace, project_id = slug.split("/", 1)
    project = rf.workspace(workspace).project(project_id)
    ptype = str(getattr(project, "type", "unknown"))
    if ptype != "object-detection":
        raise RuntimeError(f"project type is {ptype!r}, not object-detection")
    number = latest_version_number(project)
    version = project.version(number)
    dest = OUT_DIR / slug_dir(slug)
    if (dest / "data.yaml").is_file() and not refresh:
        print(f"reusing existing download in {dest.relative_to(ROOT)}", flush=True)
        location = dest
    else:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        dataset = version.download(model_format="yolov11", location=str(dest), overwrite=True)
        location = Path(dataset.location)
    names = load_names(location)
    splits, classes, labelled = analyse(location)
    if not splits:
        raise RuntimeError("download produced no train/valid/test folders")
    preview = make_preview(labelled, location, names, dest / "preview.jpg")
    return Survey(
        slug=slug,
        version=number,
        project_type=ptype,
        location=location,
        names=names,
        splits=splits,
        classes=classes,
        preview=preview,
        license=str(getattr(project, "license", "") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-download datasets even if data/candidates/<slug>/ already holds one",
    )
    refresh = parser.parse_args().refresh
    random.seed(SEED)
    np.random.seed(SEED)
    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    surveys: list[Survey] = []
    skipped: list[tuple[str, str]] = []
    for slug in CANDIDATES:
        print(f"\n=== {slug} ===", flush=True)
        try:
            surveys.append(survey_one(rf, slug, refresh))
            print(f"ok: {surveys[-1].total_images} images", flush=True)
        except Exception as error:  # noqa: BLE001 - any failure means skip and report
            reason = f"{type(error).__name__}: {str(error).strip().splitlines()[0] if str(error).strip() else ''}"
            print(f"skipped: {reason}", flush=True)
            traceback.print_exc(limit=2)
            skipped.append((slug, reason))

    report = OUT_DIR / "SURVEY.md"
    report.write_text(render(surveys, skipped), encoding="utf-8")
    print(f"\nWrote {report.relative_to(ROOT)}: {len(surveys)} surveyed, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
