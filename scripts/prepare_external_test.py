"""Collect a small labeled external test set from a public source.

Source: Hugging Face dataset ``davidshableski/weatherimages`` (Data.zip), which
contains weather photos collected from the web. We keep only the four classes
that overlap with this competition: cloudy / rainy / snowy / sunny.

Usage:
    python scripts/prepare_external_test.py --output-dir data/external_test --max-per-class 60

Note on interpretation: an external public set has a different distribution from
the competition data, so its macro-F1 is a *robustness* signal, not the official
score. The primary >=0.95 target is measured on the held-out test split that
shares the competition distribution.
"""
import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

DATA_URL = "https://huggingface.co/datasets/davidshableski/weatherimages/resolve/main/Data.zip"
KEEP_CLASSES = ("cloudy", "rainy", "snowy", "sunny")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_zip(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Reusing cached archive: {dest}")
        return dest
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"Saved archive to {dest} ({len(data) / 1e6:.1f} MB)")
    return dest


def organize(zip_path: Path, output_dir: Path, max_per_class: int) -> Dict[str, int]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for name in KEEP_CLASSES:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in KEEP_CLASSES}
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        # Map each member to one of the kept classes by looking at its path parts.
        buckets: Dict[str, List[str]] = {name: [] for name in KEEP_CLASSES}
        for member in members:
            if member.endswith("/"):
                continue
            suffix = Path(member).suffix.lower()
            if suffix not in IMG_EXTS:
                continue
            parts = [p.lower() for p in Path(member).parts]
            for name in KEEP_CLASSES:
                if name in parts:
                    buckets[name].append(member)
                    break

        for name in KEEP_CLASSES:
            selected = sorted(buckets[name])
            if max_per_class > 0:
                selected = selected[:max_per_class]
            for i, member in enumerate(selected):
                target = output_dir / name / f"{name}_{i:04d}{Path(member).suffix.lower()}"
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                counts[name] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/external_test")
    parser.add_argument("--cache", default="data/_cache/weatherimages.zip")
    parser.add_argument("--max-per-class", type=int, default=60,
                        help="0 means keep all available images per class")
    args = parser.parse_args()

    try:
        zip_path = download_zip(DATA_URL, Path(args.cache))
    except Exception as exc:  # network/proxy issues are common on graders
        print(f"ERROR downloading external dataset: {exc}", file=sys.stderr)
        print("You can manually place labeled images under "
              f"{args.output_dir}/<class>/*.jpg and re-run evaluation.", file=sys.stderr)
        return 1

    counts = organize(zip_path, Path(args.output_dir), args.max_per_class)
    print(f"External test set ready at {args.output_dir}")
    print(f"Per-class counts: {counts}")
    print(f"Total: {sum(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
