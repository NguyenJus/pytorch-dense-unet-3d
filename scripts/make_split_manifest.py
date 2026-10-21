#!/usr/bin/env python3
"""Generate a tumor-stratified train/validation split manifest for LiTS.

Cases are sorted by tumor voxel count and every ``round(1/fraction)``-th case,
from a seed-derived offset, goes to validation.  ``--link-root`` additionally
writes symlink trees ``<root>/train`` and ``<root>/validation`` so the existing
dataset discovery and split-manifest validation work unchanged.  Pairs that fail
the loader's image/mask shape or affine check are excluded before stratification
and recorded under ``excluded`` with the reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from dense_unet_3d.dataset.LITSDataset import _case_id, _validate_spatial_pair, discover_pairs


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_manifest(
    pairs: list[tuple[str, str]], *, validation_fraction: float, seed: int
) -> dict[str, Any]:
    """Return the split manifest for ``(volume, segmentation)`` pairs."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for volume, segmentation in pairs:
        volume_img, segmentation_img = nib.load(volume), nib.load(segmentation)
        try:
            _validate_spatial_pair(volume_img, segmentation_img, volume)  # type: ignore[arg-type]
        except ValueError as error:
            excluded.append(
                {
                    "id": _case_id(volume, "volume"),
                    "reason": str(error),
                    "volume_sha256": _sha256(volume),
                    "segmentation_sha256": _sha256(segmentation),
                }
            )
            continue
        labels = np.asanyarray(segmentation_img.dataobj)  # type: ignore[attr-defined]
        tumor = int((labels == 2).sum())
        rows.append(
            {
                "id": _case_id(volume, "volume"),
                "depth": int(labels.shape[2]),
                "tumor_voxels": tumor,
                "liver_voxels": int((labels == 1).sum()),
                "has_tumor": tumor > 0,
                "volume_sha256": _sha256(volume),
                "segmentation_sha256": _sha256(segmentation),
            }
        )
    rows.sort(key=lambda row: (row["tumor_voxels"], row["id"]))
    count = math.floor(len(rows) * validation_fraction + 1e-9)
    stride = round(1 / validation_fraction)
    offset = random.Random(seed).randrange(stride)
    chosen = set(range(offset, len(rows), stride)[:count])
    if len(chosen) != count:
        raise ValueError(
            f"validation fraction {validation_fraction} gives {len(chosen)} every-{stride}th "
            f"positions, not floor(N * fraction) = {count}"
        )
    for position, row in enumerate(rows):
        row["split"] = "validation" if position in chosen else "train"
    case_ids = {
        split: sorted(str(row["id"]) for row in rows if row["split"] == split)
        for split in ("train", "validation")
    }
    per_case = sorted(rows, key=lambda row: str(row["id"]))
    return {
        "case_ids": case_ids,
        "counts": {split: len(ids) for split, ids in case_ids.items()},
        "per_case": [
            {
                key: row[key]
                for key in (
                    "id",
                    "split",
                    "depth",
                    "tumor_voxels",
                    "liver_voxels",
                    "has_tumor",
                    "volume_sha256",
                    "segmentation_sha256",
                )
            }
            for row in per_case
        ],
        "excluded": sorted(excluded, key=lambda row: str(row["id"])),
        "seed": seed,
        "stratification": f"tumor_voxel_count_every_{stride}th",
        "validation_fraction": validation_fraction,
        "generator_revision": _generator_revision(),
    }


def link_split(
    manifest: dict[str, Any], pairs: list[tuple[str, str]], link_root: str | os.PathLike[str]
) -> None:
    """Write absolute symlink trees ``<root>/{train,validation}`` for the manifest."""
    root = Path(link_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"link root is not empty: {root}")
    split_of = {case_id: split for split, ids in manifest["case_ids"].items() for case_id in ids}
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True, exist_ok=True)
    for volume, segmentation in pairs:
        case_split = split_of.get(_case_id(volume, "volume"))
        if case_split is None:  # excluded from the manifest
            continue
        for source in (volume, segmentation):
            os.symlink(os.path.abspath(source), root / case_split / os.path.basename(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--link-root")
    parser.add_argument("img_dirs", nargs="+", metavar="DIR")
    args = parser.parse_args()
    if args.link_root is not None:
        root = Path(args.link_root)
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"link root is not empty: {root}")
    pairs = discover_pairs(args.img_dirs)
    manifest = build_manifest(pairs, validation_fraction=args.validation_fraction, seed=args.seed)
    if args.link_root is not None:
        link_split(manifest, pairs, args.link_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
