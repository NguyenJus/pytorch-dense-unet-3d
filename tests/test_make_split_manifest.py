import hashlib
import json
import os
import random
import sys

import nibabel as nib
import numpy as np
import pytest

from dense_unet_3d.dataset.LITSDataset import discover_pairs
from dense_unet_3d.dataset.prepare_dataset import _validate_split_manifest
from scripts import make_split_manifest as msm

TUMOR_COUNTS = [5, 0, 9, 3, 7, 1, 8, 2, 6, 4]


def _write_tree(root, tumor_counts=TUMOR_COUNTS):
    """Two batch dirs; case k has k+1 liver voxels and tumor_counts[k] tumor voxels."""
    dirs = [root / "batch1", root / "batch2"]
    for d in dirs:
        d.mkdir(parents=True)
    for k, tumor in enumerate(tumor_counts):
        seg = np.zeros((4, 4, 3 + k % 2), dtype=np.int16)
        flat = seg.reshape(-1)
        flat[: k + 1] = 1
        if tumor:
            flat[-tumor:] = 2
        vol = np.full(seg.shape, float(k), dtype=np.float32)
        d = dirs[k % 2]
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(d / f"volume-{k}.nii"))
        nib.save(nib.Nifti1Image(seg, np.eye(4)), str(d / f"segmentation-{k}.nii"))
    return [str(d) for d in dirs]


def _sha(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


@pytest.fixture()
def tree(tmp_path):
    return _write_tree(tmp_path / "src")


def _build(tree, seed=7, fraction=0.2):
    return msm.build_manifest(discover_pairs(tree), validation_fraction=fraction, seed=seed)


def test_every_fifth_position_and_floor_count(tree):
    manifest = _build(tree)
    order = sorted(range(10), key=lambda k: (TUMOR_COUNTS[k], f"-{k}"))
    offset = random.Random(7).randrange(5)
    expected = {f"-{order[p]}" for p in (offset, offset + 5)}
    assert set(manifest["case_ids"]["validation"]) == expected
    assert manifest["counts"] == {"train": 8, "validation": 2}
    assert len(manifest["case_ids"]["train"]) == 8
    assert manifest["case_ids"]["validation"] == sorted(expected)


def test_positions_spread_over_whole_range_when_stride_not_integral(tree):
    # 10 * 0.3 = 3 validation cases; floor(N / count) = 3 -> offset in [0, 3), positions o + k*10//3.
    manifest = _build(tree, fraction=0.3)
    order = sorted(range(10), key=lambda k: (TUMOR_COUNTS[k], f"-{k}"))
    offset = random.Random(7).randrange(3)
    expected = {f"-{order[offset + k * 10 // 3]}" for k in range(3)}
    assert set(manifest["case_ids"]["validation"]) == expected
    assert manifest["stratification"] == "tumor_voxel_count_every_3th"


def test_fraction_above_half_yields_floor_count(tree):
    # floor(10 * 0.6) = 6; spread positions reach all 6 (no truncation error).
    manifest = _build(tree, fraction=0.6)
    assert manifest["counts"] == {"train": 4, "validation": 6}


def test_real_run_positions_pinned():
    # N=126, fraction 0.2, seed 20261021 -> offset 4, positions 4, 9, ..., 124.
    assert random.Random(20261021).randrange(5) == 4
    assert msm._validation_positions(126, 0.2, 20261021) == list(range(4, 125, 5))


def test_tumor_ties_break_on_case_id(tmp_path):
    counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    manifest = msm.build_manifest(
        discover_pairs(_write_tree(tmp_path / "ties", counts)), validation_fraction=0.2, seed=7
    )
    order = sorted(f"-{k}" for k in range(10))
    offset = random.Random(7).randrange(5)
    assert manifest["case_ids"]["validation"] == sorted([order[offset], order[offset + 5]])


def test_deterministic_and_schema(tree):
    a, b = _build(tree), _build(tree)
    assert a == b
    assert a["seed"] == 7
    assert a["stratification"] == "tumor_voxel_count_every_5th"
    assert a["validation_fraction"] == 0.2
    assert isinstance(a["generator_revision"], str) and a["generator_revision"]
    assert len(a["per_case"]) == 10
    rows = {r["id"]: r for r in a["per_case"]}
    row = rows["-2"]
    assert set(row) == {
        "id",
        "split",
        "depth",
        "tumor_voxels",
        "liver_voxels",
        "has_tumor",
        "volume_sha256",
        "segmentation_sha256",
    }
    assert row["tumor_voxels"] == 9 and row["liver_voxels"] == 3
    assert row["depth"] == 3 and row["has_tumor"] is True
    assert row["split"] in ("train", "validation")
    assert row["volume_sha256"] == _sha(os.path.join(tree[0], "volume-2.nii"))
    assert row["segmentation_sha256"] == _sha(os.path.join(tree[0], "segmentation-2.nii"))
    assert rows["-1"]["has_tumor"] is False and rows["-1"]["tumor_voxels"] == 0


def test_seed_changes_offset(tree):
    sets = {tuple(_build(tree, seed=s)["case_ids"]["validation"]) for s in range(20)}
    assert len(sets) > 1


def test_cli_and_link_trees(tree, tmp_path, monkeypatch):
    out = tmp_path / "manifest.json"
    root = tmp_path / "links"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_split_manifest.py",
            "--seed",
            "7",
            "--validation-fraction",
            "0.2",
            "--output",
            str(out),
            "--link-root",
            str(root),
            *tree,
        ],
    )
    msm.main()
    manifest = json.loads(out.read_text())
    assert manifest == _build(tree)
    for split in ("train", "validation"):
        pairs = discover_pairs([str(root / split)])
        assert len(pairs) == manifest["counts"][split]
        for volume, seg in pairs:
            assert os.path.islink(volume) and os.path.isabs(os.readlink(volume))
            assert os.path.isfile(seg)
        config = {"experiment": {"split_manifest": str(out)}}
        _validate_split_manifest(config, split, pairs)


def test_link_root_must_be_empty(tree, tmp_path):
    root = tmp_path / "links"
    root.mkdir()
    (root / "stale").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        msm.link_split(_build(tree), discover_pairs(tree), root)


def test_cli_rejects_non_empty_link_root_before_hashing(tree, tmp_path, monkeypatch):
    root = tmp_path / "links"
    root.mkdir()
    (root / "stale").write_text("x")
    monkeypatch.setattr(msm, "build_manifest", lambda *a, **k: pytest.fail("hashed first"))
    argv = ["make_split_manifest.py", "--seed", "7", "--validation-fraction", "0.2"]
    argv += ["--output", str(tmp_path / "m.json"), "--link-root", str(root), *tree]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="not empty"):
        msm.main()


def test_affine_mismatch_excluded_and_not_linked(tree, tmp_path):
    seg_path = os.path.join(tree[1], "segmentation-3.nii")
    labels = np.asanyarray(nib.load(seg_path).dataobj)
    nib.save(nib.Nifti1Image(labels, np.diag([2.0, 1.0, 1.0, 1.0])), seg_path)
    manifest = _build(tree)
    assert [row["id"] for row in manifest["excluded"]] == ["-3"]
    assert "affine mismatch" in manifest["excluded"][0]["reason"]
    assert manifest["excluded"][0]["segmentation_sha256"] == _sha(seg_path)
    assert "-3" not in {row["id"] for row in manifest["per_case"]}
    assert sum(manifest["counts"].values()) == 9
    root = tmp_path / "links"
    msm.link_split(manifest, discover_pairs(tree), root)
    linked = [
        os.path.basename(volume)
        for split in ("train", "validation")
        for volume, _ in discover_pairs([str(root / split)])
    ]
    assert "volume-3.nii" not in linked and len(linked) == 9
