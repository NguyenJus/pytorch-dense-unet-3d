"""Foreground-centred crop sampler: geometry, randomness transport and item contract."""

import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dense_unet_3d.dataset.foreground_crops import (
    CropSampler,
    ForegroundCropDataset,
    crop_config,
)
from dense_unet_3d.dataset.slabs import NativeSlabDataset
from dense_unet_3d.training.train import unpack_training_batch


def dataset_config(**update):
    return {
        "resize_dims": {"D": 4, "H": 8, "W": 8},
        "clamp_hu": False,
        "sampling": "native_slabs",
        "resize_img": False,
        "inplane_representation": "native_tiles_v1",
        "training_representation": "foreground_crops_v1",
        "passes_per_epoch": 3,
        "tumor_centre_probability": 0.5,
        "foreground_voxel_cache": 50,
        "sampling_seed": 7,
        "random_hflip": True,
        "random_hflip_probability": 0.5,
        "scale_img": True,
        "scale_img_range": {"min": 0.8, "max": 1.2},
        **update,
    }


def write_case(root: Path, name: str, labels: np.ndarray, image=None, affine=None):
    affine = np.eye(4) if affine is None else affine
    image = labels.astype(np.float32) if image is None else image.astype(np.float32)
    volume, target = root / f"volume-{name}.nii", root / f"segmentation-{name}.nii"
    nib.save(nib.Nifti1Image(image, affine), volume)
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), affine), target)
    return str(volume), str(target)


def test_crop_config_geometry_and_disabled_augmentation():
    geometry = crop_config(dataset_config())
    assert geometry["representation"] == "foreground_crops_v1"
    assert (geometry["window"], geometry["height"], geometry["width"]) == (4, 8, 8)
    assert geometry["passes_per_epoch"] == 3
    assert geometry["tumor_centre_probability"] == 0.5
    assert geometry["foreground_voxel_cache"] == 50
    assert geometry["scale_range"] == [0.8, 1.2]
    assert geometry["hflip_probability"] == 0.5
    assert geometry["sampling_seed"] == 7
    assert geometry["image_interpolation"] == "trilinear_half_pixel"
    assert geometry["mask_interpolation"] == "nearest_exact_half_pixel"
    assert geometry["padding_target"] == -100
    disabled = crop_config(dataset_config(random_hflip=False, scale_img=False))
    assert disabled["scale_range"] == [1.0, 1.0]
    assert disabled["hflip_probability"] == 0.0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"training_representation": None}, "foreground_crops_v1"),
        ({"passes_per_epoch": 0}, "passes_per_epoch"),
        ({"passes_per_epoch": True}, "passes_per_epoch"),
        ({"tumor_centre_probability": 1.5}, "tumor_centre_probability"),
        ({"tumor_centre_probability": None}, "tumor_centre_probability"),
        ({"foreground_voxel_cache": 0}, "foreground_voxel_cache"),
        ({"sampling_seed": -1}, "sampling_seed"),
        ({"sampling_seed": 1.0}, "sampling_seed"),
        ({"random_hflip_probability": "half"}, "random_hflip_probability"),
        ({"scale_img_range": {"min": 1.2, "max": 0.8}}, "scale_img_range"),
        ({"scale_img_range": {"min": 0, "max": 1.2}}, "scale_img_range"),
        ({"inplane_representation": "full_fov_resize", "resize_img": True}, "requires"),
    ],
)
def test_crop_config_rejections(update, message):
    with pytest.raises(ValueError, match=message):
        crop_config(dataset_config(**update))


def test_crop_sampler_draws_whole_epoch_at_iterator_start_and_replays():
    generator = torch.Generator().manual_seed(11)
    state = generator.get_state()
    sampler = CropSampler(6, generator)
    assert len(sampler) == 6 and sampler.generator is generator
    iterator = iter(sampler)
    after_start = generator.get_state()
    first = list(iterator)
    assert torch.equal(generator.get_state(), after_start)
    assert sorted(index for index, _ in first) == list(range(6))
    assert all(type(seed) is int and 0 <= seed < 2**63 for _, seed in first)
    assert len({seed for _, seed in first}) == 6
    second = list(sampler)
    assert second != first
    restored = torch.Generator()
    restored.set_state(state)
    assert list(CropSampler(6, restored)) == first
    assert list(CropSampler(6, restored)) == second


def tumor_labels(shape=(20, 18, 10)):
    labels = np.zeros(shape, dtype=np.uint8)
    labels[4:16, 3:15, 2:9] = 1
    labels[6:8, 5:7, 4:6] = 2
    return labels


def test_tumor_centre_scale_and_flip_statistics(tmp_path):
    labels = tumor_labels()
    write_case(tmp_path, "1", labels)
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config())
    assert len(dataset) == 3
    seeds = torch.randint(0, 2**63 - 1, (1000,), generator=torch.Generator().manual_seed(3))
    draws = [dataset.crop_parameters(0, seed) for seed in seeds.tolist()]
    centres = [labels[tuple(p["centre_hwd"])] for p in draws]
    assert all(c >= 1 for c in centres)
    tumor_fraction = sum(c == 2 for c in centres) / len(draws)
    assert tumor_fraction >= 0.45
    assert all(labels[tuple(p["centre_hwd"])] == 2 for p in draws if p["from_tumor"])
    assert 0.45 <= sum(p["from_tumor"] for p in draws) / len(draws) <= 0.55
    assert all(0.8 <= p["scale"] <= 1.2 for p in draws)
    assert len({p["scale"] for p in draws}) > 100
    assert {p["flip"] for p in draws} == {True, False}
    assert all(p["case_index"] == 0 for p in draws)
    extents = {tuple(p["extent"]) for p in draws}
    assert {e[0] for e in extents} == {4} and {e[1] for e in extents} <= set(range(6, 11))
    assert dataset.crop_parameters(2, 123) == dataset.crop_parameters(2, 123)
    assert len({str(p) for p in draws}) == len(draws)


def test_liver_only_case_and_no_foreground_rejection(tmp_path):
    labels = tumor_labels()
    labels[labels == 2] = 1
    write_case(tmp_path, "1", labels)
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config(tumor_centre_probability=1.0))
    draws = [dataset.crop_parameters(0, seed) for seed in range(200)]
    assert not any(p["from_tumor"] for p in draws)
    assert all(labels[tuple(p["centre_hwd"])] == 1 for p in draws)
    empty = tmp_path / "empty"
    empty.mkdir()
    write_case(empty, "9", np.zeros((6, 6, 5), np.uint8))
    with pytest.raises(ValueError, match="volume-9"):
        ForegroundCropDataset([str(empty)], dataset_config())


def test_constructor_and_key_contract(tmp_path):
    write_case(tmp_path, "1", tumor_labels())
    write_case(tmp_path, "2", tumor_labels())
    with pytest.raises(ValueError, match="three-class"):
        ForegroundCropDataset([str(tmp_path)], dataset_config(), detect_tumors=False)
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config())
    assert len(dataset) == 6 and len(dataset.cases) == 2
    assert [dataset.crop_parameters(i, 5)["case_index"] for i in range(6)] == [0, 1] * 3
    with pytest.raises(TypeError):
        dataset[0]
    with pytest.raises(IndexError):
        dataset.crop_parameters(6, 5)


def linear_image(shape):
    h, w, d = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return (h + 100 * w + 10000 * d).astype(np.float32)


def test_item_contract_matches_native_tiles_and_unpacks(tmp_path):
    labels = tumor_labels()
    write_case(tmp_path, "1", labels)
    write_case(tmp_path, "2", labels)
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config())
    tiles = NativeSlabDataset(
        [str(tmp_path)],
        dataset_config(random_hflip=False, scale_img=False),
    )
    sampler = CropSampler(len(dataset), torch.Generator().manual_seed(1))
    batches = list(DataLoader(dataset, batch_size=2, sampler=sampler))
    assert len(batches) == 3
    for batch in batches:
        assert set(batch) == set(tiles[0])
        assert batch["image"].shape == (2, 1, 4, 8, 8) and batch["image"].dtype == torch.float32
        assert batch["target"].shape == (2, 1, 4, 8, 8) and batch["target"].dtype == torch.int64
        assert batch["valid_mask"].dtype == torch.bool
        assert torch.equal(batch["valid_mask"], batch["target"] != -100)
        volume, target = unpack_training_batch(batch, "cpu")
        assert volume.shape == (2, 1, 4, 8, 8) and target.shape == (2, 4, 8, 8)


def test_unscaled_landmarks_map_model_to_source(tmp_path):
    labels = tumor_labels()
    image = linear_image(labels.shape)
    affine = np.array([[0, -2, 0.1, 40], [1.5, 0, 0.2, -30], [0.3, 0, 4, 6], [0, 0, 0, 1]])
    write_case(tmp_path, "1", labels, image, affine)
    dataset = ForegroundCropDataset(
        [str(tmp_path)], dataset_config(random_hflip=False, scale_img=False)
    )
    for seed in range(5):
        sample = dataset[(0, seed)]
        params = dataset.crop_parameters(0, seed)
        assert sample["start"].tolist() == list(params["start_dhw"])
        assert bool(sample["valid_mask"].all())
        mapping = sample["model_to_source"].numpy()
        for point in [(0, 0, 0), (3, 7, 7), (1, 2, 5), (2, 6, 1)]:
            source = mapping @ np.array([*point, 1])
            h, w, d = (int(v) for v in source[:3])
            np.testing.assert_array_equal(source[:3], [h, w, d])
            assert sample["image"][(0, *point)] == image[h, w, d]
            assert sample["target"][(0, *point)] == labels[h, w, d]
            np.testing.assert_allclose(
                sample["model_to_world"].numpy() @ np.array([*point, 1]), affine @ source
            )
        d0, h0, w0 = params["start_dhw"]
        np.testing.assert_array_equal(mapping @ np.array([0, 0, 0, 1]), [h0, w0, d0, 1])


@pytest.mark.parametrize("scale", [1.25, 0.75])
def test_scaled_flipped_crop_uses_half_pixel_mapping(tmp_path, scale):
    labels = tumor_labels()
    image = linear_image(labels.shape)
    write_case(tmp_path, "1", labels, image)
    cfg = dataset_config(scale_img_range={"min": scale, "max": scale}, random_hflip_probability=1.0)
    dataset = ForegroundCropDataset([str(tmp_path)], cfg)
    sample = dataset[(0, 4)]
    params = dataset.crop_parameters(0, 4)
    extent = round(8 * scale)
    assert params["flip"] and params["extent"] == (4, extent, extent)
    d0, h0, w0 = params["start_dhw"]
    mapping = sample["model_to_source"].numpy()
    q = extent / 8
    for md, mh, mw in np.ndindex(4, 8, 8):
        source = mapping @ np.array([md, mh, mw, 1])
        np.testing.assert_allclose(source[0], h0 + q * (mh + 0.5) - 0.5)
        np.testing.assert_allclose(source[1], w0 + q * (7 - mw + 0.5) - 0.5)
        assert source[2] == d0 + md
        nearest = np.floor(source[:3] + 0.5).astype(int)
        assert sample["target"][0, md, mh, mw] == labels[tuple(nearest)]
        inside = all(s <= v <= s + extent - 1 for v, s in zip(source[:2], (h0, w0), strict=True))
        if inside:
            expected = source[0] + 100 * source[1] + 10000 * source[2]
            np.testing.assert_allclose(sample["image"][0, md, mh, mw], expected, rtol=1e-5)


def test_padding_and_valid_mask_at_volume_edges(tmp_path):
    labels = np.ones((6, 5, 3), dtype=np.uint8)
    labels[0, 0, 0] = 2
    write_case(tmp_path, "1", labels, np.full(labels.shape, 30.0))
    for flip in (0.0, 1.0):
        cfg = dataset_config(scale_img=False, random_hflip_probability=flip)
        dataset = ForegroundCropDataset([str(tmp_path)], cfg)
        params = dataset.crop_parameters(0, 0)
        assert params["start_dhw"] == (0, 0, 0)
        sample = dataset[(0, 0)]
        valid = sample["valid_mask"]
        assert torch.equal(valid, sample["target"] != -100)
        assert int(valid.sum()) == 3 * 6 * 5
        assert sample["valid_extents"].tolist() == [3, 6, 5]
        assert torch.all(sample["image"][~valid] == -200)
        assert torch.all(sample["image"][valid] == 30)
        columns = valid[0, 0, 0].nonzero().flatten().tolist()
        assert columns == (list(range(3, 8)) if flip else list(range(5)))
    # Scaled crop larger than the volume is padded on the model grid, not clamped.
    dataset = ForegroundCropDataset(
        [str(tmp_path)], dataset_config(scale_img_range={"min": 1.2, "max": 1.2})
    )
    sample = dataset[(0, 0)]
    assert torch.equal(sample["valid_mask"], sample["target"] != -100)
    assert 0 < int(sample["valid_mask"].sum()) < 4 * 8 * 8


def test_manifest_records_geometry_hashes_and_index(tmp_path):
    volume, target = write_case(tmp_path, "1", tumor_labels())
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config())
    manifest = dataset.manifest()
    assert manifest["geometry"] == crop_config(dataset_config())
    assert manifest["passes_per_epoch"] == 3 and manifest["sampling_seed"] == 7
    [case] = manifest["cases"]
    assert case["case_id"] == "-1" and case["shape"] == (20, 18, 10)
    assert case["image_sha256"] == hashlib.sha256(Path(volume).read_bytes()).hexdigest()
    assert case["target_sha256"] == hashlib.sha256(Path(target).read_bytes()).hexdigest()
    assert manifest["foreground_index_sha256"] == dataset.foreground_index_sha256
    reseeded = ForegroundCropDataset([str(tmp_path)], dataset_config(sampling_seed=8))
    assert reseeded.foreground_index_sha256 != dataset.foreground_index_sha256


def test_subsampling_is_capped_deterministic_and_seeded(tmp_path):
    labels = tumor_labels()
    write_case(tmp_path, "1", labels)
    liver_count = int((labels >= 1).sum())
    small = dataset_config(foreground_voxel_cache=30)
    first = ForegroundCropDataset([str(tmp_path)], small)
    second = ForegroundCropDataset([str(tmp_path)], small)
    liver = first.foreground_index["liver_0"]
    assert liver.dtype == np.int32 and liver.shape == (30, 3) and liver_count > 30
    assert np.all(labels[tuple(liver.T)] >= 1)
    assert len({tuple(v) for v in liver.tolist()}) == 30
    np.testing.assert_array_equal(second.foreground_index["liver_0"], liver)
    reseeded = ForegroundCropDataset([str(tmp_path)], {**small, "sampling_seed": 8})
    assert not np.array_equal(reseeded.foreground_index["liver_0"], liver)
    assert first.foreground_index["tumor_0"].shape == (8, 3)
    np.testing.assert_array_equal(first.foreground_index["liver_bbox_0"], [[4, 3, 2], [15, 14, 8]])
    np.testing.assert_array_equal(first.foreground_index["tumor_bbox_0"], [[6, 5, 4], [7, 6, 5]])


def test_cache_reused_and_rebuilt_when_source_hash_changes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    labels = tumor_labels()
    write_case(data, "1", labels)
    cache = tmp_path / "foreground-index.npz"
    built = ForegroundCropDataset([str(data)], dataset_config(), cache_path=cache)
    assert cache.exists()

    def refuse(self):
        raise AssertionError("index rebuilt despite matching hashes")

    monkeypatch.setattr(ForegroundCropDataset, "_build_index", refuse)
    reused = ForegroundCropDataset([str(data)], dataset_config(), cache_path=cache)
    assert reused.foreground_index_sha256 == built.foreground_index_sha256
    monkeypatch.undo()
    labels[labels == 2] = 1
    write_case(data, "1", labels)
    rebuilt = ForegroundCropDataset([str(data)], dataset_config(), cache_path=cache)
    assert rebuilt.foreground_index["tumor_0"].shape == (0, 3)
    assert rebuilt.foreground_index_sha256 != built.foreground_index_sha256
    reloaded = ForegroundCropDataset([str(data)], dataset_config(), cache_path=cache)
    assert reloaded.foreground_index_sha256 == rebuilt.foreground_index_sha256


def test_restored_generator_replays_crops_regardless_of_workers(tmp_path):
    write_case(tmp_path, "1", tumor_labels())
    write_case(tmp_path, "2", tumor_labels())
    dataset = ForegroundCropDataset([str(tmp_path)], dataset_config())
    generator = torch.Generator().manual_seed(5)
    state = generator.get_state()

    def epoch(workers):
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=CropSampler(len(dataset), generator),
            num_workers=workers,
        )
        return [(b["image"], b["target"]) for b in loader]

    parallel = epoch(2)
    generator.set_state(state)
    serial = epoch(0)
    assert len(parallel) == len(serial) == 3
    for (a_image, a_target), (b_image, b_target) in zip(parallel, serial, strict=True):
        assert torch.equal(a_image, b_image) and torch.equal(a_target, b_target)
    assert not torch.equal(epoch(0)[0][0], serial[0][0])
