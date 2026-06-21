"""Tests for D3 — seeded 80/20 train/val split."""

from dense_unet_3d.dataset.prepare_dataset import make_split


def _ids(n: int = 131) -> list[str]:
    """Synthetic list of volume IDs mirroring 131 LiTS volumes."""
    return [f"volume-{i:03d}" for i in range(n)]


class TestMakeSplitDeterminism:
    def test_same_seed_produces_identical_train_lists(self) -> None:
        ids = _ids()
        train_a, _ = make_split(ids, val_fraction=0.2, seed=42)
        train_b, _ = make_split(ids, val_fraction=0.2, seed=42)
        assert train_a == train_b

    def test_same_seed_produces_identical_val_lists(self) -> None:
        ids = _ids()
        _, val_a = make_split(ids, val_fraction=0.2, seed=42)
        _, val_b = make_split(ids, val_fraction=0.2, seed=42)
        assert val_a == val_b

    def test_different_seeds_produce_different_splits(self) -> None:
        ids = _ids()
        train_a, _ = make_split(ids, val_fraction=0.2, seed=42)
        train_b, _ = make_split(ids, val_fraction=0.2, seed=99)
        assert train_a != train_b


class TestMakeSplitPartition:
    def test_union_equals_all_volumes(self) -> None:
        ids = _ids()
        train, val = make_split(ids, val_fraction=0.2, seed=42)
        assert sorted(train + val) == sorted(ids)

    def test_train_and_val_are_disjoint(self) -> None:
        ids = _ids()
        train, val = make_split(ids, val_fraction=0.2, seed=42)
        assert set(train).isdisjoint(set(val))

    def test_no_volume_in_both_splits(self) -> None:
        ids = _ids()
        train, val = make_split(ids, val_fraction=0.2, seed=42)
        overlap = set(train) & set(val)
        assert len(overlap) == 0

    def test_approximate_80_20_ratio_131_volumes(self) -> None:
        ids = _ids(131)
        train, val = make_split(ids, val_fraction=0.2, seed=42)
        # 80/20 of 131 → ~105 train / ~26 val; allow ±1 rounding
        assert 104 <= len(train) <= 106
        assert 25 <= len(val) <= 27
        assert len(train) + len(val) == 131

    def test_val_fraction_respected(self) -> None:
        ids = _ids(100)
        train, val = make_split(ids, val_fraction=0.2, seed=0)
        assert len(val) == 20
        assert len(train) == 80
