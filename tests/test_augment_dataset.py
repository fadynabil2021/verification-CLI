"""Tests for data/augment_dataset.py — augmentation strategies."""
from __future__ import annotations

from typing import Dict, List

from data.augment_dataset import augment_dataset, augment_sample
import random


def _base_rows() -> List[Dict]:
    return [
        {
            "log":         "ASSERT_FAIL: Counter overflow at cycle 23\nSimulation terminated.",
            "label":       "Off-by-One Error",
            "explanation": "Counter exceeded MAX_DEPTH.",
            "confidence":  1.0,
            "mutation_id": "counter_boundary_violation",
            "module":      "fifo",
        },
        {
            "log":         "ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle 31",
            "label":       "Handshake Protocol Violation",
            "explanation": "Ready gating removed.",
            "confidence":  1.0,
            "mutation_id": "handshake_violation",
            "module":      "fifo",
        },
        {
            "log":         "SCOREBOARD_FAIL: Data mismatch expected=12 actual=9 at cycle 150",
            "label":       "Data Integrity Error",
            "explanation": "Data path corruption.",
            "confidence":  1.0,
            "mutation_id": "nonblocking_to_blocking",
            "module":      "fifo",
        },
    ]


class TestAugmentDataset:
    def test_returns_base_plus_augmented(self) -> None:
        rows = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        # 3 base + (3 × 4) augmented = 15
        assert len(result) == 15

    def test_base_rows_preserved_exactly(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        # First 3 entries should be originals (one per base sample)
        original_ids = {r["mutation_id"] for r in rows}
        result_originals = [r for r in result if not r.get("augmented")]
        assert len(result_originals) == len(rows)
        for orig in result_originals:
            assert orig["mutation_id"] in original_ids

    def test_augmented_rows_have_aug_flag(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        augmented = [r for r in result if r.get("augmented")]
        assert len(augmented) == 12
        for r in augmented:
            assert r["augmented"] is True
            assert "aug_id" in r

    def test_augmented_logs_are_non_empty(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        for r in result:
            assert r["log"].strip() != ""

    def test_labels_preserved_in_augmentation(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        for r in result:
            assert r["label"] in {
                "Off-by-One Error", "Handshake Protocol Violation", "Data Integrity Error"
            }

    def test_augmented_logs_differ_from_base(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        base_logs = {r["log"] for r in rows}
        augmented = [r for r in result if r.get("augmented")]
        # At least some augmented logs should differ from originals
        different = sum(1 for r in augmented if r["log"] not in base_logs)
        assert different > 0

    def test_seed_reproducibility(self) -> None:
        rows = _base_rows()
        r1   = augment_dataset(rows, n_aug=4, seed=99)
        r2   = augment_dataset(rows, n_aug=4, seed=99)
        logs1 = [r["log"] for r in r1]
        logs2 = [r["log"] for r in r2]
        assert logs1 == logs2

    def test_different_seeds_differ(self) -> None:
        rows = _base_rows()
        r1   = augment_dataset(rows, n_aug=4, seed=1)
        r2   = augment_dataset(rows, n_aug=4, seed=2)
        logs1 = [r["log"] for r in r1 if r.get("augmented")]
        logs2 = [r["log"] for r in r2 if r.get("augmented")]
        assert logs1 != logs2

    def test_n_aug_zero_returns_only_base(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=0, seed=42)
        assert len(result) == len(rows)

    def test_single_sample_augment(self) -> None:
        rows   = [_base_rows()[0]]
        result = augment_dataset(rows, n_aug=8, seed=42)
        assert len(result) == 9  # 1 base + 8 augmented

    def test_module_field_set_in_augmented(self) -> None:
        rows   = _base_rows()
        result = augment_dataset(rows, n_aug=4, seed=42)
        augmented = [r for r in result if r.get("augmented")]
        valid_modules = {"fifo", "uart_tx", "spi_master", "gpio", "i2c_master"}
        for r in augmented:
            assert r["module"] in valid_modules
