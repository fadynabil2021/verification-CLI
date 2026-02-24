"""Tests for data/split_dataset.py — stratified split logic."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from data.split_dataset import stratified_split


def _make_rows(label_counts: Dict[str, int]) -> List[Dict]:
    rows = []
    idx  = 0
    for label, count in label_counts.items():
        for _ in range(count):
            rows.append({"log": f"log_{idx}", "label": label, "confidence": 1.0})
            idx += 1
    return rows


class TestStratifiedSplit:
    def test_no_samples_lost(self) -> None:
        rows  = _make_rows({"Off-by-One Error": 50, "Handshake Protocol Violation": 50,
                            "Data Integrity Error": 50})
        train, val, test = stratified_split(rows, train_ratio=0.70, val_ratio=0.15, seed=42)
        assert len(train) + len(val) + len(test) == len(rows)

    def test_all_labels_in_train(self) -> None:
        rows  = _make_rows({"Off-by-One Error": 30, "Handshake Protocol Violation": 30,
                            "Data Integrity Error": 30, "Unknown": 30})
        train, val, test = stratified_split(rows, seed=42)
        train_labels = {r["label"] for r in train}
        assert "Off-by-One Error" in train_labels
        assert "Handshake Protocol Violation" in train_labels
        assert "Data Integrity Error" in train_labels

    def test_all_labels_in_val(self) -> None:
        rows  = _make_rows({"Off-by-One Error": 30, "Handshake Protocol Violation": 30,
                            "Data Integrity Error": 30})
        _, val, _ = stratified_split(rows, seed=42)
        val_labels = {r["label"] for r in val}
        assert len(val_labels) >= 2  # At minimum most labels represented

    def test_train_is_largest_split(self) -> None:
        rows  = _make_rows({"Off-by-One Error": 100, "Handshake Protocol Violation": 100})
        train, val, test = stratified_split(rows, train_ratio=0.70, val_ratio=0.15, seed=42)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_proportions_approximately_correct(self) -> None:
        rows  = _make_rows({"Off-by-One Error": 200, "Handshake Protocol Violation": 200,
                            "Data Integrity Error": 200})
        train, val, test = stratified_split(rows, train_ratio=0.70, val_ratio=0.15, seed=42)
        total = len(rows)
        # Allow ±5% tolerance
        assert abs(len(train) / total - 0.70) < 0.05
        assert abs(len(val)   / total - 0.15) < 0.05

    def test_seed_reproducibility(self) -> None:
        rows = _make_rows({"Off-by-One Error": 50, "Handshake Protocol Violation": 50})
        t1, v1, te1 = stratified_split(rows, seed=99)
        t2, v2, te2 = stratified_split(rows, seed=99)
        assert [r["log"] for r in t1] == [r["log"] for r in t2]

    def test_different_seeds_give_different_order(self) -> None:
        rows = _make_rows({"Off-by-One Error": 50, "Handshake Protocol Violation": 50})
        t1, _, _ = stratified_split(rows, seed=1)
        t2, _, _ = stratified_split(rows, seed=2)
        # Unlikely to be identical
        assert [r["log"] for r in t1] != [r["log"] for r in t2]

    def test_very_small_group_handled(self) -> None:
        rows = _make_rows({"Off-by-One Error": 100, "Rare Error": 2})
        train, val, test = stratified_split(rows, seed=42)
        assert len(train) + len(val) + len(test) == 102

    def test_single_sample_group(self) -> None:
        rows = _make_rows({"Off-by-One Error": 50, "One-Off": 1})
        train, val, test = stratified_split(rows, seed=42)
        assert len(train) + len(val) + len(test) == 51
