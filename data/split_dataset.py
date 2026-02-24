"""data/split_dataset.py — Train / Validation / Test split utility.

Splits a JSONL dataset (typically the augmented dataset) into three
stratified splits, ensuring each failure-class label is represented in
all three splits proportionally.

Output files (default):
    data/splits/train.jsonl       — 70% (or ~400 samples for training)
    data/splits/val.jsonl         — 15% (held-out for early stopping)
    data/splits/test.jsonl        — 15% (held-out for final eval, never seen during training)

Stratification:
    - Groups samples by `label` field.
    - Applies the split ratio to each group independently.
    - Randomly shuffles within each group before splitting (seeded).

Usage:
    python -m data.split_dataset \
        --input  data/dataset_augmented.jsonl \
        --out-dir data/splits \
        --train  0.70 \
        --val    0.15 \
        --seed   42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(
    rows: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split — each label gets proportional train/val/test rows."""
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-9
    test_ratio = 1.0 - train_ratio - val_ratio

    rng = random.Random(seed)

    # Group by label
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        label = row.get("label", "Unknown")
        by_label[label].append(row)

    train_rows: List[Dict] = []
    val_rows:   List[Dict] = []
    test_rows:  List[Dict] = []

    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        # Remaining go to test — ensures no sample is lost
        n_test  = n - n_train - n_val
        if n_test < 0:
            # Very small group: give at least 1 to train
            n_train = max(1, n - 2)
            n_val   = 1 if n > 1 else 0
            n_test  = max(0, n - n_train - n_val)

        train_rows.extend(group[:n_train])
        val_rows.extend(  group[n_train : n_train + n_val])
        test_rows.extend( group[n_train + n_val :])

    # Shuffle final splits so labels aren't grouped
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    return train_rows, val_rows, test_rows


def print_split_stats(
    rows: List[Dict],
    label: str,
) -> None:
    from collections import Counter
    labels = Counter(r.get("label", "Unknown") for r in rows)
    print(f" {label}: {len(rows)} samples")
    for lbl, cnt in sorted(labels.items()):
        print(f"   {lbl}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for JSONL dataset"
    )
    parser.add_argument("--input",   default="data/dataset_augmented.jsonl",
                        help="Input JSONL (augmented dataset)")
    parser.add_argument("--out-dir", default="data/splits",
                        help="Output directory for split files")
    parser.add_argument("--train",   type=float, default=0.70,
                        help="Training set fraction (default: 0.70)")
    parser.add_argument("--val",     type=float, default=0.15,
                        help="Validation set fraction (default: 0.15)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.out_dir)

    rows = load_jsonl(in_path)
    print(f"Loaded {len(rows)} samples from {in_path}")

    train, val, test = stratified_split(
        rows,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
    )

    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val,   out_dir / "val.jsonl")
    write_jsonl(test,  out_dir / "test.jsonl")

    print("\nSplit complete:")
    print_split_stats(train, "Train")
    print_split_stats(val,   "Val  ")
    print_split_stats(test,  "Test ")
    print(f"\nFiles written to {out_dir}/")


if __name__ == "__main__":
    main()
