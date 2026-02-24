from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from model.inference import classify_log


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(rows: Iterable[Dict]) -> Dict[str, object]:
    total = 0
    correct = 0
    unknown = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        log = row.get("log", "")
        true_label = row.get("label", "Unknown")
        pred = classify_log(log)
        pred_label = str(pred.get("label", "Unknown"))

        total += 1
        if pred_label == true_label:
            correct += 1
        if pred_label == "Unknown":
            unknown += 1
        confusion[true_label][pred_label] += 1

    accuracy = (correct / total) if total else 0.0
    unknown_rate = (unknown / total) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "confusion": {
            label: dict(counter.most_common()) for label, counter in confusion.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier over JSONL dataset")
    parser.add_argument(
        "--dataset",
        default="data/dataset.jsonl",
        help="Path to JSONL dataset with log + label fields",
    )
    args = parser.parse_args()

    dataset = load_jsonl(Path(args.dataset))
    metrics = evaluate(dataset)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
