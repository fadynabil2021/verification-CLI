from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def evaluate(rows: List[Dict]) -> Dict:
    total = 0
    correct = 0
    unknown = 0
    details: List[Dict] = []

    for row in rows:
        total += 1
        expected = row.get("label", "Unknown")
        pred = classify_log(row.get("log", ""))
        predicted = pred.get("label", "Unknown")
        if predicted == expected:
            correct += 1
        if predicted == "Unknown":
            unknown += 1
        details.append(
            {
                "id": row.get("id", "unknown"),
                "expected": expected,
                "predicted": predicted,
                "confidence": pred.get("confidence", 0.0),
            }
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "unknown_rate": (unknown / total) if total else 0.0,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate natural bug sanity-check logs")
    parser.add_argument("--dataset", default="data/natural_bugs.jsonl")
    parser.add_argument("--out", default="reports/natural_bug_report.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    report = evaluate(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote natural bug report to {out}")


if __name__ == "__main__":
    main()
