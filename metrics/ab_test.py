from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, List


def load_json(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def evaluate(samples: List[Dict]) -> Dict:
    manual = [float(s["manual_minutes"]) for s in samples]
    ai = [float(s["ai_minutes"]) for s in samples]

    manual_p50 = median(manual) if manual else 0.0
    ai_p50 = median(ai) if ai else 0.0
    manual_p95 = _p95(manual)
    ai_p95 = _p95(ai)

    p50_reduction = ((manual_p50 - ai_p50) / manual_p50) if manual_p50 else 0.0

    return {
        "samples": len(samples),
        "manual_p50_min": manual_p50,
        "ai_p50_min": ai_p50,
        "manual_p95_min": manual_p95,
        "ai_p95_min": ai_p95,
        "time_reduction_p50": p50_reduction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute A/B summary statistics")
    parser.add_argument("--input", default="reports/ab_test_samples.json")
    parser.add_argument("--out", default="reports/ab_test_report.json")
    args = parser.parse_args()

    report = evaluate(load_json(Path(args.input)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote A/B report to {out}")


if __name__ == "__main__":
    main()
