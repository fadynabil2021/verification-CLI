from metrics.ab_test import evaluate


def test_ab_evaluate_stats() -> None:
    samples = [
        {"manual_minutes": 40, "ai_minutes": 20},
        {"manual_minutes": 30, "ai_minutes": 15},
        {"manual_minutes": 50, "ai_minutes": 25},
    ]
    out = evaluate(samples)
    assert out["samples"] == 3
    assert out["manual_p50_min"] == 40
    assert out["ai_p50_min"] == 20
    assert out["manual_p95_min"] == 50
    assert out["ai_p95_min"] == 25
    assert out["time_reduction_p50"] == 0.5
