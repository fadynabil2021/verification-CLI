from metrics.dashboard import render_dashboard


def test_render_dashboard_contains_sections() -> None:
    text = render_dashboard(
        {"accuracy": 0.4, "correct": 2, "total": 5, "unknown_rate": 0.0},
        {"accuracy": 1.0, "correct": 5, "total": 5, "unknown_rate": 0.0},
        {
            "samples": 10,
            "manual_p50_min": 38.0,
            "ai_p50_min": 17.5,
            "manual_p95_min": 50.0,
            "ai_p95_min": 25.0,
            "time_reduction_p50": 0.539,
        },
    )
    assert "# VeriLog AI Metrics Dashboard" in text
    assert "## Model Eval" in text
    assert "## Natural Bug Sanity Check" in text
    assert "## A/B Pilot" in text
    assert "53.9%" in text
