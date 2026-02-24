from model.natural_eval import evaluate


def test_natural_eval_basic() -> None:
    rows = [
        {
            "id": "n1",
            "label": "Off-by-One Error",
            "log": "ASSERT_FAIL: Counter overflow count=17 at cycle 10",
        },
        {
            "id": "n2",
            "label": "Handshake Protocol Violation",
            "log": "ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle 11",
        },
    ]

    out = evaluate(rows)
    assert out["total"] == 2
    assert out["correct"] == 2
    assert out["accuracy"] == 1.0
