import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.main as api_main
from api.reliability import CircuitBreaker


def test_confidence_threshold_maps_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_SERVER_ENABLED", "0")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.7")
    monkeypatch.setattr(
        api_main,
        "classify_log",
        lambda _log: {
            "label": "Data Integrity Error",
            "explanation": "raw",
            "confidence": 0.42,
        },
    )

    client = TestClient(api_main.app)
    resp = client.post("/classify", json={"log": "x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "Unknown"
    assert "Low-confidence classification" in body["explanation"]
    assert body["confidence"] == 0.42


def test_circuit_breaker_opens_and_fallback_is_used(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_SERVER_ENABLED", "1")
    monkeypatch.setenv("MODEL_SERVER_FALLBACK", "1")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.0")

    api_main._breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=3600)

    calls = {"server": 0, "fallback": 0}

    def fail_server(_log: str):
        calls["server"] += 1
        raise RuntimeError("server down")

    def fallback(_log: str):
        calls["fallback"] += 1
        return {
            "label": "Handshake Protocol Violation",
            "explanation": "fallback",
            "confidence": 0.8,
        }

    monkeypatch.setattr(api_main, "classify_via_server", fail_server)
    monkeypatch.setattr(api_main, "classify_log", fallback)

    client = TestClient(api_main.app)

    first = client.post("/classify", json={"log": "x"})
    assert first.status_code == 200
    assert first.json()["label"] == "Handshake Protocol Violation"

    second = client.post("/classify", json={"log": "x"})
    assert second.status_code == 200
    assert second.json()["label"] == "Handshake Protocol Violation"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["circuit_state"] == "OPEN"
    assert health.json()["circuit_failures"] >= 1

    assert calls["server"] == 1
    assert calls["fallback"] == 2


# Avoid leaking test env mutations if tests are run without isolation.
os.environ.pop("MODEL_SERVER_ENABLED", None)
os.environ.pop("MODEL_SERVER_FALLBACK", None)
os.environ.pop("CONFIDENCE_THRESHOLD", None)
