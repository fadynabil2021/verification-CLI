import os

from fastapi import FastAPI, HTTPException

from .schemas import ClassifyRequest, ClassifyResponse
from .reliability import CircuitBreaker, ServiceUnavailableError
from model.inference import classify_log
from model.client import classify_via_server

app = FastAPI(title="VeriLog AI")
_breaker = CircuitBreaker(
    failure_threshold=int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5")),
    recovery_timeout=int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT_SEC", "60")),
)


def _apply_confidence_gate(result: dict) -> dict:
    threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    confidence = float(result.get("confidence", 0.0))
    if confidence >= threshold:
        return result
    return {
        "label": "Unknown",
        "explanation": (
            f"Low-confidence classification ({confidence:.2f} < {threshold:.2f}). "
            "Check manually."
        ),
        "confidence": confidence,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "circuit_state": _breaker.state,
        "circuit_failures": _breaker.failure_count,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    use_server = os.getenv("MODEL_SERVER_ENABLED", "1") == "1"
    if use_server:
        try:
            result = _breaker.call(classify_via_server, req.log)
        except ServiceUnavailableError as exc:
            if os.getenv("MODEL_SERVER_FALLBACK", "1") == "1":
                result = classify_log(req.log)
            else:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            if os.getenv("MODEL_SERVER_FALLBACK", "1") == "1":
                result = classify_log(req.log)
            else:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        result = classify_log(req.log)
    return ClassifyResponse(**_apply_confidence_gate(result))
