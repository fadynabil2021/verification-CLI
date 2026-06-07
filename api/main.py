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
    # 1. Validation: Reject excessively large logs to prevent resource exhaustion
    if len(req.log) > 100000:
        raise HTTPException(
            status_code=400,
            detail="Log length exceeds maximum limit of 100,000 characters."
        )

    # 2. Safety Truncation: Keep only the end of the log where simulator errors occur
    log_content = req.log
    if len(log_content) > 50000:
        log_content = "... [TRUNCATED] ...\n" + log_content[-50000:]

    use_server = os.getenv("MODEL_SERVER_ENABLED", "1") == "1"
    if use_server:
        try:
            result = _breaker.call(classify_via_server, log_content)
        except ServiceUnavailableError as exc:
            if os.getenv("MODEL_SERVER_FALLBACK", "1") == "1":
                result = classify_log(log_content)
            else:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            if os.getenv("MODEL_SERVER_FALLBACK", "1") == "1":
                result = classify_log(log_content)
            else:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        result = classify_log(log_content)
    return ClassifyResponse(**_apply_confidence_gate(result))
