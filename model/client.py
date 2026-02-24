from __future__ import annotations

import os
from typing import Dict

import httpx

DEFAULT_URL = "http://localhost:8001"  # local model server


def _url() -> str:
    return os.getenv("MODEL_SERVER_URL", DEFAULT_URL).rstrip("/")


def classify_via_server(log: str) -> Dict[str, str | float]:
    url = f"{_url()}/classify"
    resp = httpx.post(url, json={"log": log}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()
