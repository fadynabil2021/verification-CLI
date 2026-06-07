"""model/prompt.py — Shared prompt template formatting logic.

Consolidates the prompt template used during both training and inference (server)
to prevent train/serve skew.
"""
from __future__ import annotations


def format_prompt(log: str) -> str:
    """Formats a raw simulation log into the prompt expected by the model."""
    return (
        "You are a verification log classifier. "
        "Return JSON with keys: label, explanation, confidence (0-1). "
        "Log:\n"
        f"{log}\n"
        "JSON:"
    )
