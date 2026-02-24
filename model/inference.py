"""model/inference.py — Rule-based fallback classifier.

Used when:
  - MODEL_SERVER_ENABLED=0 (local rule-based only)
  - Model server is OPEN in circuit-breaker state (graceful degradation)

Implements a priority-ordered rule chain covering all 10 failure classes
across FIFO, UART, SPI, GPIO, and I2C modules. Rules match structured
TB_LOG fields first (high confidence) then fall through to unstructured
regex patterns (lower confidence).

This is NOT the fine-tuned model. It is the safety net.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
_Rule = Tuple[re.Pattern, str, str, float]


# ---------------------------------------------------------------------------
# Rule definitions — ordered by specificity (most specific first)
# ---------------------------------------------------------------------------
# Each entry: (pattern, label, explanation, confidence)

_RULES: List[_Rule] = [

    # ── Tier 2: Overflow Guard Removal ─────────────────────────────────────
    (
        re.compile(
            r"UART.*BAUD.*overflow|baud_shadow\s*=\s*\d+|UART.*BAUD_OVERFLOW"
            r"|Baud counter overflow|timer.*overrun|watchdog.*expired",
            re.IGNORECASE,
        ),
        "Overflow Guard Removal",
        "Baud/bit counter exceeded its limit by one tick due to removal of the "
        "'-1' guard in the counter comparison. The counter wrapped or ran one "
        "extra cycle, violating protocol timing.",
        0.87,
    ),
    (
        re.compile(
            r"SPI.*bit.*counter.*overflow|SPI.*transfer.*watchdog|SPI_BIT_COUNTER_OVERFLOW"
            r"|I2C.*bit.*overflow|I2C_BIT_OVERFLOW",
            re.IGNORECASE,
        ),
        "Overflow Guard Removal",
        "SPI/I2C bit counter exceeded the expected word length, "
        "likely due to removal of the boundary guard in the clock divider "
        "or bit counter decrement condition.",
        0.85,
    ),

    # ── Tier 2: Parity Check Removal ───────────────────────────────────────
    (
        re.compile(
            r"parity.*error|UART_PARITY_ERROR|parity.*mismatch"
            r"|frame.*integrity.*failed|parity_err\s*=\s*1",
            re.IGNORECASE,
        ),
        "Parity Check Removal",
        "UART parity register was zeroed or parity check was disabled. "
        "The transmitter sends frames with incorrect parity, causing the "
        "receiver to flag a frame integrity error.",
        0.88,
    ),

    # ── Tier 1: Enable Signal Polarity Flip ────────────────────────────────
    (
        re.compile(
            r"ENABLE_POLARITY|enable.*polarity|write.*while.*wr_en.*=.*0"
            r"|operation.*disabled.*enable|wr_en.*0.*write|GPIO_IRQ_STUCK"
            r"|IRQ.*stuck.*high.*>.*1.*cycle",
            re.IGNORECASE,
        ),
        "Enable Signal Polarity Flip",
        "A write-enable or read-enable signal was active-low flipped. "
        "Operations now occur when the enable is de-asserted and are "
        "suppressed when it is asserted, corrupting data output.",
        0.85,
    ),
    (
        re.compile(
            r"GPIO.*output.*enable.*mask|GPIO_OUTPUT_MASK"
            r"|output.*pin.*driven.*input.*direction",
            re.IGNORECASE,
        ),
        "Enable Signal Polarity Flip",
        "GPIO output enable mask was removed. Output pins are being driven "
        "for pins configured as inputs, violating the direction register.",
        0.83,
    ),

    # ── Tier 1: Data Width Truncation ──────────────────────────────────────
    (
        re.compile(
            r"SB_WIDTH_MISMATCH|MSB.*truncated|shift.*register.*MSB.*lost"
            r"|width.*mismatch",
            re.IGNORECASE,
        ),
        "Data Width Truncation",
        "Shift register MSB was silently truncated due to an off-by-one in "
        "the concatenation slice. The received/output data is missing its "
        "most significant bits.",
        0.86,
    ),

    # ── Tier 1: Assignment Semantics Change ────────────────────────────────
    (
        re.compile(
            r"SB_DATA_MISMATCH|Data mismatch.*expected.*actual"
            r"|Unexpected data.*actual.*cycle"
            r"|SCOREBOARD_FAIL.*Data mismatch"
            r"|data.*path.*corruption",
            re.IGNORECASE,
        ),
        "Data Integrity Error",
        "Scoreboard detected a mismatch between expected and actual data. "
        "Likely caused by a nonblocking-to-blocking assignment change that "
        "corrupts the data path's pipeline registers.",
        0.85,
    ),

    # ── Tier 2: Handshake Protocol Violation ───────────────────────────────
    (
        re.compile(
            r"Backpressure violation|write_fire\s*=\s*1\s+ready\s*=\s*0"
            r"|FIFO_BACKPRESSURE_WRITE|protocol assertion tripped"
            r"|valid\s*=\s*1.*ready\s*=\s*0"
            r"|SPI_CS_DEASSERTED|CS_N deasserted during"
            r"|UART_READY_STUCK|ready.*stuck.*high.*handshake"
            r"|I2C_NACK_ERROR|NACK.*received.*ACK.*window",
            re.IGNORECASE,
        ),
        "Handshake Protocol Violation",
        "A write/transmit operation occurred while the receiver was not ready "
        "(ready=0 / CS_N deasserted / NACK). The ready-gating condition was "
        "likely removed by mutation.",
        0.88,
    ),

    # ── Tier 2: Counter Boundary Violation (Off-by-One) ────────────────────
    (
        re.compile(
            r"Counter overflow|FIFO_COUNTER_OVERFLOW|count\w*\s*=\s*\d+.*MAX"
            r"|Counter exceeded MAX|boundary.*check.*failed"
            r"|UART_BAUD_OVERFLOW|bit.*exceeded.*upper.*bound",
            re.IGNORECASE,
        ),
        "Off-by-One Error",
        "Counter exceeded its maximum depth (MAX_DEPTH) by one. The "
        "boundary check was likely relaxed from `< MAX` to `<= MAX`, "
        "allowing the counter to reach an invalid state.",
        0.87,
    ),

    # ── Tier 1: Edge Sensitivity Flip ──────────────────────────────────────
    (
        re.compile(
            r"Clock edge violation|negedge.*sensitivity|CLOCK_EDGE_FAIL"
            r"|flip-flop.*sampled.*wrong.*clock.*edge"
            r"|clock.*negedge.*unexpected",
            re.IGNORECASE,
        ),
        "Edge Sensitivity Flip",
        "Clock edge sensitivity was flipped from posedge to negedge. "
        "Sequential logic now samples on the wrong edge, causing "
        "timing violations and spurious counter increments.",
        0.84,
    ),

    # ── Tier 1: Reset Polarity Inversion ───────────────────────────────────
    (
        re.compile(
            r"Reset polarity|RESET_POLARITY|rst_n.*=.*1.*reset.*asserted"
            r"|DUT.*active.*while.*reset|I2C_SDA_IDLE"
            r"|UART_IDLE_HIGH|UART.*not.*idle.*high|UART.*TX.*not.*idle"
            r"|SPI_SCLK_ACTIVE_WHEN_IDLE|SCLK.*active.*while.*CS_N.*high",
            re.IGNORECASE,
        ),
        "Reset Polarity Inversion",
        "Active-low reset check was inverted from `!rst_n` to `rst_n`. "
        "The DUT is now in reset when rst_n=1 (operating) and active "
        "when rst_n=0 (reset), causing output corruption at power-on. "
        "UART idle-high and SPI SCLK violations at startup are typical symptoms.",
        0.84,
    ),

    # ── Generic SCOREBOARD_FAIL catch-all (lower confidence) ───────────────
    (
        re.compile(r"SCOREBOARD_FAIL|scoreboard.*fail", re.IGNORECASE),
        "Data Integrity Error",
        "Testbench scoreboard detected a data mismatch. "
        "Review data path, shift register, or pointer logic for corruption.",
        0.72,
    ),

    # ── Generic ASSERT_FAIL + counter tokens ───────────────────────────────
    (
        re.compile(
            r"ASSERT_FAIL.*(?:count|overflow|boundary|DEPTH|MAX)",
            re.IGNORECASE,
        ),
        "Off-by-One Error",
        "Assertion failure involving a counter or boundary check. "
        "Likely an off-by-one introduced by a `<` → `<=` mutation.",
        0.75,
    ),

    # ── Generic ASSERT_FAIL + handshake tokens ─────────────────────────────
    (
        re.compile(
            r"ASSERT_FAIL.*(?:ready|valid|handshake|backpressure|nack|cs_n)",
            re.IGNORECASE,
        ),
        "Handshake Protocol Violation",
        "Assertion failure involving a handshake or ready/valid signal. "
        "Likely a ready-gating condition was removed by mutation.",
        0.75,
    ),
]


def classify_log(log: str) -> Dict[str, str | float]:
    """Classify a simulation log using the rule-based fallback engine.

    Returns a dict with keys: label, explanation, confidence.
    """
    for pattern, label, explanation, confidence in _RULES:
        if pattern.search(log):
            return {
                "label":       label,
                "explanation": explanation,
                "confidence":  confidence,
            }

    return {
        "label":       "Unknown",
        "explanation": (
            "No known pattern matched. The log does not contain recognisable "
            "failure signatures. Review the simulation output manually."
        ),
        "confidence":  0.3,
    }
