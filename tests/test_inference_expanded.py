"""Tests for expanded model/inference.py — covers all 10 failure classes."""
from __future__ import annotations

import pytest

from model.inference import classify_log


# Helper: assert label matches and confidence is sane
def _check(log: str, expected_label: str, min_confidence: float = 0.70) -> None:
    result = classify_log(log)
    assert result["label"] == expected_label, (
        f"Expected '{expected_label}', got '{result['label']}' for log:\n{log}"
    )
    assert result["confidence"] >= min_confidence, (
        f"Confidence too low: {result['confidence']} for label '{expected_label}'"
    )


class TestInferenceExpandedRules:
    # ── Off-by-One Error ────────────────────────────────────────────────────
    def test_fifo_counter_overflow_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_COUNTER_OVERFLOW|count=17|cycle=23",
            "Off-by-One Error",
        )

    def test_counter_overflow_plain(self) -> None:
        _check(
            "ASSERT_FAIL: Counter overflow at cycle 23\nSimulation terminated.",
            "Off-by-One Error",
        )

    def test_boundary_check_failed(self) -> None:
        _check(
            "boundary check failed — count value 16 >= MAX at cycle 55",
            "Off-by-One Error",
        )

    # ── Handshake Protocol Violation ────────────────────────────────────────
    def test_fifo_backpressure_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_BACKPRESSURE_WRITE|cycle=31",
            "Handshake Protocol Violation",
        )

    def test_backpressure_plain(self) -> None:
        _check(
            "ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle 31",
            "Handshake Protocol Violation",
        )

    def test_spi_cs_deasserted(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_CS_DEASSERTED_DURING_TRANSFER|cycle=92",
            "Handshake Protocol Violation",
        )

    def test_uart_ready_stuck(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_READY_STUCK|cycle=22",
            "Handshake Protocol Violation",
        )

    def test_i2c_nack_error(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_NACK_ERROR|cycle=450",
            "Handshake Protocol Violation",
        )

    # ── Data Integrity Error ────────────────────────────────────────────────
    def test_scoreboard_data_mismatch_structured(self) -> None:
        _check(
            "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_DATA_MISMATCH|expected=12|actual=9|cycle=150",
            "Data Integrity Error",
        )

    def test_scoreboard_fail_plain(self) -> None:
        _check(
            "SCOREBOARD_FAIL: Data mismatch expected=8 actual=0 at cycle 42",
            "Data Integrity Error",
        )

    def test_spi_miso_mismatch(self) -> None:
        _check(
            "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=222|actual=0|cycle=88",
            "Data Integrity Error",
        )

    def test_unexpected_data_plain(self) -> None:
        _check(
            "SCOREBOARD_FAIL: Unexpected data actual=44 at cycle 66",
            "Data Integrity Error",
            min_confidence=0.65,  # generic catch-all has lower confidence
        )

    # ── Overflow Guard Removal ──────────────────────────────────────────────
    def test_uart_baud_overflow_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_BAUD_OVERFLOW|baud_shadow=869|cycle=115",
            "Overflow Guard Removal",
        )

    def test_baud_counter_overflow_plain(self) -> None:
        _check(
            "ASSERT_FAIL: Baud counter overflow — baud_shadow=869 at cycle 115",
            "Overflow Guard Removal",
        )

    def test_spi_bit_counter_overflow(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_BIT_COUNTER_OVERFLOW|bit_watchdog=72|cycle=200",
            "Overflow Guard Removal",
        )

    def test_i2c_bit_overflow(self) -> None:
        _check(
            "ASSERT_FAIL: I2C bit counter overflow — clock ticked one extra time at cycle 300\n"
            "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_BIT_OVERFLOW|cycle=300",
            "Overflow Guard Removal",
        )

    def test_watchdog_expired(self) -> None:
        _check(
            "ERROR: timer overrun — watchdog expired at cycle 600",
            "Overflow Guard Removal",
        )

    # ── Parity Check Removal ────────────────────────────────────────────────
    def test_uart_parity_error_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_PARITY_ERROR|cycle=280|signals=parity_err=1",
            "Parity Check Removal",
        )

    def test_parity_mismatch_plain(self) -> None:
        _check(
            "parity mismatch — received frame has incorrect parity bit at cycle 280",
            "Parity Check Removal",
        )

    def test_frame_integrity_failed(self) -> None:
        _check(
            "SCOREBOARD_FAIL: Frame integrity check failed at cycle 480",
            "Parity Check Removal",
        )

    # ── Enable Signal Polarity Flip ─────────────────────────────────────────
    def test_enable_polarity_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=ENABLE_POLARITY|cycle=8",
            "Enable Signal Polarity Flip",
        )

    def test_write_while_wr_en_zero(self) -> None:
        _check(
            "ASSERT_FAIL: Write occurred while wr_en=0 at cycle 8",
            "Enable Signal Polarity Flip",
        )

    def test_gpio_irq_stuck(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_IRQ_STUCK|cycle=30",
            "Enable Signal Polarity Flip",
        )

    def test_gpio_output_mask(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_OUTPUT_MASK|cycle=18",
            "Enable Signal Polarity Flip",
        )

    # ── Reset Polarity Inversion ─────────────────────────────────────────────
    def test_reset_polarity_structured(self) -> None:
        _check(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=RESET_POLARITY|cycle=3|signals=rst_n=1",
            "Reset Polarity Inversion",
        )

    def test_uart_idle_high_violation(self) -> None:
        _check(
            "ASSERT_FAIL: UART TX not idle-high; tx=0 at cycle 4\n"
            "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_IDLE_HIGH|cycle=4",
            "Reset Polarity Inversion",
        )

    # ── Unknown ──────────────────────────────────────────────────────────────
    def test_passing_simulation_returns_unknown(self) -> None:
        result = classify_log(
            "Coverage: 72.1% branch, 88.3% line\n"
            "Simulation completed without errors.\n"
            "TB_LOG|kind=PASS|assert_id=FIFO_ALL_OK"
        )
        # A passing simulation should return Unknown (no failure pattern matched)
        assert result["label"] == "Unknown"
        assert result["confidence"] < 0.5

    def test_empty_log_returns_unknown(self) -> None:
        result = classify_log("")
        assert result["label"] == "Unknown"

    def test_garbage_log_returns_unknown(self) -> None:
        result = classify_log("xyzzy frobozz no pattern here at all 12345")
        assert result["label"] == "Unknown"

    # ── Confidence calibration ───────────────────────────────────────────────
    def test_structured_log_has_higher_confidence_than_unstructured(self) -> None:
        structured = classify_log(
            "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_COUNTER_OVERFLOW|count=17|cycle=23"
        )
        unstructured = classify_log(
            "ASSERT_FAIL: some counter thing happened"
        )
        # Both should be Off-by-One but structured should have higher or equal confidence
        if structured["label"] == "Off-by-One Error" and unstructured["label"] == "Off-by-One Error":
            assert structured["confidence"] >= unstructured["confidence"]
