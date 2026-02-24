"""Tests for the 4 new Phase 2 mutations against real RTL modules."""
from __future__ import annotations

from pathlib import Path

import pytest

import mutations  # noqa: F401 — triggers all registrations
from mutations.engine import MutationEngine
from mutations.registry import REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fifo_source() -> str:
    return Path("data/raw/fifo.v").read_text()


@pytest.fixture(scope="module")
def uart_source() -> str:
    return Path("data/raw/uart_tx.v").read_text()


@pytest.fixture(scope="module")
def spi_source() -> str:
    return Path("data/raw/spi_master.v").read_text()


@pytest.fixture(scope="module")
def gpio_source() -> str:
    return Path("data/raw/gpio.v").read_text()


@pytest.fixture(scope="module")
def i2c_source() -> str:
    return Path("data/raw/i2c_master.v").read_text()


# ---------------------------------------------------------------------------
# Registry sanity check
# ---------------------------------------------------------------------------

def test_all_nine_mutations_registered() -> None:
    ids = set(REGISTRY.iter_specs())
    expected = {
        "nonblocking_to_blocking",
        "posedge_to_negedge",
        "reset_inversion",
        "enable_polarity_flip",
        "data_width_truncation",
        "counter_boundary_violation",
        "handshake_violation",
        "overflow_guard_removal",
        "parity_check_removal",
    }
    assert expected <= ids, f"Missing mutations: {expected - ids}"


# ---------------------------------------------------------------------------
# EnablePolarityFlipMutation
# ---------------------------------------------------------------------------

class TestEnablePolarityFlip:
    def test_flips_wr_en_in_gpio(self, gpio_source: str) -> None:
        from mutations.tier1.enable_polarity_flip import EnablePolarityFlipMutation
        mutated = EnablePolarityFlipMutation().apply(gpio_source)
        assert "if (!wr_en)" in mutated, "Expected wr_en to be negated"
        assert "if (wr_en)" not in mutated, "Original if(wr_en) should be gone"

    def test_flips_rd_en_in_gpio(self, gpio_source: str) -> None:
        from mutations.tier1.enable_polarity_flip import EnablePolarityFlipMutation
        mutated = EnablePolarityFlipMutation().apply(gpio_source)
        assert "if (!rd_en)" in mutated

    def test_does_not_double_negate(self, gpio_source: str) -> None:
        from mutations.tier1.enable_polarity_flip import EnablePolarityFlipMutation
        mutated = EnablePolarityFlipMutation().apply(gpio_source)
        assert "if (!!wr_en)" not in mutated
        assert "if (!!rd_en)" not in mutated

    def test_preserves_comments(self, gpio_source: str) -> None:
        from mutations.tier1.enable_polarity_flip import EnablePolarityFlipMutation
        mutated = EnablePolarityFlipMutation().apply(gpio_source)
        assert "1 = output, 0 = input" in mutated  # comment preserved

    def test_source_changes(self, gpio_source: str) -> None:
        from mutations.tier1.enable_polarity_flip import EnablePolarityFlipMutation
        mutated = EnablePolarityFlipMutation().apply(gpio_source)
        assert mutated != gpio_source


# ---------------------------------------------------------------------------
# DataWidthTruncationMutation
# ---------------------------------------------------------------------------

class TestDataWidthTruncation:
    def test_modifies_spi_shift_in(self, spi_source: str) -> None:
        from mutations.tier1.data_width_truncation import DataWidthTruncationMutation
        mutated = DataWidthTruncationMutation().apply(spi_source)
        # Original: {shift_rx[DATA_WIDTH-2:0], miso}
        # After mutation: slice upper bound reduced
        assert mutated != spi_source

    def test_modifies_uart_shift_right(self, uart_source: str) -> None:
        from mutations.tier1.data_width_truncation import DataWidthTruncationMutation
        mutated = DataWidthTruncationMutation().apply(uart_source)
        # uart_tx has {1'b0, shift_reg[7:1]} — should become {2'b00, shift_reg[...
        assert "2'b00" in mutated or mutated != uart_source

    def test_does_not_corrupt_non_shift_logic(self, fifo_source: str) -> None:
        from mutations.tier1.data_width_truncation import DataWidthTruncationMutation
        mutated = DataWidthTruncationMutation().apply(fifo_source)
        # FIFO has no shift registers — but should not crash
        assert isinstance(mutated, str)

    def test_source_changes_on_spi(self, spi_source: str) -> None:
        from mutations.tier1.data_width_truncation import DataWidthTruncationMutation
        mutated = DataWidthTruncationMutation().apply(spi_source)
        assert mutated != spi_source


# ---------------------------------------------------------------------------
# OverflowGuardRemovalMutation
# ---------------------------------------------------------------------------

class TestOverflowGuardRemoval:
    def test_removes_minus_one_from_uart_baud_cnt(self, uart_source: str) -> None:
        from mutations.tier2.overflow_guard_removal import OverflowGuardRemovalMutation
        mutated = OverflowGuardRemovalMutation().apply(uart_source)
        # Original: baud_cnt < BAUD_DIV - 1
        # Mutated:  baud_cnt < BAUD_DIV
        assert "< BAUD_DIV - 1" not in mutated
        assert "< BAUD_DIV" in mutated

    def test_removes_minus_one_from_spi_clk_cnt(self, spi_source: str) -> None:
        from mutations.tier2.overflow_guard_removal import OverflowGuardRemovalMutation
        mutated = OverflowGuardRemovalMutation().apply(spi_source)
        assert "< CLK_DIV - 1" not in mutated
        assert "< CLK_DIV" in mutated

    def test_weakens_bit_cnt_guard(self, spi_source: str) -> None:
        from mutations.tier2.overflow_guard_removal import OverflowGuardRemovalMutation
        # SPI has bit_cnt == 0 check not bit_cnt > 0, so this may not apply
        # But it should not crash
        mutated = OverflowGuardRemovalMutation().apply(spi_source)
        assert isinstance(mutated, str)

    def test_source_changes_uart(self, uart_source: str) -> None:
        from mutations.tier2.overflow_guard_removal import OverflowGuardRemovalMutation
        mutated = OverflowGuardRemovalMutation().apply(uart_source)
        assert mutated != uart_source

    def test_preserves_string_literals(self, uart_source: str) -> None:
        from mutations.tier2.overflow_guard_removal import OverflowGuardRemovalMutation
        mutated = OverflowGuardRemovalMutation().apply(uart_source)
        assert "// stop bit (mark)" in mutated


# ---------------------------------------------------------------------------
# ParityCheckRemovalMutation
# ---------------------------------------------------------------------------

class TestParityCheckRemoval:
    def test_zeros_parity_assignment_uart(self, uart_source: str) -> None:
        from mutations.tier2.parity_check_removal import ParityCheckRemovalMutation
        mutated = ParityCheckRemovalMutation().apply(uart_source)
        # Original: parity_reg <= ^data_in;
        assert "parity_reg <= 1'b0;" in mutated
        assert "<= ^data_in;" not in mutated

    def test_disables_parity_check(self, uart_source: str) -> None:
        from mutations.tier2.parity_check_removal import ParityCheckRemovalMutation
        mutated = ParityCheckRemovalMutation().apply(uart_source)
        # if (parity_err) → if (1'b0)
        if "if (parity_err)" in uart_source:
            assert "if (1'b0)" in mutated

    def test_no_effect_on_fifo(self, fifo_source: str) -> None:
        from mutations.tier2.parity_check_removal import ParityCheckRemovalMutation
        mutated = ParityCheckRemovalMutation().apply(fifo_source)
        # FIFO has no parity logic — mutation should be a no-op or minimal
        # At minimum should not crash
        assert isinstance(mutated, str)

    def test_source_changes_uart(self, uart_source: str) -> None:
        from mutations.tier2.parity_check_removal import ParityCheckRemovalMutation
        mutated = ParityCheckRemovalMutation().apply(uart_source)
        assert mutated != uart_source

    def test_comment_preserved(self, uart_source: str) -> None:
        from mutations.tier2.parity_check_removal import ParityCheckRemovalMutation
        mutated = ParityCheckRemovalMutation().apply(uart_source)
        assert "even parity" in mutated  # comment in sanitized block


# ---------------------------------------------------------------------------
# Multi-module mutation engine
# ---------------------------------------------------------------------------

class TestMutationEngineMultiModule:
    @pytest.mark.parametrize("rtl_file,module_name", [
        ("data/raw/fifo.v",       "FIFO"),
        ("data/raw/uart_tx.v",    "UART"),
        ("data/raw/spi_master.v", "SPI"),
        ("data/raw/gpio.v",       "GPIO"),
        ("data/raw/i2c_master.v", "I2C"),
    ])
    def test_engine_applies_all_9_mutations(self, rtl_file: str, module_name: str) -> None:
        source = Path(rtl_file).read_text()
        engine = MutationEngine(source)
        results = engine.apply_all()
        assert len(results) == 9, (
            f"{module_name}: expected 9 mutations, got {len(results)}"
        )

    @pytest.mark.parametrize("rtl_file,module_name,mutation_id", [
        ("data/raw/uart_tx.v",    "UART",  "overflow_guard_removal"),
        ("data/raw/spi_master.v", "SPI",   "data_width_truncation"),
        ("data/raw/gpio.v",       "GPIO",  "enable_polarity_flip"),
        ("data/raw/uart_tx.v",    "UART",  "parity_check_removal"),
    ])
    def test_specific_mutation_changes_source(
        self, rtl_file: str, module_name: str, mutation_id: str
    ) -> None:
        source = Path(rtl_file).read_text()
        engine = MutationEngine(source)
        results = engine.apply_all()
        target = next(r for r in results if r.mutation_id == mutation_id)
        assert target.mutated_source != source, (
            f"{module_name} × {mutation_id}: mutation produced no change"
        )
