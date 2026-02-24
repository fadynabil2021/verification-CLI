"""data/generate_dataset.py — Synthetic dataset generation pipeline.

Generates a labeled JSONL dataset by applying mutation engine to RTL modules,
running Verilator simulation (optional), and parsing failure logs.

Supports:
  - Single module generation (original FIFO flow)
  - Multi-module batch generation (FIFO, UART, SPI, GPIO, I2C)
  - --no-sim mode using deterministic synthetic logs (no Verilator needed)

Usage (single module, no sim):
    python -m data.generate_dataset --no-sim

Usage (multi-module, no sim — generates 9 mutations × 5 modules = 45 base samples):
    python -m data.generate_dataset --no-sim --all-modules

Usage (with Verilator, single module):
    python -m data.generate_dataset --rtl data/raw/fifo.v --module fifo
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mutations  # noqa: F401 — trigger mutation registration
from mutations.engine import MutationEngine, behavioral_hash

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent
RAW_DIR      = BASE_DIR / "raw"
MUTATED_DIR  = BASE_DIR / "mutated"
LOGS_DIR     = BASE_DIR / "logs"
DATASET_PATH = BASE_DIR / "dataset.jsonl"
TB_DIR       = Path("testbenches")

# ---------------------------------------------------------------------------
# Multi-module configuration
# ---------------------------------------------------------------------------
MODULE_CONFIG: Dict[str, Dict] = {
    "fifo": {
        "rtl":    RAW_DIR / "fifo.v",
        "tb":     TB_DIR  / "fifo_tb.sv",
        "top":    "fifo_tb",
    },
    "uart_tx": {
        "rtl":    RAW_DIR / "uart_tx.v",
        "tb":     TB_DIR  / "uart_tx_tb.sv",
        "top":    "uart_tx_tb",
    },
    "spi_master": {
        "rtl":    RAW_DIR / "spi_master.v",
        "tb":     TB_DIR  / "spi_master_tb.sv",
        "top":    "spi_master_tb",
    },
    "gpio": {
        "rtl":    RAW_DIR / "gpio.v",
        "tb":     TB_DIR  / "gpio_tb.sv",
        "top":    "gpio_tb",
    },
    "i2c_master": {
        "rtl":    RAW_DIR / "i2c_master.v",
        "tb":     None,   # no dedicated TB yet — uses synthetic logs only
        "top":    None,
    },
}

# ---------------------------------------------------------------------------
# Synthetic log templates per (mutation_id, module)
# These represent deterministic, module-aware failure signatures.
# Format is chosen to exercise the log parser fully (TB_LOG, assert_id, etc.)
# ---------------------------------------------------------------------------
_SYNTHETIC_LOGS: Dict[Tuple[str, str], str] = {
    # ---- FIFO ---------------------------------------------------------------
    ("nonblocking_to_blocking",  "fifo"):
        "SCOREBOARD_FAIL: Data mismatch expected=8 actual=0 at cycle 42\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_DATA_MISMATCH|expected=8|actual=0|cycle=42|"
        "signals=count=3,wptr=4,rptr=0,valid=1,ready=1,wr_en=0,rd_en=1\n"
        "Simulation terminated...",

    ("posedge_to_negedge",       "fifo"):
        "ASSERT_FAIL: Counter overflow at cycle 19\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_COUNTER_OVERFLOW|count=17|cycle=19|"
        "signals=count=17,wptr=1,rptr=0,full=1,empty=0\n"
        "Simulation terminated...",

    ("reset_inversion",          "fifo"):
        "ASSERT_FAIL: Backpressure violation — data pushed while ready=0\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_BACKPRESSURE_WRITE|cycle=12|"
        "signals=write_fire=1,ready=0,count=5\n"
        "Simulation terminated...",

    ("counter_boundary_violation","fifo"):
        "ASSERT_FAIL: Counter overflow at cycle 23\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_COUNTER_OVERFLOW|count=16|cycle=23|"
        "signals=count=16,wptr=0,rptr=0,full=1,empty=0\n"
        "Simulation terminated...",

    ("handshake_violation",       "fifo"):
        "ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle 31\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_BACKPRESSURE_WRITE|cycle=31|"
        "signals=write_fire=1,ready=0,count=8\n"
        "Simulation terminated...",

    ("enable_polarity_flip",      "fifo"):
        "SCOREBOARD_FAIL: Data mismatch expected=15 actual=0 at cycle 50\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_DATA_MISMATCH|expected=15|actual=0|cycle=50|"
        "signals=count=0,wr_en=0,rd_en=0\n"
        "Simulation terminated...",

    ("data_width_truncation",     "fifo"):
        "SCOREBOARD_FAIL: Data mismatch expected=200 actual=72 at cycle 64\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_DATA_MISMATCH|expected=200|actual=72|cycle=64|"
        "signals=count=3,wptr=4,rptr=1\n"
        "Simulation terminated...",

    ("overflow_guard_removal",    "fifo"):
        "ASSERT_FAIL: Counter overflow at cycle 55\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=FIFO_COUNTER_OVERFLOW|count=17|cycle=55|"
        "signals=count=17,wptr=1,rptr=0,full=1\n"
        "Simulation terminated...",

    ("parity_check_removal",      "fifo"):
        "SCOREBOARD_FAIL: Data mismatch expected=10 actual=3 at cycle 78\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_DATA_MISMATCH|expected=10|actual=3|cycle=78|"
        "signals=count=4,wr_en=1,rd_en=0\n"
        "Simulation terminated...",

    # ---- UART TX ------------------------------------------------------------
    ("nonblocking_to_blocking",  "uart_tx"):
        "SCOREBOARD_FAIL: Data mismatch expected=165 actual=0 at cycle 210\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_UART_FRAME|expected=165|actual=0|cycle=210|"
        "signals=tx=0,ready=1,valid=0\n"
        "Simulation terminated...",

    ("posedge_to_negedge",       "uart_tx"):
        "ASSERT_FAIL: Clock edge violation — negedge sensitivity at cycle 44\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_BAUD_OVERFLOW|cycle=44|"
        "signals=tx=1,ready=0,valid=1,baud_shadow=192\n"
        "Simulation terminated...",

    ("reset_inversion",          "uart_tx"):
        "ASSERT_FAIL: UART TX not idle-high; tx=0 at cycle 4\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_IDLE_HIGH|cycle=4|"
        "signals=tx=0,ready=1,valid=0\n"
        "Simulation terminated...",

    ("counter_boundary_violation","uart_tx"):
        "ASSERT_FAIL: Baud counter overflow — baud_shadow=868 at cycle 100\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_BAUD_OVERFLOW|baud_shadow=868|cycle=100|"
        "signals=baud_shadow=868,tx=0,ready=0\n"
        "Simulation terminated...",

    ("handshake_violation",       "uart_tx"):
        "ASSERT_FAIL: UART ready stayed high after valid handshake at cycle 22\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_READY_STUCK|cycle=22|"
        "signals=ready=1,valid=1,tx=1\n"
        "Simulation terminated...",

    ("enable_polarity_flip",      "uart_tx"):
        "SCOREBOARD_FAIL: Frame mismatch expected=0xAB actual=0x00 at cycle 300\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_UART_FRAME|expected=171|actual=0|cycle=300|"
        "signals=tx=1,ready=0,valid=0\n"
        "Simulation terminated...",

    ("data_width_truncation",     "uart_tx"):
        "SCOREBOARD_FAIL: Frame mismatch expected=0xC3 actual=0x43 at cycle 320\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_UART_FRAME|expected=195|actual=67|cycle=320|"
        "signals=shift_reg=0x43\n"
        "Simulation terminated...",

    ("overflow_guard_removal",    "uart_tx"):
        "ASSERT_FAIL: Baud counter overflow — baud_shadow=869 at cycle 115\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_BAUD_OVERFLOW|baud_shadow=869|cycle=115|"
        "signals=baud_shadow=869,state=DATA\n"
        "Simulation terminated...",

    ("parity_check_removal",      "uart_tx"):
        "ASSERT_FAIL: UART parity error asserted at cycle 280\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=UART_PARITY_ERROR|cycle=280|"
        "signals=parity_err=1,tx=1,ready=0\n"
        "Simulation terminated...",

    # ---- SPI Master ---------------------------------------------------------
    ("nonblocking_to_blocking",  "spi_master"):
        "SCOREBOARD_FAIL: SPI MISO mismatch expected=0xDE actual=0x00 at cycle 88\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=222|actual=0|cycle=88|"
        "signals=sclk=0,cs_n=1,mosi=0\n"
        "Simulation terminated...",

    ("posedge_to_negedge",       "spi_master"):
        "ASSERT_FAIL: SPI transfer watchdog expired — bit counter may have overflowed at cycle 200\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_BIT_COUNTER_OVERFLOW|bit_watchdog=72|cycle=200|"
        "signals=sclk=1,cs_n=0,done=0\n"
        "Simulation terminated...",

    ("reset_inversion",          "spi_master"):
        "ASSERT_FAIL: SPI SCLK active while CS_N high at cycle 5\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_SCLK_ACTIVE_WHEN_IDLE|cycle=5|"
        "signals=sclk=1,cs_n=1\n"
        "Simulation terminated...",

    ("counter_boundary_violation","spi_master"):
        "ASSERT_FAIL: SPI transfer watchdog expired at cycle 180\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_BIT_COUNTER_OVERFLOW|bit_watchdog=68|cycle=180|"
        "signals=bit_cnt=0,clk_cnt=4,done=0\n"
        "Simulation terminated...",

    ("handshake_violation",       "spi_master"):
        "ASSERT_FAIL: SPI CS_N deasserted during active transfer at cycle 92\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_CS_DEASSERTED_DURING_TRANSFER|cycle=92|"
        "signals=cs_n=1,sclk=1\n"
        "Simulation terminated...",

    ("enable_polarity_flip",      "spi_master"):
        "SCOREBOARD_FAIL: SPI MISO mismatch expected=0xAA actual=0x00 at cycle 110\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=170|actual=0|cycle=110|"
        "signals=cs_n=1,sclk=0\n"
        "Simulation terminated...",

    ("data_width_truncation",     "spi_master"):
        "SCOREBOARD_FAIL: SPI MISO mismatch expected=0xF0 actual=0x70 at cycle 140\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=240|actual=112|cycle=140|"
        "signals=shift_rx=0x70\n"
        "Simulation terminated...",

    ("overflow_guard_removal",    "spi_master"):
        "ASSERT_FAIL: SPI transfer watchdog expired — bit counter overflowed at cycle 200\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=SPI_BIT_COUNTER_OVERFLOW|cycle=200|"
        "signals=bit_cnt=255,clk_cnt=0\n"
        "Simulation terminated...",

    ("parity_check_removal",      "spi_master"):
        "SCOREBOARD_FAIL: SPI MISO mismatch expected=0x55 actual=0x15 at cycle 160\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_SPI_MISO_MISMATCH|expected=85|actual=21|cycle=160|"
        "signals=sclk=0,cs_n=1\n"
        "Simulation terminated...",

    # ---- GPIO ---------------------------------------------------------------
    ("nonblocking_to_blocking",  "gpio"):
        "SCOREBOARD_FAIL: GPIO data_out mismatch expected=0xF0 actual=0x00 at cycle 15\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_GPIO|expected=240|actual=0|cycle=15|"
        "signals=dir_reg=0xF0,data_in=0xFF,wr_en=1\n"
        "Simulation terminated...",

    ("posedge_to_negedge",       "gpio"):
        "ASSERT_FAIL: GPIO IRQ stuck high for >1 cycle at cycle 30\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_IRQ_STUCK|cycle=30|"
        "signals=irq=1,pin_drive=0x01\n"
        "Simulation terminated...",

    ("reset_inversion",          "gpio"):
        "SCOREBOARD_FAIL: GPIO write errors=1 at cycle 6\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_GPIO|write_errors=1|irq_errors=0|cycle=6|"
        "signals=dir_reg=0xFF,data_in=0xAA,wr_en=1\n"
        "Simulation terminated...",

    ("counter_boundary_violation","gpio"):
        "ASSERT_FAIL: GPIO output enable mask violated at cycle 18\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_OUTPUT_MASK|cycle=18|"
        "signals=data_in=0xFF,dir_reg=0x0F\n"
        "Simulation terminated...",

    ("handshake_violation",       "gpio"):
        "ASSERT_FAIL: GPIO write occurred while wr_en=0 at cycle 12\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=ENABLE_POLARITY|cycle=12|"
        "signals=wr_en=0,data_in=0xAA,dir_reg=0xFF\n"
        "Simulation terminated...",

    ("enable_polarity_flip",      "gpio"):
        "ASSERT_FAIL: Write occurred while wr_en=0 at cycle 8\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=ENABLE_POLARITY|cycle=8|"
        "signals=wr_en=0,data_in=0x55,dir_reg=0xF0\n"
        "Simulation terminated...",

    ("data_width_truncation",     "gpio"):
        "SCOREBOARD_FAIL: GPIO data_out mismatch expected=0xA5 actual=0x25 at cycle 20\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_GPIO|expected=165|actual=37|cycle=20|"
        "signals=dir_reg=0xFF,in_reg=0xA5\n"
        "Simulation terminated...",

    ("overflow_guard_removal",    "gpio"):
        "ASSERT_FAIL: IRQ stuck high for >1 cycle at cycle 40\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=GPIO_IRQ_STUCK|cycle=40|"
        "signals=irq=1,in_reg=0x01,in_reg_prev=0x01\n"
        "Simulation terminated...",

    ("parity_check_removal",      "gpio"):
        "SCOREBOARD_FAIL: GPIO data_out mismatch expected=0xC3 actual=0x43 at cycle 25\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_GPIO|expected=195|actual=67|cycle=25|"
        "signals=dir_reg=0xFF\n"
        "Simulation terminated...",

    # ---- I2C Master ---------------------------------------------------------
    ("nonblocking_to_blocking",  "i2c_master"):
        "SCOREBOARD_FAIL: I2C data byte mismatch expected=0x42 actual=0x00 at cycle 500\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_I2C_DATA|expected=66|actual=0|cycle=500|"
        "signals=sda=0,scl=0,state=DATA\n"
        "Simulation terminated...",

    ("posedge_to_negedge",       "i2c_master"):
        "ASSERT_FAIL: I2C SCL timing violation at cycle 120\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_SCL_TIMING|cycle=120|"
        "signals=scl=0,sda=1,clk_cnt=250\n"
        "Simulation terminated...",

    ("reset_inversion",          "i2c_master"):
        "ASSERT_FAIL: I2C SDA not idle-high after reset at cycle 3\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_SDA_IDLE|cycle=3|"
        "signals=sda=0,scl=1\n"
        "Simulation terminated...",

    ("counter_boundary_violation","i2c_master"):
        "ASSERT_FAIL: I2C bit counter overflow — clock ticked one extra time at cycle 300\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_BIT_OVERFLOW|cycle=300|"
        "signals=bit_cnt=255,clk_cnt=0,state=ADDR\n"
        "Simulation terminated...",

    ("handshake_violation",       "i2c_master"):
        "ASSERT_FAIL: I2C NACK received during ACK window at cycle 450\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_NACK_ERROR|cycle=450|"
        "signals=nack=1,sda=1,scl=1\n"
        "Simulation terminated...",

    ("enable_polarity_flip",      "i2c_master"):
        "ASSERT_FAIL: I2C write started while wr_en=0 at cycle 10\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=ENABLE_POLARITY|cycle=10|"
        "signals=wr_en=0,state=START\n"
        "Simulation terminated...",

    ("data_width_truncation",     "i2c_master"):
        "SCOREBOARD_FAIL: I2C data byte mismatch expected=0xF5 actual=0x75 at cycle 520\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_I2C_DATA|expected=245|actual=117|cycle=520|"
        "signals=shift_reg=0x75\n"
        "Simulation terminated...",

    ("overflow_guard_removal",    "i2c_master"):
        "ASSERT_FAIL: I2C clock counter overflow at cycle 600\n"
        "TB_LOG|kind=ASSERT_FAIL|assert_id=I2C_BIT_OVERFLOW|cycle=600|"
        "signals=clk_cnt=250,state=DATA\n"
        "Simulation terminated...",

    ("parity_check_removal",      "i2c_master"):
        "SCOREBOARD_FAIL: I2C frame integrity check failed at cycle 480\n"
        "TB_LOG|kind=SCOREBOARD_FAIL|assert_id=SB_I2C_DATA|expected=255|actual=127|cycle=480|"
        "signals=state=ACK2\n"
        "Simulation terminated...",
}


def _synthetic_log(mutation_id: str, module: str = "fifo") -> str:
    """Return a deterministic synthetic log for (mutation_id, module) pair."""
    key = (mutation_id, module)
    if key in _SYNTHETIC_LOGS:
        return _SYNTHETIC_LOGS[key]
    # Fallback: generate a generic log based on mutation label pattern
    return (
        f"ASSERT_FAIL: Unknown failure signature for mutation={mutation_id} in module={module}\n"
        "Simulation terminated..."
    )


def run_verilator(rtl_path: Path, tb_path: Path, top: str) -> Tuple[str, Optional[str]]:
    if shutil.which("verilator") is None:
        return "", "verilator not found"

    with tempfile.TemporaryDirectory(prefix="verilog_ai_") as tmp:
        tmp_dir = Path(tmp)
        local_rtl = tmp_dir / rtl_path.name
        local_tb  = tmp_dir / tb_path.name
        local_rtl.write_text(rtl_path.read_text())
        local_tb.write_text(tb_path.read_text())

        # Copy all .sv / .svh includes from testbench directory
        for pattern in ("*.sv", "*.svh"):
            for dep in tb_path.parent.glob(pattern):
                (tmp_dir / dep.name).write_text(dep.read_text())

        cmd = [
            "verilator", "-Wall", "-sv", "--timing", "--binary",
            "--top-module", top,
            str(local_rtl), str(local_tb),
            "-o", "sim.out",
        ]
        build = subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True)
        if build.returncode != 0:
            return build.stdout + "\n" + build.stderr, "verilator build failed"

        run = subprocess.run(
            [str(tmp_dir / "sim.out")], cwd=tmp_dir, capture_output=True, text=True
        )
        log = run.stdout + "\n" + run.stderr
        if run.returncode != 0:
            return log, "simulation failed"
        return log, None


# ---------------------------------------------------------------------------
# Log parsing (unchanged from Phase 1 — extended later in Phase 3+)
# ---------------------------------------------------------------------------

def _extract(pattern: str, log: str) -> Optional[Tuple[str, ...]]:
    match = re.search(pattern, log)
    return match.groups() if match else None


def _extract_assert_id(log: str) -> Optional[str]:
    match = re.search(r"(?:ASSERT_ID|assert_id|id)\s*[:=]\s*([A-Za-z0-9_]+)", log)
    return match.group(1) if match else None


def _extract_signal_dump(log: str) -> Optional[str]:
    sig_match = re.search(r"(?:signal_dump|signals?)\s*[:=]\s*([^\n]+)", log)
    if sig_match:
        return sig_match.group(1).strip()
    pairs = re.findall(
        r"\b(count|wptr|rptr|full|empty|ready|valid|wr_en|rd_en|write_fire|expected|actual"
        r"|baud_shadow|bit_cnt|shift_reg|tx|cs_n|sclk|mosi|irq|parity_err)\s*=\s*([0-9A-Za-z_x]+)",
        log,
    )
    return ", ".join(f"{k}={v}" for k, v in pairs) if pairs else None


def _parse_tb_structured_fields(log: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for line in log.splitlines():
        if not line.startswith("TB_LOG|"):
            continue
        for part in line[len("TB_LOG|"):].split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key.strip()] = value.strip()
        if parsed:
            return parsed
    return parsed


def parse_log_for_label(log: str) -> Tuple[str, str, float]:
    """Parse a simulation log and return (label, explanation, confidence)."""
    structured = _parse_tb_structured_fields(log)
    if structured:
        kind      = structured.get("kind", "").upper()
        assert_id = structured.get("assert_id", "")
        cycle     = structured.get("cycle")
        signals   = structured.get("signals")

        if "SCOREBOARD_FAIL" in kind:
            expected = structured.get("expected")
            actual   = structured.get("actual")
            details  = []
            if assert_id: details.append(f"assertion_id={assert_id}")
            if expected is not None and actual is not None:
                details.append(f"expected={expected}, actual={actual}")
            if signals: details.append(f"signals={signals}")
            at_cycle = f" at cycle {cycle}" if cycle else ""
            suffix   = f" ({'; '.join(details)})" if details else ""
            return ("Data Integrity Error",
                    f"Structured scoreboard failure{at_cycle}{suffix}.", 0.92)

        if "ASSERT_FAIL" in kind:
            lower_id = assert_id.lower()
            at_cycle = f" at cycle {cycle}" if cycle else ""

            if any(t in lower_id for t in ["overflow", "counter", "boundary", "baud"]):
                details = []
                if assert_id: details.append(f"assertion_id={assert_id}")
                count = structured.get("count") or structured.get("baud_shadow")
                if count: details.append(f"count={count}")
                if signals: details.append(f"signals={signals}")
                suffix = f" ({'; '.join(details)})" if details else ""
                return ("Off-by-One Error",
                        f"Counter boundary assertion failed{at_cycle}{suffix}.", 0.92)

            if any(t in lower_id for t in ["handshake", "backpressure", "ready",
                                            "cs_deasserted", "ready_stuck", "nack"]):
                details = []
                if assert_id: details.append(f"assertion_id={assert_id}")
                write_fire = structured.get("write_fire")
                ready_val  = structured.get("ready")
                if write_fire and ready_val:
                    details.append(f"write_fire={write_fire}, ready={ready_val}")
                if signals: details.append(f"signals={signals}")
                suffix = f" ({'; '.join(details)})" if details else ""
                return ("Handshake Protocol Violation",
                        f"Ready/valid assertion failed{at_cycle}{suffix}.", 0.90)

            if any(t in lower_id for t in ["parity", "idle_high", "uart"]):
                details = []
                if assert_id: details.append(f"assertion_id={assert_id}")
                if signals: details.append(f"signals={signals}")
                suffix = f" ({'; '.join(details)})" if details else ""
                return ("Parity Check Removal",
                        f"UART frame integrity violation{at_cycle}{suffix}.", 0.88)

            if any(t in lower_id for t in ["enable", "polarity", "wr_en", "rd_en"]):
                details = []
                if assert_id: details.append(f"assertion_id={assert_id}")
                if signals: details.append(f"signals={signals}")
                suffix = f" ({'; '.join(details)})" if details else ""
                return ("Enable Signal Polarity Flip",
                        f"Enable signal polarity violation{at_cycle}{suffix}.", 0.88)

            if any(t in lower_id for t in ["irq_stuck", "irq", "gpio"]):
                details = []
                if assert_id: details.append(f"assertion_id={assert_id}")
                if signals: details.append(f"signals={signals}")
                suffix = f" ({'; '.join(details)})" if details else ""
                return ("Enable Signal Polarity Flip",
                        f"GPIO IRQ/enable assertion failed{at_cycle}{suffix}.", 0.82)

    # ---- Unstructured log parsing -------------------------------------------
    assert_id   = _extract_assert_id(log)
    signal_dump = _extract_signal_dump(log)
    cycle_match = _extract(r"cycle\s*[=:]?\s*(\d+)", log)
    cycle_text  = cycle_match[0] if cycle_match else None

    # Counter / overflow patterns
    counter = _extract(r"Counter overflow[^=]*?count[=:]?(\d+)[^@]*?at cycle (\d+)", log)
    if counter or re.search(r"Counter overflow|BAUD.*overflow|bit.counter.*overflow", log, re.I):
        count = counter[0] if counter else "?"
        cycle = counter[1] if counter else cycle_text or "?"
        details = []
        if assert_id:   details.append(f"assertion_id={assert_id}")
        if signal_dump: details.append(f"signals={signal_dump}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return ("Off-by-One Error",
                f"Counter exceeded limit (count={count}) at cycle {cycle}{suffix}.", 0.90)

    # Backpressure / handshake
    if re.search(r"Backpressure violation|write_fire=1\s+ready=0|CS_N deasserted|ready_stuck"
                 r"|NACK.*received|protocol assertion", log, re.I):
        cycle_m = _extract(r"at cycle (\d+)", log)
        cycle   = cycle_m[0] if cycle_m else cycle_text or "?"
        details = []
        if assert_id:   details.append(f"assertion_id={assert_id}")
        if signal_dump: details.append(f"signals={signal_dump}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return ("Handshake Protocol Violation",
                f"Write occurred while receiver not ready at cycle {cycle}{suffix}.", 0.85)

    # Scoreboard / data mismatch
    mismatch = _extract(r"Data mismatch expected[=:]?(\w+)\s+actual[=:]?(\w+)\s+at cycle\s+(\d+)", log)
    if mismatch or re.search(r"SCOREBOARD_FAIL|Data mismatch|Unexpected data|frame.*mismatch", log, re.I):
        if mismatch:
            expected, actual, cycle = mismatch
            details = []
            if assert_id:   details.append(f"assertion_id={assert_id}")
            if signal_dump: details.append(f"signals={signal_dump}")
            suffix = f" ({'; '.join(details)})" if details else ""
            return ("Data Integrity Error",
                    f"Scoreboard mismatch at cycle {cycle} "
                    f"(expected={expected}, actual={actual}){suffix}.", 0.88)
        return ("Data Integrity Error",
                "Scoreboard mismatch detected in simulation log. "
                "Check data path and pointer logic.", 0.78)

    # Parity errors
    if re.search(r"parity.*error|parity.*mismatch|frame.*integrity", log, re.I):
        details = []
        if assert_id:   details.append(f"assertion_id={assert_id}")
        if signal_dump: details.append(f"signals={signal_dump}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return ("Parity Check Removal",
                f"UART parity / frame integrity error detected{suffix}.", 0.83)

    # Enable polarity
    if re.search(r"enable.*polarity|wr_en.*0|write.*while.*disabled|idle.*high.*tx", log, re.I):
        details = []
        if assert_id:   details.append(f"assertion_id={assert_id}")
        if signal_dump: details.append(f"signals={signal_dump}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return ("Enable Signal Polarity Flip",
                f"Enable signal polarity violation detected{suffix}.", 0.80)

    # Assert ID fallback
    if assert_id:
        lower_id = assert_id.lower()
        details  = []
        if signal_dump: details.append(f"signals={signal_dump}")
        if cycle_text:  details.append(f"cycle={cycle_text}")
        suffix   = f" ({'; '.join(details)})" if details else ""

        if any(t in lower_id for t in ["overflow", "counter", "boundary", "baud"]):
            return ("Off-by-One Error",
                    f"Assertion indicates a counter boundary issue (assertion_id={assert_id}){suffix}.", 0.78)
        if any(t in lower_id for t in ["handshake", "backpressure", "ready", "nack", "cs"]):
            return ("Handshake Protocol Violation",
                    f"Assertion indicates a ready/valid protocol issue (assertion_id={assert_id}){suffix}.", 0.78)
        if any(t in lower_id for t in ["scoreboard", "mismatch", "data", "frame"]):
            return ("Data Integrity Error",
                    f"Assertion indicates a data mismatch (assertion_id={assert_id}){suffix}.", 0.75)
        if any(t in lower_id for t in ["parity", "uart", "idle"]):
            return ("Parity Check Removal",
                    f"Assertion indicates a parity/frame error (assertion_id={assert_id}){suffix}.", 0.73)
        if any(t in lower_id for t in ["enable", "polarity", "wr_en", "irq"]):
            return ("Enable Signal Polarity Flip",
                    f"Assertion indicates an enable polarity issue (assertion_id={assert_id}){suffix}.", 0.73)

    return ("Unknown", "No known pattern matched. Review the log manually.", 0.3)


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------

def generate_dataset(
    base_rtl: Path,
    module_name: str,
    use_sim: bool = True,
    include_inert: bool = False,
    top_module: str = "fifo_tb",
    tb_path: Optional[Path] = None,
) -> List[Dict]:
    base_source = base_rtl.read_text()
    engine      = MutationEngine(base_source)
    results     = engine.write_mutations(MUTATED_DIR)

    dataset: List[Dict] = []
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    base_log       = ""
    base_hash      = ""
    base_sim_error: Optional[str] = None

    if use_sim and tb_path is not None:
        base_log, base_sim_error = run_verilator(base_rtl, tb_path, top_module)
        if base_sim_error is None:
            base_hash = behavioral_hash(base_log)
            (LOGS_DIR / f"{module_name}_golden.log").write_text(base_log)

    for result in results:
        log   = ""
        inert = False

        if use_sim and tb_path is not None and base_sim_error is None:
            rtl_path = MUTATED_DIR / f"{result.mutation_id}.v"
            log, sim_err = run_verilator(rtl_path, tb_path, top_module)
            if sim_err is None:
                inert = behavioral_hash(log) == base_hash
        else:
            log = _synthetic_log(result.mutation_id, module_name)

        log_path = LOGS_DIR / f"{module_name}_{result.mutation_id}.log"
        log_path.write_text(log)

        if use_sim and tb_path is not None and base_sim_error is None:
            label, explanation, confidence = parse_log_for_label(log)
            label_source = "log_parser"
        else:
            label, explanation, confidence = result.label, result.description, 1.0
            label_source = "mutation"

        row: Dict = {
            "log":          log,
            "label":        label,
            "explanation":  explanation,
            "confidence":   confidence,
            "label_source": label_source,
            "module":       module_name,
            "mutation_id":  result.mutation_id,
            "tier":         result.tier,
            "is_inert":     inert,
        }

        if not inert or include_inert:
            dataset.append(row)

    return dataset


def generate_all_modules(
    use_sim: bool = True,
    include_inert: bool = False,
) -> List[Dict]:
    """Generate dataset for all configured modules and merge."""
    all_rows: List[Dict] = []
    for module_name, cfg in MODULE_CONFIG.items():
        rtl_path: Path = cfg["rtl"]
        if not rtl_path.exists():
            print(f"[SKIP] RTL not found: {rtl_path}")
            continue
        tb_path : Optional[Path] = cfg.get("tb")
        top     : str            = cfg.get("top") or "unknown_tb"

        print(f"[GEN ] {module_name} ({rtl_path.name})")
        rows = generate_dataset(
            base_rtl    = rtl_path,
            module_name = module_name,
            use_sim     = use_sim and (tb_path is not None),
            include_inert=include_inert,
            top_module  = top,
            tb_path     = tb_path,
        )
        print(f"       → {len(rows)} samples")
        all_rows.extend(rows)

    return all_rows


def write_jsonl(dataset: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset JSONL (single or multi-module)"
    )
    parser.add_argument("--rtl",    default=str(RAW_DIR / "fifo.v"),
                        help="Path to base RTL file (ignored with --all-modules)")
    parser.add_argument("--module", default="fifo",
                        help="Module name tag (ignored with --all-modules)")
    parser.add_argument("--out",    default=str(DATASET_PATH),
                        help="Output JSONL path")
    parser.add_argument("--top",    default="fifo_tb",
                        help="Top module for Verilator")
    parser.add_argument("--no-sim", action="store_true",
                        help="Skip Verilator; use deterministic synthetic logs")
    parser.add_argument("--all-modules", action="store_true",
                        help="Generate for all configured RTL modules")
    parser.add_argument("--include-inert", action="store_true",
                        help="Include inert mutations in dataset")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.all_modules:
        print("Multi-module dataset generation")
        dataset = generate_all_modules(
            use_sim       = not args.no_sim,
            include_inert = args.include_inert,
        )
    else:
        tb_cfg  = MODULE_CONFIG.get(args.module, {})
        tb_path = tb_cfg.get("tb")
        dataset = generate_dataset(
            base_rtl    = Path(args.rtl),
            module_name = args.module,
            use_sim     = not args.no_sim and tb_path is not None,
            include_inert=args.include_inert,
            top_module  = tb_cfg.get("top", args.top),
            tb_path     = tb_path,
        )

    write_jsonl(dataset, out_path)
    print(f"\nWrote {len(dataset)} samples to {out_path}")


if __name__ == "__main__":
    main()
