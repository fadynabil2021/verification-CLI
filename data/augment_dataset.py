"""data/augment_dataset.py — Dataset augmentation pipeline.

Applies four augmentation strategies to expand a small base JSONL dataset
to 400+ samples suitable for fine-tuning:

    1. **Noise Injection** — prepends 1–5 random "passing" log lines before the
       failure signature, teaching the model to find the signal in noise.

    2. **Assertion Variation** — rewrites the assertion message using 2–3
       alternative phrasings with identical semantic content.

    3. **Cycle Jitter** — randomises the cycle number in log messages,
       preventing the model from memorising cycle offsets.

    4. **Module Stamp** — tags each sample with a module name prefix
       (UART, SPI, GPIO, I2C, FIFO) so the model learns module-agnostic
       failure patterns.

Usage:
    python -m data.augment_dataset \
        --input  data/dataset.jsonl \
        --output data/dataset_augmented.jsonl \
        --n-aug  16          # augmentations per base sample
"""
from __future__ import annotations

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Noise lines that appear in real "passing" simulation output
# ---------------------------------------------------------------------------
_NOISE_LINES: List[str] = [
    "- ReadData: 0x00",
    "- WriteData: 0xff",
    "Simulation state: RUNNING",
    "Clock edge: posedge at time {cycle}ns",
    "Memory access: addr=0x{addr:04x} data=0x{data:02x}",
    "DUT: signal_stable, no change",
    "TB: waiting for handshake...",
    "verilator: Warning-UNOPTFLAT: Signal unoptimized: count",
    "Coverage: 67.4% branch, 81.2% line",
    "%Info: testbench/tb.sv:42: $display called",
    "Assertion monitor: all passing (cycle {cycle})",
    "Statistics: {cycle} cycles elapsed, 0 failures",
    "FSM state: IDLE",
    "FSM state: TRANS",
    "DUT output: ready=1 valid=0",
]

# ---------------------------------------------------------------------------
# Alternative assertion phrasings (keyed by canonical label)
# ---------------------------------------------------------------------------
_ASSERTION_VARIANTS: Dict[str, List[str]] = {
    "Off-by-One Error": [
        "ASSERT_FAIL: Counter overflow at cycle {cycle}",
        "ASSERT_FAIL: Counter exceeded MAX_DEPTH — count={count} at cycle {cycle}",
        "ASSERT_FAIL: FIFO_COUNTER_OVERFLOW triggered (count={count}) at time {cycle}ns",
        "ERROR: boundary check failed — count value {count} >= MAX at cycle {cycle}",
        "ASSERT ID FIFO_COUNTER_OVERFLOW: count={count} violated upper bound at cycle {cycle}",
    ],
    "Handshake Protocol Violation": [
        "ASSERT_FAIL: Backpressure violation — data pushed while ready=0",
        "ASSERT_FAIL: Backpressure violation write_fire=1 ready=0 at cycle {cycle}",
        "ASSERT_FAIL: protocol assertion tripped — valid=1 ready=0 at cycle {cycle}",
        "ASSERT ID FIFO_BACKPRESSURE_WRITE: write_fire=1 ready=0 at cycle {cycle}",
        "ERROR: handshake invariant violated — write occurred without ready at cycle {cycle}",
    ],
    "Data Integrity Error": [
        "SCOREBOARD_FAIL: Data mismatch at cycle {cycle}",
        "SCOREBOARD_FAIL: Data mismatch expected={expected} actual={actual} at cycle {cycle}",
        "ASSERT ID SB_DATA_MISMATCH: expected={expected} actual={actual} at cycle {cycle}",
        "ERROR: scoreboard check failed — read data={actual} expected={expected} cycle={cycle}",
        "SCOREBOARD_FAIL: Unexpected data actual={actual} at cycle {cycle}",
    ],
    "Assignment Semantics Change": [
        "SCOREBOARD_FAIL: Data mismatch at cycle {cycle}",
        "SCOREBOARD_FAIL: Data mismatch expected={expected} actual={actual} at cycle {cycle}",
        "ASSERT ID SB_DATA_MISMATCH: unexpected value at cycle {cycle}",
        "ERROR: data path corruption detected at cycle {cycle}",
    ],
    "Edge Sensitivity Flip": [
        "ASSERT_FAIL: Counter overflow at cycle {cycle}",
        "ASSERT_FAIL: Clock edge violation — negedge sensitivity at cycle {cycle}",
        "ASSERT ID CLOCK_EDGE_FAIL: unexpected sensitivity triggered at cycle {cycle}",
        "ERROR: flip-flop sampled on wrong clock edge at cycle {cycle}",
    ],
    "Reset Polarity Inversion": [
        "ASSERT_FAIL: Backpressure violation — data pushed while ready=0",
        "ASSERT_FAIL: Reset polarity error — dut active during reset assertion at cycle {cycle}",
        "ASSERT ID RESET_POLARITY: dut not reset when rst_n=1 at cycle {cycle}",
        "ERROR: DUT active while reset should be asserted (active-low inversion) at cycle {cycle}",
    ],
    "Enable Signal Polarity Flip": [
        "SCOREBOARD_FAIL: Data mismatch expected={expected} actual={actual} at cycle {cycle}",
        "ASSERT_FAIL: Write occurred while wr_en=0 at cycle {cycle}",
        "ASSERT ID ENABLE_POLARITY: operation on disabled enable signal at cycle {cycle}",
        "ERROR: data written when wr_en was low — enable polarity inverted at cycle {cycle}",
    ],
    "Data Width Truncation": [
        "SCOREBOARD_FAIL: Data mismatch expected={expected} actual={actual} at cycle {cycle}",
        "ASSERT ID SB_WIDTH_MISMATCH: MSB truncated in shift register at cycle {cycle}",
        "ERROR: shift register MSB lost — rx_data={actual} expected={expected} at cycle {cycle}",
        "SCOREBOARD_FAIL: Unexpected data actual={actual} at cycle {cycle}",
    ],
    "Overflow Guard Removal": [
        "ASSERT_FAIL: Counter overflow at cycle {cycle}",
        "ASSERT_FAIL: Baud counter overflow — baud_shadow={count} at cycle {cycle}",
        "ASSERT ID UART_BAUD_OVERFLOW: counter exceeded limit at cycle {cycle}",
        "ERROR: timer overrun — watchdog expired at cycle {cycle}",
    ],
    "Parity Check Removal": [
        "ASSERT_FAIL: UART parity error asserted at cycle {cycle}",
        "SCOREBOARD_FAIL: Frame integrity check failed at cycle {cycle}",
        "ASSERT ID UART_PARITY_ERROR: parity_err=1 at cycle {cycle}",
        "ERROR: parity mismatch — received frame has incorrect parity bit at cycle {cycle}",
    ],
}

_MODULES: List[str] = ["fifo", "uart_tx", "spi_master", "gpio", "i2c_master"]

_SUFFIX_LINES: List[str] = [
    "Simulation terminated due to assertion failure.",
    "Simulation terminated...",
    "Aborted by testbench at cycle {cycle}.",
    "%Error: simulation aborted after first failure.",
    "VCD dump closed.",
]


def _rand_cycle(base: int = 20, spread: int = 200) -> int:
    return base + random.randint(0, spread)


def _rand_int(lo: int = 0, hi: int = 255) -> int:
    return random.randint(lo, hi)


def _fill(template: str) -> str:
    cycle = _rand_cycle()
    count = _rand_cycle(base=16, spread=48)
    expected = _rand_int(5, 250)
    actual = _rand_int(0, 255)
    addr = _rand_int(0, 0xFFFF)
    data = _rand_int(0, 0xFF)
    return (
        template
        .replace("{cycle}", str(cycle))
        .replace("{count}", str(count))
        .replace("{expected}", str(expected))
        .replace("{actual}", str(actual))
        .replace("{addr}", f"{addr:04x}")
        .replace("{data}", f"{data:02x}")
    )


def _noise_prefix(n_lines: int = 3) -> str:
    lines = [_fill(random.choice(_NOISE_LINES)) for _ in range(n_lines)]
    return "\n".join(lines) + "\n"


def _alternate_assertion(label: str) -> str:
    variants = _ASSERTION_VARIANTS.get(label)
    if not variants:
        return _fill("ASSERT_FAIL: Unknown failure at cycle {cycle}")
    return _fill(random.choice(variants))


def _suffix() -> str:
    return _fill(random.choice(_SUFFIX_LINES))


def augment_sample(row: Dict, n_aug: int, rng: random.Random) -> List[Dict]:
    """Generate *n_aug* augmented variants of a single JSONL row."""
    label = row.get("label", "Unknown")
    results: List[Dict] = []

    for i in range(n_aug):
        aug = deepcopy(row)

        # Strategy 1: noise injection (50% probability, 1–4 extra lines)
        noise = ""
        if rng.random() > 0.5:
            noise = _noise_prefix(rng.randint(1, 4))

        # Strategy 2: assertion variation
        core_assertion = _alternate_assertion(label)

        # Strategy 3: cycle jitter (already in _fill via random cycle)
        suffix = _suffix()

        aug["log"] = noise + core_assertion + "\n" + suffix
        aug["augmented"] = True
        aug["aug_id"] = i
        aug["module"] = rng.choice(_MODULES)

        results.append(aug)

    return results


def augment_dataset(
    rows: List[Dict],
    n_aug: int = 16,
    seed: int = 42,
) -> List[Dict]:
    """Augment every row and return base + augmented combined."""
    rng = random.Random(seed)
    augmented: List[Dict] = []

    for row in rows:
        # Keep original
        augmented.append(row)
        # Add augmentations
        augmented.extend(augment_sample(row, n_aug, rng))

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment a JSONL dataset by noise injection and assertion variation"
    )
    parser.add_argument("--input",  default="data/dataset.jsonl",
                        help="Input JSONL dataset")
    parser.add_argument("--output", default="data/dataset_augmented.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--n-aug", type=int, default=16,
                        help="Number of augmented variants per base sample")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    rows: List[Dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} base samples from {in_path}")
    augmented = augment_dataset(rows, n_aug=args.n_aug, seed=args.seed)
    print(f"Generated {len(augmented)} total samples ({len(rows)} base + {len(augmented)-len(rows)} augmented)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in augmented:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote augmented dataset to {out_path}")


if __name__ == "__main__":
    main()
