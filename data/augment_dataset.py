import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

# Configuration for augmentation
_MODULES = ["fifo", "uart_tx", "spi_master", "gpio", "i2c_master"]

_NOISE_LINES = [
    "[INFO] Initializing testbench...",
    "[INFO] Resetting DUT...",
    "[INFO] Sequence started.",
    "[INFO] Configuration phase complete.",
    "[DEBUG] Transaction CRC matched.",
    "[DEBUG] Buffer level: {count} words.",
    "[DEBUG] Clock cycle: {cycle}.",
    "[WARN] Minor jitter detected on sclk.",
    "[INFO] Phase shifted for sampling.",
]

_ASSERTION_VARIANTS = {
    "Off-by-One Error": [
        "ASSERT_FAIL: Counter reached {count} at cycle {cycle}. Expected < {count}.",
        "TB_LOG|Cycle {cycle}|FIFO_BOUND_FAIL|count={count}|max=16",
        "ERROR: Pointer overflow at address {addr}."
    ],
    "Handshake Protocol Violation": [
        "ASSERT_FAIL: Handshake timeout at cycle {cycle}. Valid high but Ready low.",
        "TB_LOG|Cycle {cycle}|READY_STUCK|tx_valid=1|ready=0",
        "PROTOCOL_ERR: Unexpected signal state at cycle {cycle}."
    ],
    "Data Integrity Error": [
        "ASSERT_FAIL: Scoreboard mismatch at cycle {cycle}. Expected {expected}, got {actual}.",
        "TB_LOG|Cycle {cycle}|DATA_MISMATCH|exp={expected}|act={actual}",
        "DATA_CORRUPT: Mismatch on bus [31:0] at cycle {cycle}."
    ],
    "Overflow Guard Removal": [
        "ASSERT_FAIL: Counter overflow at cycle {cycle}. count={count}",
        "TB_LOG|Cycle {cycle}|OVERFLOW_FAIL|max=64|count={count}",
        "CRITICAL: Resource exhaustion detected."
    ],
    "Reset Polarity Inversion": [
        "ASSERT_FAIL: DUT active while reset asserted at cycle {cycle}.",
        "TB_LOG|Cycle {cycle}|RESET_POLARITY|rst_n=0|active=1",
        "UART_IDLE_HIGH: TX line low after reset."
    ],
    "Enable Signal Polarity Flip": [
        "ASSERT_FAIL: Write observed when wr_en is 0 at cycle {cycle}.",
        "TB_LOG|Cycle {cycle}|WRITE_DISABLE_FAIL|wr_en=0|written={data}",
        "SPI_CS_DEASSERTED: Activity outside Chip Select."
    ],
    "Edge Sensitivity Flip": [
        "ASSERT_FAIL: State transition on wrong edge at cycle {cycle}.",
        "TB_LOG|Cycle {cycle}|EDGE_FAIL|clock=negedge|triggered=1",
        "TIMING_VIOLATION: Setup time violation."
    ],
    "Data Width Truncation": [
        "ASSERT_FAIL: Data width mismatch. Expected 8 bits, got {count}.",
        "TB_LOG|Cycle {cycle}|WIDTH_FAIL|exp=8|act={count}",
        "PARITY_ERR: Data bus corruption detected."
    ],
    "Parity Check Removal": [
        "ASSERT_FAIL: Parity error not detected at cycle {cycle}.",
        "TB_LOG|Cycle {cycle}|PARITY_FAIL|parity_expected=1|found=0",
        "PROTOCOL_ERR: Invalid parity bit."
    ],
    "Handshake Protocol Violation (UART)": [
        "ASSERT_FAIL: UART Ready stuck after byte transfer at cycle {cycle}.",
        "TB_LOG|Cycle {cycle}|UART_READY_STUCK|busy=1|done=1"
    ]
}

_SUFFIX_LINES = [
    "Simulation failed at cycle {cycle}.",
    "Cleaning up and exiting...",
    "Aborted by testbench at cycle {cycle}.",
    "%Error: simulation aborted after first failure.",
    "VCD dump closed.",
]


def _rand_cycle(rng: random.Random, base: int = 20, spread: int = 200) -> int:
    return base + rng.randint(0, spread)


def _rand_int(rng: random.Random, lo: int = 0, hi: int = 255) -> int:
    return rng.randint(lo, hi)


def _fill(template: str, rng: random.Random) -> str:
    cycle = _rand_cycle(rng)
    count = _rand_cycle(rng, base=16, spread=48)
    expected = _rand_int(rng, 5, 250)
    actual = _rand_int(rng, 0, 255)
    addr = _rand_int(rng, 0, 0xFFFF)
    data = _rand_int(rng, 0, 0xFF)
    return (
        template
        .replace("{cycle}", str(cycle))
        .replace("{count}", str(count))
        .replace("{expected}", str(expected))
        .replace("{actual}", str(actual))
        .replace("{addr}", f"{addr:04x}")
        .replace("{data}", f"{data:02x}")
    )


def _noise_prefix(rng: random.Random, n_lines: int = 3) -> str:
    lines = [_fill(rng.choice(_NOISE_LINES), rng) for _ in range(n_lines)]
    return "\n".join(lines) + "\n"


def _alternate_assertion(label: str, rng: random.Random) -> str:
    variants = _ASSERTION_VARIANTS.get(label)
    if not variants:
        return _fill("ASSERT_FAIL: Unknown failure at cycle {cycle}", rng)
    return _fill(rng.choice(variants), rng)


def _suffix(rng: random.Random) -> str:
    return _fill(rng.choice(_SUFFIX_LINES), rng)


def augment_sample(row: Dict, n_aug: int, rng: random.Random) -> List[Dict]:
    """Generate *n_aug* augmented variants of a single JSONL row."""
    label = row.get("label", "Unknown")
    results: List[Dict] = []

    for i in range(n_aug):
        # Deterministic state per augmentation index
        aug_rng = random.Random(rng.getrandbits(32) + i)
        aug = deepcopy(row)

        # Strategy 1: noise injection (50% probability, 1–4 extra lines)
        noise = ""
        if aug_rng.random() > 0.5:
            noise = _noise_prefix(aug_rng, aug_rng.randint(1, 4))

        # Strategy 2: assertion variation
        core_assertion = _alternate_assertion(label, aug_rng)

        # Strategy 3: cycle jitter (handled in _fill)
        suffix = _suffix(aug_rng)

        aug["log"] = noise + core_assertion + "\n" + suffix
        aug["augmented"] = True
        aug["aug_id"] = i
        aug["module"] = aug_rng.choice(_MODULES)

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
    if in_path.exists():
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
