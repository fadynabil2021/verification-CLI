from pathlib import Path

from data.generate_dataset import generate_dataset


def test_generate_dataset():
    base_rtl = Path("data/raw/fifo.v")
    dataset = generate_dataset(base_rtl, "fifo", use_sim=False)
    assert len(dataset) >= 5
    assert all("log" in row for row in dataset)
