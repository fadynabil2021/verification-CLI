from pathlib import Path

import mutations  # noqa: F401
from mutations.engine import MutationEngine


def test_mutations_change_source():
    base = Path("data/raw/fifo.v").read_text()
    engine = MutationEngine(base)
    results = engine.apply_all()
    assert results

    for result in results:
        assert result.mutated_source != base
