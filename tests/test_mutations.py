from pathlib import Path

import mutations  # noqa: F401
from mutations.engine import MutationEngine


def test_mutations_change_source():
    """Test that the engine produces variants. Note that some may be inert for specific modules."""
    base = Path("data/raw/fifo.v").read_text()
    engine = MutationEngine(base)
    results = engine.apply_all()
    assert len(results) >= 5
    
    # Assert that at least 5 mutations produced a change (the Tier 1/2 core set)
    changed = [r for r in results if r.mutated_source != base]
    assert len(changed) >= 5, f"Expected at least 5 mutations to change the source, got {len(changed)}"
