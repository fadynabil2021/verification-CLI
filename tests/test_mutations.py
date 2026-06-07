from pathlib import Path

import mutations  # noqa: F401
from mutations.engine import MutationEngine


def test_mutations_change_source():
    base = Path("data/raw/fifo.v").read_text()
    engine = MutationEngine(base)
    results = engine.apply_all()
    assert results

    # Only assert change for mutations that are applicable to fifo.v
    applicable_ids = {
        "nonblocking_to_blocking",
        "posedge_to_negedge",
        "reset_inversion",
        "counter_boundary_violation",
    }
    for result in results:
        if result.mutation_id in applicable_ids:
            assert result.mutated_source != base, f"Mutation {result.mutation_id} did not change source"
