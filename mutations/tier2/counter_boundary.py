import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class CounterBoundaryMutation(Mutation):
    spec = MutationSpec(
        mutation_id="counter_boundary_violation",
        label="Off-by-One Error",
        tier=2,
        description="Relax counter boundary from < MAX_DEPTH to <= MAX_DEPTH.",
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        pattern = re.compile(r"\b(count\w*)\s*<\s*(MAX_DEPTH|MAX\w*)")
        mutated = pattern.sub(r"\1 <= \2", sanitized)
        return restore(mutated, mapping)
