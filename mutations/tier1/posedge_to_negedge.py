import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class PosedgeToNegedgeMutation(Mutation):
    spec = MutationSpec(
        mutation_id="posedge_to_negedge",
        label="Edge Sensitivity Flip",
        tier=1,
        description="Flip posedge sensitivity to negedge.",
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        mutated = re.sub(r"\bposedge\b", "negedge", sanitized)
        return restore(mutated, mapping)
