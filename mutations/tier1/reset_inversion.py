import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class ResetInversionMutation(Mutation):
    spec = MutationSpec(
        mutation_id="reset_inversion",
        label="Reset Polarity Inversion",
        tier=1,
        description="Invert active-low reset checks in if statements.",
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        mutated = re.sub(r"\bif\s*\(\s*!\s*rst_n\s*\)", "if (rst_n)", sanitized)
        return restore(mutated, mapping)
