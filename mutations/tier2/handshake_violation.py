import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class HandshakeViolationMutation(Mutation):
    spec = MutationSpec(
        mutation_id="handshake_violation",
        label="Handshake Protocol Violation",
        tier=2,
        description="Remove ready gating in valid/ready handshake.",
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        mutated = re.sub(r"\s*&&\s*ready\b", "", sanitized)
        return restore(mutated, mapping)
