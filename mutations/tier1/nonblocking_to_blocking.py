import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class NonblockingToBlockingMutation(Mutation):
    spec = MutationSpec(
        mutation_id="nonblocking_to_blocking",
        label="Assignment Semantics Change",
        tier=1,
        description="Replace nonblocking assignments with blocking in procedural blocks.",
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        pattern = re.compile(r"(?m)^(\s*\w[\w\[\]\.]*\s*)<=", re.MULTILINE)
        mutated = pattern.sub(r"\1=", sanitized)
        return restore(mutated, mapping)
