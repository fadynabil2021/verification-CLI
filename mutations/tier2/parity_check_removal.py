import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class ParityCheckRemovalMutation(Mutation):
    """Tier 2 — Remove parity computation/check from the data path.

    Targets patterns like:
        parity_reg <= ^data_in;          → parity_reg <= 1'b0;
        if (parity_err) ...              → if (1'b0) ...  (check disabled)
        parity_out <= parity_reg ^ tx;   → parity_out <= tx;

    Without parity the transmitter silently sends corrupted frames.
    The testbench detects this as a frame integrity / scoreboard failure.
    """

    spec = MutationSpec(
        mutation_id="parity_check_removal",
        label="Parity Check Removal",
        tier=2,
        description=(
            "Remove parity computation by zeroing the parity register assignment, "
            "causing the transmitter to produce frames with incorrect parity "
            "and the receiver scoreboard to detect a data integrity error."
        ),
    )

    # Pattern A: parity_reg <= ^expr;  →  parity_reg <= 1'b0;
    _PARITY_ASSIGN = re.compile(
        r"(parity_reg\s*<=\s*)\^[^;]+;",
        re.MULTILINE,
    )

    # Pattern B: if (parity_err)  →  if (1'b0)
    _PARITY_CHECK = re.compile(r"\bif\s*\(\s*parity_err\s*\)", re.MULTILINE)

    # Pattern C: XOR reduction in parity output line
    _PARITY_OUT = re.compile(
        r"(\bparity\w*\s*<=\s*parity_reg\s*)\^\s*\w+\s*;",
        re.MULTILINE,
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        mutated = self._PARITY_ASSIGN.sub(r"\g<1>1'b0;", sanitized)
        mutated = self._PARITY_CHECK.sub("if (1'b0)", mutated)
        mutated = self._PARITY_OUT.sub(r"\1;", mutated)
        return restore(mutated, mapping)
