import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class OverflowGuardRemovalMutation(Mutation):
    """Tier 2 — Remove the overflow/underflow guard in counter decrement/increment.

    Targets patterns like:
        if (bit_cnt > 0) ...      → removes guard (bit_cnt decrements below 0)
        if (count < MAX - 1) ...  → removes the -1 tightening guard
        if (baud_cnt < DIV - 1)   → changes limit to DIV (one extra tick)

    Without the guard the counter wraps to its maximum value (for unsigned),
    producing a burst of extra clock ticks or state transitions that violate
    protocol timing.  The testbench detects this as a counter overflow assertion.
    """

    spec = MutationSpec(
        mutation_id="overflow_guard_removal",
        label="Overflow Guard Removal",
        tier=2,
        description=(
            "Remove the -1 tightening guard in baud/bit counter comparisons, "
            "allowing the counter to tick one extra time and violating protocol timing."
        ),
    )

    # Pattern: `< SOMETHING - 1`  →  `< SOMETHING`
    # Covers: baud_cnt < BAUD_DIV - 1, clk_cnt < CLK_DIV - 1, etc.
    _MINUS_ONE = re.compile(
        r"(<\s*)(\w+)\s*-\s*1\b",
        re.MULTILINE,
    )

    # Pattern: `bit_cnt > 0`  →  `bit_cnt >= 0`  (always true → never stops)
    _BIT_GT_ZERO = re.compile(r"\bbit_cnt\s*>\s*0\b", re.MULTILINE)

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        # Remove the "- 1" guard from counter limit comparisons
        mutated = self._MINUS_ONE.sub(r"\1\2", sanitized)
        # Weaken > 0 to >= 0 (always true for unsigned)
        mutated = self._BIT_GT_ZERO.sub("bit_cnt >= 0", mutated)
        return restore(mutated, mapping)
