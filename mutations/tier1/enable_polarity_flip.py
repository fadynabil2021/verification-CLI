import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class EnablePolarityFlipMutation(Mutation):
    """Tier 1 — Flip active-high enable signals to active-low (removes `!` or adds it).

    Targets patterns like:
        if (wr_en) ...        → if (!wr_en) ...
        if (rd_en) ...        → if (!rd_en) ...
        if (valid && ready)   → if (!valid && ready)

    This causes the controlled block to fire on the wrong polarity, leading to
    data written when disabled, or missed writes when enabled. The result is a
    Data Integrity Error that the testbench scoreboard detects.
    """

    spec = MutationSpec(
        mutation_id="enable_polarity_flip",
        label="Enable Signal Polarity Flip",
        tier=1,
        description=(
            "Flip active-high enable/write-enable checks to active-low, "
            "causing operations to occur when disabled and be suppressed when enabled."
        ),
    )

    # Matches: if (wr_en), if (rd_en), if (tx_en), if (rx_en), if (oe), if (cs)
    # Avoids already-negated: if (!wr_en)
    _PATTERN = re.compile(
        r"\bif\s*\(\s*(?!!)(\b(?:wr_en|rd_en|tx_en|rx_en|oe\b|cs\b|en\b|enable\b))\s*\)",
        re.MULTILINE,
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)
        mutated = self._PATTERN.sub(r"if (!\1)", sanitized)
        return restore(mutated, mapping)
