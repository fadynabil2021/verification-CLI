import re

from ..base import Mutation, MutationSpec
from ..registry import register_mutation
from ..utils import sanitize, restore


@register_mutation
class DataWidthTruncationMutation(Mutation):
    """Tier 1 — Truncate the MSB of shift register assignments.

    Targets patterns like:
        signal <= {signal[N-2:0], input_bit};          (shift-in from LSB)
        signal <= {1'b0, signal[N-1:1]};               (shift-right)
        signal <= {signal[DATA_WIDTH-2:0], miso};      (parametric width)

    Reduces the slice upper bound by 1 (literal) or appends -1 (parametric),
    effectively dropping the MSB of the shift chain.  Also replaces the 1'b0
    prefix in shift-right concatenations with 2'b00, widening the constant and
    causing a type mismatch / silent truncation.
    """

    spec = MutationSpec(
        mutation_id="data_width_truncation",
        label="Data Width Truncation",
        tier=1,
        description=(
            "Truncate MSB of shift register by reducing the slice upper bound by 1 "
            "or replacing the 1'b0 prefix with 2'b00, creating a data width mismatch "
            "that corrupts the output."
        ),
    )

    # Matches shift-right:  {1'b0, signal[N-1:1]}
    _SHIFT_RIGHT = re.compile(r"\{1'b0,\s*(\w+)\[", re.MULTILINE)

    # Matches shift-in with literal upper bound:  {sig[6:0], tail}
    _SHIFT_IN_LIT = re.compile(
        r"\{(\w+)\[(\d+):0\],\s*(\w+)\}",
        re.MULTILINE,
    )

    # Matches shift-in with parametric upper bound: {sig[DATA_WIDTH-2:0], tail}
    _SHIFT_IN_PARAM = re.compile(
        r"\{(\w+)\[(\w+(?:[-+]\d+)?):0\],\s*(\w+)\}",
        re.MULTILINE,
    )

    def apply(self, source: str) -> str:
        sanitized, mapping = sanitize(source)

        # Strategy A: replace 1'b0 prefix with 2'b00 (shift-right)
        mutated = self._SHIFT_RIGHT.sub(r"{2'b00, \1[", sanitized)

        # Strategy B: reduce literal slice upper bound by 1
        def _dec_lit(m: re.Match) -> str:
            sig   = m.group(1)
            upper = max(0, int(m.group(2)) - 1)
            tail  = m.group(3)
            return f"{{{sig}[{upper}:0], {tail}}}"

        mutated = self._SHIFT_IN_LIT.sub(_dec_lit, mutated)

        # Strategy C: append -1 to parametric upper bound expression
        def _dec_param(m: re.Match) -> str:
            sig   = m.group(1)
            expr  = m.group(2)   # e.g. "DATA_WIDTH-2"
            tail  = m.group(3)
            # Avoid double-decrement if Strategy B already handled it
            try:
                int(expr)          # pure literal already handled above
                return m.group(0)  # unchanged
            except ValueError:
                return f"{{{sig}[{expr}-1:0], {tail}}}"

        mutated = self._SHIFT_IN_PARAM.sub(_dec_param, mutated)
        return restore(mutated, mapping)
