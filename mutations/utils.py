import re
from typing import Dict, Tuple

_PLACEHOLDER_PREFIX = "__SANITIZED_"

_SANITIZE_PATTERN = re.compile(
    r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"',
    re.MULTILINE | re.DOTALL,
)


def sanitize(source: str) -> Tuple[str, Dict[str, str]]:
    """Replace comments and string literals with stable placeholders."""
    mapping: Dict[str, str] = {}
    counter = 0

    def _repl(match: re.Match) -> str:
        nonlocal counter
        key = f"{_PLACEHOLDER_PREFIX}{counter}__"
        mapping[key] = match.group(0)
        counter += 1
        return key

    sanitized = _SANITIZE_PATTERN.sub(_repl, source)
    return sanitized, mapping


def restore(sanitized: str, mapping: Dict[str, str]) -> str:
    """Restore placeholders to original comment/string content."""
    restored = sanitized
    for key, val in mapping.items():
        restored = restored.replace(key, val)
    return restored
