from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .base import Mutation
from .registry import REGISTRY

_NOISE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b",
    r"/tmp/[\w/]+",
    r"seed\s*=\s*\d+",
    r"Simulation time: \d+ ns",
]


def behavioral_hash(log: str) -> str:
    normalized = log
    for pattern in _NOISE_PATTERNS:
        normalized = re.sub(pattern, "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class MutationResult:
    mutation_id: str
    label: str
    tier: int
    description: str
    mutated_source: str


class MutationEngine:
    def __init__(self, base_source: str) -> None:
        self.base_source = base_source
        self.mutations: List[Mutation] = REGISTRY.create_all()

    def apply_all(self) -> List[MutationResult]:
        results: List[MutationResult] = []
        for mutation in self.mutations:
            mutated = mutation.apply(self.base_source)
            results.append(
                MutationResult(
                    mutation_id=mutation.spec.mutation_id,
                    label=mutation.spec.label,
                    tier=mutation.spec.tier,
                    description=mutation.spec.description,
                    mutated_source=mutated,
                )
            )
        return results

    def write_mutations(self, out_dir: Path) -> List[MutationResult]:
        out_dir.mkdir(parents=True, exist_ok=True)
        results = self.apply_all()
        for result in results:
            (out_dir / f"{result.mutation_id}.v").write_text(result.mutated_source)
        return results
