from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class MutationSpec:
    mutation_id: str
    label: str
    tier: int
    description: str


class Mutation(ABC):
    spec: MutationSpec

    @abstractmethod
    def apply(self, source: str) -> str:
        raise NotImplementedError
