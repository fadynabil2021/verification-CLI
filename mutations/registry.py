from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Type

from .base import Mutation


class MutationRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[Mutation]] = {}

    def register(self, mutation_cls: Type[Mutation]) -> Type[Mutation]:
        mutation_id = mutation_cls.spec.mutation_id
        if mutation_id in self._registry:
            raise ValueError(f"Duplicate mutation id: {mutation_id}")
        self._registry[mutation_id] = mutation_cls
        return mutation_cls

    def create_all(self) -> List[Mutation]:
        return [cls() for cls in self._registry.values()]

    def iter_specs(self) -> Iterable[str]:
        return self._registry.keys()


REGISTRY = MutationRegistry()


def register_mutation(cls: Type[Mutation]) -> Type[Mutation]:
    return REGISTRY.register(cls)
