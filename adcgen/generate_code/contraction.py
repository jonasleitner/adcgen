from collections.abc import Sequence
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
import itertools
import json

from ..expression import TermContainer
from ..indices import Index, Indices, sort_idx_canonical


_config_file = "config.json"


@dataclass(frozen=True, slots=True)
class Sizes:
    """
    Explicit sizes for each of the spaces (occ, virt, ...).
    Used to estimate the costs of a contraction.
    """
    core: int = 0
    occ: int = 0
    virt: int = 0
    general: int = 0

    @staticmethod
    def from_dict(input: dict[str, int]) -> "Sizes":
        """
        Construct an instance from dictionary. The size of the "general" space
        is evaluated on the fly as sum of the sizes of the other spaces
        if not provided.
        """
        if "general" not in input:
            input["general"] = sum(input.values())
        return Sizes(**input)

    @staticmethod
    def from_config() -> "Sizes":
        """
        Construct an instance using the values in the config file
        (by default: "config.json"). The size of the "general" space is
        evaluated on the fly as sum of the sizes of the other spaces if not
        present in the config file.
        """
        config_file = Path(__file__).parent.resolve() / _config_file
        sizes: dict[str, int] | None = (
            json.load(open(config_file, "r")).get("sizes", None)
        )
        if sizes is None:
            raise KeyError(f"Invalid config file {config_file}. "
                           "Missing key 'sizes'.")
        return Sizes.from_dict(sizes)


class Contraction:
    """
    Represents a single contration of n objects.

    Parameters
    ----------
    indices: tuple[tuple[Index]]
        The indices of the contracted tensors.
    names: tuple[str]
        The names of the contracted tensors.
    term_target_indices: tuple[Index]
        The target indices of the term the contraction belongs to.
    """
    # use counter that essentially counts how many class instances have
    # been created
    # -> unique id for every instance
    # -> easy to differentiate and identify individual instances
    _base_name = "contraction"
    _instance_counter = itertools.count(0, 1)

    # fallback sizes to estimate the costs of a contraction
    _sizes = Sizes.from_config()

    def __init__(self, indices: Sequence[tuple[Index, ...]],
                 names: Sequence[str],
                 term_target_indices: Sequence[Index]) -> None:
        if not isinstance(indices, tuple):
            indices = tuple(indices)
        if isinstance(names, str):
            names = (names,)
        elif not isinstance(names, tuple):
            names = tuple(names)

        self.indices: tuple[tuple[Index, ...], ...] = indices
        self.names: tuple[str, ...] = names
        self.contracted: tuple[Index, ...] = tuple()
        self.target: tuple[Index, ...] = tuple()
        self.scaling: Scaling
        self.id: int = next(self._instance_counter)
        self.contraction_name: str = f"{self._base_name}_{self.id}"
        self._determine_contracted_and_target(term_target_indices)
        self._determine_scaling()

    def __str__(self):
        return (f"Contraction(indices={self.indices}, names={self.names}, "
                f"contracted={self.contracted}, target={self.target}, "
                f"scaling={self.scaling}), id={self.id}, "
                f"contraction_name={self.contraction_name})")

    def __repr__(self):
        return self.__str__()

    def _determine_contracted_and_target(self,
                                         term_target_indices: Sequence[Index]
                                         ) -> None:
        """
        Determines and sets the contracted and target indices on the
        contraction using the provided target indices of the term
        the contraction is a part of. In case the target indices of the
        contraction contain the same indices as the target indices of the
        term, the target indices of the term will be used instead.
        """
        contracted, target = self._split_contracted_and_target(
            self.indices, term_target_indices
        )
        # sort the indices canonical
        contracted = sorted(contracted, key=sort_idx_canonical)
        target = sorted(target, key=sort_idx_canonical)
        # if the contraction is an outer contraction, we have to use the
        # provided target indices as target indices since their order
        # might be different from the canonical order.
        if sorted(term_target_indices, key=sort_idx_canonical) == target:
            target = term_target_indices
        self.contracted = tuple(contracted)
        self.target = tuple(target)

    @staticmethod
    def _split_contracted_and_target(indices: Sequence[tuple[Index, ...]],
                                     term_target_indices: Sequence[Index]
                                     ) -> tuple[list[Index], list[Index]]:
        """
        Splits the given indices in contracted and target indices using
        the provided target indices of the term the contraction is a
        part of.
        """
        idx_counter = Counter(itertools.chain.from_iterable(indices))
        contracted: list[Index] = []
        target: list[Index] = []
        for idx, count in idx_counter.items():
            if count == 1 or idx in term_target_indices:
                target.append(idx)
            else:
                contracted.append(idx)
        return contracted, target

    def _determine_scaling(self) -> None:
        """Determine the computational and memory scaling of the contraction"""
        contracted_by_space = Counter(idx.space for idx in self.contracted)
        target_by_space = Counter(idx.space for idx in self.target)
        # computational scaling
        componentwise = {
            space: contracted_by_space[space] + target_by_space[space]
            for space in Indices.base
        }
        comp_scaling = ScalingComponent(total=sum(componentwise.values()),
                                        **componentwise)
        # memory scaling
        componentwise = {
            space: target_by_space[space] for space in Indices.base
        }
        mem_scaling = ScalingComponent(total=len(self.target),
                                       **componentwise)
        # overall scaling
        self.scaling = Scaling(computational=comp_scaling, memory=mem_scaling)

    def evaluate_costs(self, sizes: Sizes | None = None
                       ) -> tuple[int, int]:
        """
        Estimate the costs of the contraction. Returns a tuple containing
        the flop count and the memory foot print of the result tensor.

        Parameters
        ----------
        sizes: dict[str, int] | Sizes | None, optional
            The sizes of the individual spaces used to estimate the
            computational costs and the memory footprint of the contraction.
        """
        if sizes is None:
            sizes = self._sizes
        return self.scaling.evaluate_costs(sizes)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Contraction):
            return False
        return (self.indices == other.indices and
                self.names == other.names and
                self.contracted == other.contracted and
                self.target == other.target and self.scaling == other.scaling)

    @staticmethod
    def is_contraction(name: str) -> bool:
        return name.startswith(Contraction._base_name)


def term_memory_requirements(term: TermContainer) -> "ScalingComponent":
    """Determines the maximum memory requirements for the given term."""
    mem_scaling: list[ScalingComponent] = []
    for obj in term.objects:
        space = obj.space
        scaling = {"total": len(space)}
        for field in fields(ScalingComponent):
            if field.name == "total":
                continue
            scaling[field.name] = space.count(field.name[0])
        mem_scaling.append(ScalingComponent(**scaling))
    return max(mem_scaling)


@dataclass(frozen=True, slots=True, order=True)
class Scaling:
    computational: "ScalingComponent"
    memory: "ScalingComponent"

    def evaluate_costs(self, sizes: Sizes) -> tuple[int, int]:
        """
        Estimate the computational costs and the memory footprint using the
        provided sizes for the spaces.
        """
        return (self.computational.evaluate_costs(sizes),
                self.memory.evaluate_costs(sizes))


@dataclass(frozen=True, slots=True, order=True)
class ScalingComponent:
    total: int
    general: int
    virt: int
    occ: int
    core: int

    def evaluate_costs(self, sizes: Sizes) -> int:
        """
        Estimate the costs of the component using the provided sizes for the
        spaces.
        """
        costs = 1
        for field in fields(sizes):
            base = getattr(sizes, field.name)
            power = getattr(self, field.name, None)
            assert power is not None
            if base:
                costs *= base ** power
        return costs
