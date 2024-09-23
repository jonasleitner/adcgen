from ..expr_container import Term
from ..indices import Index, sort_idx_canonical

from collections import Counter
from dataclasses import dataclass, fields
import itertools


class Contraction:
    """
    Represents a single contration of n objects.

    Parameters
    ----------
    indices: tuple[tuple[Index]]
        The indices of the contracted tensors
    names: tuple[str]
        The names of the contracted tensors
    target_indices: tuple[Index]
        The target indices of the term the contraction belongs to
    """
    # use counter that essentially counts how many class instances have
    # been created
    # -> unique id for every instance
    # -> easy to differentiate and identify individual instances
    _base_name = "contraction"
    _instance_counter = itertools.count(0, 1)

    def __init__(self, indices: tuple[tuple[Index]],
                 names: tuple[str],
                 target_indices: tuple[Index]) -> None:
        self.indices: tuple[tuple[Index]] = indices
        self.names: tuple[str] = names
        self.contracted: tuple[Index] = None
        self.target: tuple[Index] = None
        self.scaling: Scaling = None
        self.id: int = next(self._instance_counter)
        self.contraction_name = f"{self._base_name}_{self.id}"
        self._determine_contracted_and_target_indices(target_indices)
        self._determine_scaling()

    def __str__(self):
        return (f"Contraction(indices={self.indices}, names={self.names}, "
                f"contracted={self.contracted}, target={self.target}, "
                f"scaling={self.scaling}), id={self.id}, "
                f"contraction_name={self.contraction_name})")

    def __repr__(self):
        return self.__str__()

    def _determine_contracted_and_target_indices(self,
                                                 target_indices: tuple[Index]
                                                 ) -> None:
        """Determines the contracted and target indices of the contraction."""
        idx_counter = Counter(itertools.chain.from_iterable(self.indices))
        # determine the contracted and target indices
        # takink into account that the einstein sum convention might not be
        # valid
        contracted = [idx for idx, n in idx_counter.items()
                      if n > 1 and idx not in target_indices]
        target = [idx for idx, n in idx_counter.items()
                  if n == 1 or idx in target_indices]
        # sort the indices canonical
        contracted = sorted(contracted, key=sort_idx_canonical)
        target = sorted(target, key=sort_idx_canonical)
        # if the contraction is an outer contraction, we have to use the
        # provided target indices as target indices since their order
        # might be different from the canonical order.
        if sorted(target_indices, key=sort_idx_canonical) == target:
            target = target_indices

        self.contracted = tuple(contracted)
        self.target = tuple(target)

    def _determine_scaling(self) -> None:
        """Determine the computational and memory scaling of the contraction"""

        all_spaces = ["general", "virt", "occ"]

        contracted_by_space = Counter(idx.space for idx in self.contracted)
        target_by_space = Counter(idx.space for idx in self.target)
        # computational scaling
        componentwise = {
            space: contracted_by_space[space] + target_by_space[space]
            for space in all_spaces
        }
        comp_scaling = ScalingComponent(total=sum(componentwise.values()),
                                        **componentwise)
        # memory scaling
        componentwise = {
            space: target_by_space[space] for space in all_spaces
        }
        mem_scaling = ScalingComponent(total=len(self.target),
                                       **componentwise)
        # overall scaling
        self.scaling = Scaling(computational=comp_scaling, memory=mem_scaling)

    def __eq__(self, other: "Contraction"):
        if not isinstance(other, Contraction):
            return False
        return (self.indices == other.indices and
                self.names == other.names and
                self.contracted == other.contracted and
                self.target == other.target and self.scaling == other.scaling)

    @staticmethod
    def is_contraction(name: str):
        return name.startswith(Contraction._base_name)


def term_memory_requirements(term: Term) -> "ScalingComponent":
    """Determines the maximum memory requirements for the given term."""
    mem_scaling = []
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


@dataclass(frozen=True, slots=True, order=True)
class ScalingComponent:
    total: int
    general: int
    virt: int
    occ: int