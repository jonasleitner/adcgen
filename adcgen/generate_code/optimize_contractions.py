from ..expr_container import Term
from ..indices import get_symbols, Index
from ..sympy_objects import SymbolicTensor, KroneckerDelta

from .contraction import Contraction, ScalingComponent

from sympy import Symbol
from typing import Generator
import dataclasses
import itertools


def optimize_contractions(term: Term, target_indices: str | None = None,
                          max_itmd_dim: int | None = None
                          ) -> list[Contraction]:
    """
    Find the optimal contraction scheme with the lowest computational
    and memory scaling for a given term. Thereby, the computational scaling
    is prioritized over the memory scaling.

    Parameters
    ----------
    target_indices: str, optional
        The target indices of the term. If not given, the canonical target
        indices of the term will be used For instance, 2 occupied and 2 virtual
        indices will always be in the order 'ijab'. Therefore, target indices
        have to be provided if the result tensor has indices 'iajb'.
    max_itmd_dim: int, optional
        Upper bound for the dimensionality of intermediates created by
        inner contractions if the contractions are nested, i.e.,
        the dimensionality of the result of contr2 and contr3 is restricted in
        "contr1(contr2(contr3(...)))".
    """
    # - import (or extract) the target indices
    if target_indices is None:
        target_indices = term.target
    else:
        target_indices = tuple(get_symbols(target_indices))
    # - extract the relevant part (tensors and deltas) of the term
    #   and track on which objects each index occures
    relevant_obj_names: list[str] = []
    relevant_obj_indices: list[tuple[Index]] = []
    for obj in term.objects:
        base, exp = obj.base_and_exponent
        if obj.sympy.is_number or isinstance(base, Symbol):  # skip prefactor
            continue
        if not isinstance(base, (SymbolicTensor, KroneckerDelta)):
            raise NotImplementedError("Contractions can only be optimized for "
                                      "tensors and 'KroneckerDeltas'.")
        name, indices = obj.longname(), obj.idx
        relevant_obj_names.extend(name for _ in range(exp))
        relevant_obj_indices.extend(indices for _ in range(exp))
    assert len(relevant_obj_names) == len(relevant_obj_indices)

    if not relevant_obj_names:  # no tensors or deltas in the term
        return []
    elif len(relevant_obj_names) == 1:
        # trivial: only a single tensor/delta with exponent 1
        # - resorting of indices
        # - trace
        names, indices = relevant_obj_names[0], relevant_obj_indices[0]
        return Contraction(indices=indices, names=names,
                           target_indices=target_indices)
    # lazily find the contraction schemes
    contraction_schemes = _optimize_contractions(
        relevant_obj_names=tuple(relevant_obj_names),
        relevant_obj_indices=tuple(relevant_obj_indices),
        target_indices=target_indices, max_itmd_dim=max_itmd_dim
    )
    # go through all schemes and find the one with the lowest scaling by
    # considering the:
    # 1) Maximum scaling of each field
    # 2) Sum of all scalings for each field
    # Thereby, we rely on the order in which the fields are defined on the
    # dataclass. Furthermore, the computational scaling is prioritized over the
    # memory scaling.
    optimal_scaling = None
    optimal_scheme = None
    for scheme in contraction_schemes:
        # build the scaling for current variant
        scaling = []
        mem = []
        for field in dataclasses.fields(ScalingComponent):
            comp_values = [getattr(contr.scaling.computational, field.name)
                           for contr in scheme]
            mem_values = [getattr(contr.scaling.memory, field.name)
                          for contr in scheme]
            scaling.extend([max(comp_values), sum(comp_values)])
            mem.extend([max(mem_values), sum(mem_values)])
        scaling.extend(mem)
        # compare the scaling
        if optimal_scaling is None or scaling < optimal_scaling:
            optimal_scheme = scheme
            optimal_scaling = scaling
    return optimal_scheme


def _optimize_contractions(relevant_obj_names: tuple[str],
                           relevant_obj_indices: tuple[tuple[Index]],
                           target_indices: tuple[Index],
                           max_itmd_dim: int | None = None
                           ) -> Generator[list[Contraction], None, None]:
    """
    Find the optimal contractions for the given relevant objects of a term.
    """
    assert len(relevant_obj_indices) == len(relevant_obj_names)
    if len(relevant_obj_names) < 2:
        raise ValueError("Need at least 2 objects to define a contraction.")

    # split the relevant objects into subgroups that share contracted indices
    # and therefore should be contracted simultaneously
    connected_groups = _group_objects(
        obj_indices=relevant_obj_indices, target_indices=target_indices
    )

    for group in connected_groups:
        contr_indices = [relevant_obj_indices[pos] for pos in group]
        contr_names = [relevant_obj_names[pos] for pos in group]
        contraction = Contraction(indices=contr_indices, names=contr_names,
                                  target_indices=target_indices)
        # if the contraction is not an outer contraction we have to check
        # the dimensionality of the intermediate tensor
        if max_itmd_dim is not None and \
                contraction.target == target_indices and \
                len(contraction.target) > max_itmd_dim:
            continue
        # remove the contracted names and indices
        remaining_pos = [pos for pos in range(len(relevant_obj_names))
                         if pos not in group]
        remaining_names = ("contraction", *(relevant_obj_names[pos]
                                            for pos in remaining_pos))
        remaining_indices = (contraction.target, *(relevant_obj_indices[pos]
                                                   for pos in remaining_pos))
        # there are no objects left to contract -> we are done
        if len(remaining_names) == 1:
            yield [contraction]
            continue
        # recurse to build further contractions
        completed_schemes = _optimize_contractions(
            relevant_obj_names=remaining_names,
            relevant_obj_indices=remaining_indices,
            target_indices=target_indices, max_itmd_dim=max_itmd_dim
        )
        for contraction_scheme in completed_schemes:
            contraction_scheme.insert(0, contraction)
            yield contraction_scheme


def _group_objects(obj_indices: tuple[tuple[Index]],
                   target_indices: tuple[Index]) -> tuple[tuple[int]]:
    """
    Split the provided relevant objects into subgroups that share common
    contracted indices.
    """
    # remove all target indices
    contracted_indices = [
        set(idx for idx in indices if idx not in target_indices)
        for indices in obj_indices
    ]
    # track the on which objects a contracted index appears
    idx_occurences: dict[Index, list[int]] = {}
    for pos, indices in enumerate(contracted_indices):
        for idx in indices:
            if idx not in idx_occurences:
                idx_occurences[idx] = []
            idx_occurences[idx].append(pos)

    # list for grouped objects and isolated objects (outer products)
    groups: list[tuple[int]] = []
    grouped_objects = set()
    # iterate over all pairs of contracted indices (objects)
    for (pos1, indices1), (pos2, indices2) in \
            itertools.combinations(enumerate(contracted_indices), 2):
        # check if the objects have any common contracted indices
        # or if we already have a group containing both objects (avoid
        # duplicated groups)
        common_contracted = indices1 & indices2
        if not common_contracted or \
                any(pos1 in positions and pos2 in positions
                    for positions in groups):
            continue

        positions = {pos1, pos2}
        # self-consistently pull in new objects holding the contracted indices
        # and update the common contracted indices for those new positions.
        # This corresponds to maximizing the group size.
        while True:
            # pull in all positions that also hold the common contracted idx
            new_positions = set()
            for idx in common_contracted:
                new_positions.update(idx_occurences[idx])
            if new_positions == positions:  # no new positions
                break
            positions = new_positions
            # pull in additional contracted indices
            new_common_contracted = set()
            for p1, p2 in itertools.combinations(positions, 2):
                new_common_contracted.update(
                    contracted_indices[p1] & contracted_indices[p2]
                )
            if common_contracted == new_common_contracted:
                # no new contracted indices pulled in
                break
            common_contracted = new_common_contracted
        groups.append(tuple(sorted(positions)))
        grouped_objects.update(positions)
    # collect all objects with no overlap to any other object
    # -> outer product or a number
    # -> we need to try each object (except self) to multiply with each
    #    isolated object
    isolated = tuple(pos for pos in range(len(obj_indices))
                     if pos not in grouped_objects)
    for pos in isolated:
        groups.extend(tuple(sorted([p, pos])) for p in range(len(obj_indices))
                      if p != pos)
    return tuple(groups)
