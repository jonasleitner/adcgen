from ..expr_container import Term
from ..indices import get_symbols, Index
from ..sympy_objects import SymbolicTensor, KroneckerDelta

from .contraction import Contraction, Sizes

from sympy import Symbol
from typing import Generator
import itertools


def optimize_contractions(term: Term, target_indices: str | None = None,
                          target_spin: str | None = None,
                          max_itmd_dim: int | None = None,
                          max_n_simultaneous_contracted: int | None = None,
                          space_dims: dict[str, int] | None = None
                          ) -> list[Contraction]:
    """
    Find the optimal contraction scheme with the lowest computational
    and memory scaling for a given term. Thereby, the computational scaling
    is prioritized over the memory scaling.

    Parameters
    ----------
    term: Term
        Find the optimal contraction scheme for this term.
    target_indices: str | None, optional
        The target indices of the term. If not given, the canonical target
        indices of the term according to the Einstein sum convention
        will be used. For instance, 2 occupied and 2 virtual
        indices will always be in the order 'ijab'. Therefore, target indices
        have to be provided if the result tensor has indices 'iajb'.
    target_spin: str | None, optional
        The spin of the target indices, e.g., "aabb" for
        alpha, alpha, beta, beta. If not given, target indices without spin
        will be used.
    max_itmd_dim: int | None, optional
        Upper bound for the dimensionality of intermediates created by
        inner contractions if the contractions are nested, i.e.,
        the dimensionality of the result of contr2 and contr3 is restricted in
        "contr1(contr2(contr3(...)))".
    max_n_simultaneous_contracted: int | None, optional
        The maximum number of objects allowed to be contracted
        simultaneously in a single contraction. (default: None)
    space_dims: dict[str, int] | None, optional
        The sizes of the spaces (occ, virt, ...) used to estimate the cost of
        contractions. If not provided, the sizes from "config.json" will be
        used.
    """
    # - import (or extract) the target indices
    if target_indices is None:
        target_indices = term.target
    else:
        target_indices = tuple(get_symbols(target_indices, target_spin))
    # - import the space sizes/dims
    if isinstance(space_dims, dict):
        space_dims = Sizes.from_dict(space_dims)
    assert space_dims is None or isinstance(space_dims, Sizes)
    # - extract the relevant part (tensors and deltas) of the term
    relevant_obj_names: list[str] = []
    relevant_obj_indices: list[tuple[Index]] = []
    for obj in term.objects:
        base, exp = obj.base_and_exponent
        if obj.sympy.is_number:  # skip number prefactor
            continue
        elif exp < 0:
            raise NotImplementedError(f"Found object {obj} with exponent "
                                      f"{exp} < 0. Contractions not "
                                      "implemented for divisions.")
        elif isinstance(base, Symbol):  # skip symbolic prefactor
            continue
        elif not isinstance(base, (SymbolicTensor, KroneckerDelta)):
            raise NotImplementedError("Contractions can only be optimized for "
                                      "tensors and KroneckerDeltas.")
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
                           term_target_indices=target_indices)
    # lazily find the contraction schemes
    contraction_schemes = _optimize_contractions(
        relevant_obj_names=tuple(relevant_obj_names),
        relevant_obj_indices=tuple(relevant_obj_indices),
        target_indices=target_indices, max_itmd_dim=max_itmd_dim,
        max_n_simultaneous_contracted=max_n_simultaneous_contracted
    )
    # go through all schemes and find the one with the lowest scaling by
    # considering the:
    # 1) Maximum scaling of each field
    # 2) The number of contractions with the maximum scaling
    # Thereby, we rely on the order in which the fields are defined on the
    # dataclass. Furthermore, the computational scaling is prioritized over the
    # memory scaling.
    optimal_scaling = None
    optimal_scheme = None
    for scheme in contraction_schemes:
        # determine the costs for the current contraction scheme
        arithmetic = 0
        memory = 0
        for contr in scheme:
            nflops, mem = contr.evaluate_costs(space_dims)
            arithmetic += nflops
            memory += mem
        scaling = (arithmetic, memory)
        if optimal_scaling is None or scaling < optimal_scaling:
            optimal_scheme = scheme
            optimal_scaling = scaling
    # the generator is empty, i.e., we could not find any contraction scheme
    if optimal_scheme is None:
        raise RuntimeError("Could not find a valid contraction scheme for "
                           f"term {term} while restricting the maximum "
                           f"dimensionality of intermediates to "
                           f"{max_itmd_dim} and allowing simultaneous "
                           f"contractions of {max_n_simultaneous_contracted} "
                           "objects.")
    return optimal_scheme


def _optimize_contractions(relevant_obj_names: tuple[str],
                           relevant_obj_indices: tuple[tuple[Index]],
                           target_indices: tuple[Index],
                           max_itmd_dim: int | None = None,
                           max_n_simultaneous_contracted: int | None = None,
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
        obj_indices=relevant_obj_indices, target_indices=target_indices,
        max_group_size=max_n_simultaneous_contracted
    )

    for group in connected_groups:
        contr_indices = tuple(relevant_obj_indices[pos] for pos in group)
        contr_names = tuple(relevant_obj_names[pos] for pos in group)
        contraction = Contraction(indices=contr_indices, names=contr_names,
                                  term_target_indices=target_indices)
        # if the contraction is not an outer contraction we have to check
        # the dimensionality of the intermediate tensor
        if max_itmd_dim is not None and \
                contraction.target != target_indices and \
                len(contraction.target) > max_itmd_dim:
            continue
        # remove the contracted names and indices
        remaining_pos = [pos for pos in range(len(relevant_obj_names))
                         if pos not in group]
        remaining_names = (contraction.contraction_name,
                           *(relevant_obj_names[pos] for pos in remaining_pos))
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
            target_indices=target_indices, max_itmd_dim=max_itmd_dim,
            max_n_simultaneous_contracted=max_n_simultaneous_contracted
        )
        for contraction_scheme in completed_schemes:
            contraction_scheme.insert(0, contraction)
            yield contraction_scheme


def _group_objects(obj_indices: tuple[tuple[Index]],
                   target_indices: tuple[Index],
                   max_group_size: int | None = None) -> tuple[tuple[int]]:
    """
    Split the provided relevant objects into subgroups that share common
    contracted indices. Thereby, a group can at most contain 'max_group_size'
    objects. By default, all objects are allowed to be in one group.
    """
    # NOTE: the algorithm currently maximizes the number of contracted
    #       indices, i.e., a contraction runs over all common contracted
    #       indices. While this is fine in most cases, it might be benefitial
    #       to not contract all possible indices simultaneously
    #       in certain cases, since this leads to an increased group size:
    #             0    1    2
    #           d_ijk d_ij d_jl
    #       0 and 1 share i and j. A contraction running only over i can be
    #       performed for the pair (0, 1). However, if the contraction runs
    #       runs over i and j, we have to consider the triple (0, 1, 2).

    # sanity checks for input
    assert len(obj_indices) > 1  # we need at least 2 objects
    if max_group_size is None:
        max_group_size = len(obj_indices)
    assert max_group_size > 1  # group size has to be at least 2

    # track on which objects the indices appear
    idx_occurences: dict[Index, list[int]] = {}
    for pos, indices in enumerate(obj_indices):
        for idx in indices:
            if idx not in idx_occurences:
                idx_occurences[idx] = []
            idx_occurences[idx].append(pos)

    # store grouped objects and isolated objects (outer products)
    # for the groups we are using a dict, since it by default returns
    # keys in the order they were inserted. A set would need to be sorted
    # before returning to produce consistent results.
    groups: dict[tuple[int], None] = {}
    outer_products: list[tuple[int]] = []
    # iterate over all pairs of objects (index tuples)
    for (pos1, indices1), (pos2, indices2) in \
            itertools.combinations(enumerate(obj_indices), 2):
        # check if the objects have any common contracted indices
        # -> outer products can be treated as pair
        contracted, _ = Contraction._split_contracted_and_target(
            (indices1, indices2), target_indices
        )
        if not contracted:
            outer_products.append((pos1, pos2))
            continue
        # get all the objects any of the contracted indices appears
        positions = {pos for idx in contracted for pos in idx_occurences[idx]}
        # group too large
        if len(positions) > max_group_size:
            continue
        # avoid duplication: 0, 1 and 2 are connected by a common index
        # -> the pair 0,1 and 0,2 will both give the triple 0,1,2
        #    which will then grow in the same way independent of the starting
        #    pair.
        key = tuple(sorted(positions))
        if key in groups:
            continue
        # store the minimal group
        groups[key] = None

        # self-consistently update the contracted indices and the positions
        # This corresponds to maximizing the group size.
        # However, it is unclear if growing the group leads to a better
        # scaling contraction. Therefore, also store smaller groups
        while True:
            # update the contracted indices
            new_contracted, _ = Contraction._split_contracted_and_target(
                [obj_indices[pos] for pos in positions], target_indices
            )
            # no new contracted indices pulled in -> we are done
            if contracted == new_contracted:
                break
            # update the positions
            new_positions = {
                pos for idx in new_contracted for pos in idx_occurences[idx]
            }
            # no new positions or the extended group is too large
            if new_positions == positions or \
                    len(new_positions) > max_group_size:
                break
            # store the current group before trying to further
            # increase the size
            groups[tuple(sorted(new_positions))] = None
            contracted = new_contracted
            positions = new_positions
    return (*groups.keys(), *outer_products)


def unoptimized_contraction(term: Term, target_indices: str | None = None,
                            target_spin: str | None = None
                            ) -> list[Contraction]:
    """
    Determines the unoptimized contraction for the given term, i.e.,
    a simultaneous hyper-contraction of all tensors and deltas.

    Parameters
    ----------
    term: Term
        Build an unoptimized contraction for the given term.
    target_indices: str | None, optional
        The target indices of the term. If not given, the canonical target
        indices of the term according to the Einstein sum convention
        will be used.
    target_sin: str | None, optional
        The spin of the target indices, e.g., "aabb" for
        alpha, alpha, beta, beta. If not given, target indices without spin
        will be used.
    """
    # - import (or extract) the target indices
    if target_indices is None:
        target_indices = term.target
    else:
        target_indices = tuple(get_symbols(target_indices, target_spin))
    # extract the relevant part of the term
    relevant_obj_names: list[str] = []
    relevant_obj_indices: list[tuple[Index]] = []
    contracted = set()
    for obj in term.objects:
        base, exp = obj.base_and_exponent
        if obj.sympy.is_number:  # skip number prefactor
            continue
        elif exp < 0:
            raise NotImplementedError(f"Found object {obj} with exponent "
                                      f"{exp} < 0. Contractions not "
                                      "implemented for divisions.")
        elif isinstance(base, Symbol):  # skip symbolic prefactor
            continue
        elif not isinstance(base, (SymbolicTensor, KroneckerDelta)):
            raise NotImplementedError("Contractions only implemented for "
                                      "tensors and KroneckerDeltas.")
        name, indices = obj.longname(), obj.idx
        relevant_obj_names.extend(name for _ in range(exp))
        relevant_obj_indices.extend(indices for _ in range(exp))
        contracted.update(idx for idx in indices if idx not in target_indices)
    assert len(relevant_obj_indices) == len(relevant_obj_names)
    return [Contraction(indices=relevant_obj_indices, names=relevant_obj_names,
                        term_target_indices=target_indices)]
