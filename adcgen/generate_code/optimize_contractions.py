from collections.abc import Sequence
from typing import Generator
import itertools

from sympy import Symbol, S

from ..expression import TermContainer
from ..indices import get_symbols, Index, sort_idx_canonical
from ..sympy_objects import SymbolicTensor, KroneckerDelta
from .contraction import Contraction, Sizes


def optimize_contractions(term: TermContainer,
                          target_indices: str | None = None,
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
    term: TermContainer
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
        target_symbols = term.target
    else:
        target_symbols = tuple(get_symbols(target_indices, target_spin))
    # - import the space sizes/dims
    if isinstance(space_dims, dict):
        space_sizes = Sizes.from_dict(space_dims)
    else:
        assert space_dims is None
        space_sizes = None
    # - extract the relevant part (tensors and deltas) of the term
    relevant_obj_names: list[str] = []
    relevant_obj_indices: list[tuple[Index, ...]] = []
    for obj in term.objects:
        base, exp = obj.base_and_exponent
        if obj.inner.is_number:  # skip number prefactor
            continue
        elif exp < S.Zero:
            raise NotImplementedError(f"Found object {obj} with exponent "
                                      f"{exp} < 0. Contractions not "
                                      "implemented for divisions.")
        elif isinstance(base, Symbol):  # skip symbolic prefactor
            continue
        elif not isinstance(base, (SymbolicTensor, KroneckerDelta)):
            raise NotImplementedError("Contractions can only be optimized for "
                                      "tensors and KroneckerDeltas.")
        name, indices = obj.longname(), obj.idx
        assert name is not None
        assert exp.is_Integer
        relevant_obj_names.extend(name for _ in range(int(exp)))
        relevant_obj_indices.extend(indices for _ in range(int(exp)))
    assert len(relevant_obj_names) == len(relevant_obj_indices)

    if not relevant_obj_names:  # no tensors or deltas in the term
        return []
    elif len(relevant_obj_names) == 1:
        # trivial: only a single tensor/delta with exponent 1
        # - resorting of indices
        # - trace
        return [Contraction(
            indices=relevant_obj_indices, names=relevant_obj_names,
            term_target_indices=target_symbols
        )]
    # lazily find the contraction schemes
    contraction_schemes = _optimize_contractions(
        relevant_obj_names=tuple(relevant_obj_names),
        relevant_obj_indices=tuple(relevant_obj_indices),
        target_indices=target_symbols, max_itmd_dim=max_itmd_dim,
        max_n_simultaneous_contracted=max_n_simultaneous_contracted
    )
    # go through all schemes and find the one with the lowest scaling by
    # considering the:
    # 1) Computational scaling (flop count)
    # 2) Memory scaling (number of elements to store)
    optimal_scaling = None
    optimal_scheme = None
    for scheme in contraction_schemes:
        # determine the costs for the current contraction scheme
        arithmetic = 0
        memory = 0
        for contr in scheme:
            nflops, mem = contr.evaluate_costs(space_sizes)
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


def _optimize_contractions(relevant_obj_names: Sequence[str],
                           relevant_obj_indices: Sequence[tuple[Index, ...]],
                           target_indices: Sequence[Index],
                           max_itmd_dim: int | None = None,
                           max_n_simultaneous_contracted: int | None = None,
                           ) -> Generator[list[Contraction], None, None]:
    """
    Find the optimal contractions for the given relevant objects of a term.

    Parameters
    ----------
    relevant_obj_names: Sequence[str]
        The names of the objects to consider.
    relevant_obj_indices: Sequence[tuple[Index, ...]]
        The indices of the objects to consider.
    target_indices: Sequence[Index]
        The target indices of the term.
    max_itmd_dim: int, optional
        The maximum allowed dimensionality of temporary intermediate results.
        If not given, the dimensionality is not restricted.
    max_n_simultaneous_contracted: int, optional
        The maximum number of tensors allowed in a single contraction, e.g.,
        2 to prevent any hyper-contractions.
    """
    assert len(relevant_obj_indices) == len(relevant_obj_names)
    if len(relevant_obj_names) < 2:
        raise ValueError("Need at least 2 objects to define a contraction.")

    # split the relevant objects into subgroups that share contracted indices
    connected_groups = _group_objects(
        obj_indices=relevant_obj_indices, target_indices=target_indices,
        max_group_size=max_n_simultaneous_contracted
    )
    for group, contracted in connected_groups:
        # build a contraction for the objects and the contracted indices
        indices = [relevant_obj_indices[pos] for pos in group]
        names = [relevant_obj_names[pos] for pos in group]
        contraction = Contraction(
            indices=indices, names=names, term_target_indices=target_indices,
            contracted=contracted
        )
        # if the contraction is not an outer contraction we have to check
        # the dimensionality of the intermediate tensor
        if max_itmd_dim is not None and \
                contraction.target != target_indices and \
                len(contraction.target) > max_itmd_dim:
            continue
        # update the data excluding the just contracted objects
        # and adding the contraction to the pool
        remaining_pos = [pos for pos in range(len(relevant_obj_names))
                         if pos not in group]
        remaining_names = (
            contraction.contraction_name,
            *(relevant_obj_names[pos] for pos in remaining_pos)
        )
        remaining_indices = (
            contraction.target,
            *(relevant_obj_indices[pos] for pos in remaining_pos)
        )
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
            # ensure that the contracted indices don't appear in any later
            # contraction again
            assert not any(
                s in idx for s in contraction.contracted
                for c in contraction_scheme for idx in c.indices
            )
            contraction_scheme.insert(0, contraction)
            yield contraction_scheme


def _group_objects(
        obj_indices: Sequence[tuple[Index, ...]],
        target_indices: Sequence[Index],
        max_group_size: int | None = None
        ) -> Generator[tuple[tuple[int, ...], tuple[Index, ...]], None, None]:
    """
    Split the provided relevant objects defined by their indices
    (``obj_indices``) into subgroups that share common contracted indices.
    Thereby, a group can at most contain ``max_group_size``
    objects and produce a result with ``max_result_dim`` dimensions.
    By default, all objects are allowed to be in one group and arbitrary
    result dimensionalities are allowed.
    """
    # sanity checks for input
    assert len(obj_indices) > 1  # we need at least 2 objects
    if max_group_size is None:
        max_group_size = len(obj_indices)
    assert max_group_size > 1  # group size has to be at least 2

    # track on which objects the indices appear
    idx_occurences: dict[Index, list[int]] = {}
    for pos, idx in enumerate(obj_indices):
        for s in idx:
            if s not in idx_occurences:
                idx_occurences[s] = []
            if s not in idx_occurences[s]:
                idx_occurences[s].append(pos)
        del idx

    # cache already encountered valid groups
    # excluding outer products since they can not appear twice
    seen_groups: set[tuple[tuple[int, ...], tuple[Index, ...]]] = set()
    # iterate over all pairs of objects (index tuples)
    for (pos1, indices1), (pos2, indices2) in \
            itertools.combinations(enumerate(obj_indices), 2):
        # check if the objects have any common contracted indices
        # -> outer products can be treated as pair
        contracted, _ = Contraction._split_contracted_and_target(
            indices=(indices1, indices2), term_target_indices=target_indices
        )
        if not contracted:  # outer product
            yield ((pos1, pos2), tuple())
            continue
        contracted = sorted(contracted, key=sort_idx_canonical)
        # Starting from the given pair try to explore all sensible
        # combinations of contracted indices possibly increasing
        # the group size (also exploring hyper-contractions)
        groups = _explore_group(
            seen_groups=seen_groups, obj_indices=obj_indices,
            target_indices=target_indices, max_group_size=max_group_size,
            idx_occurences=idx_occurences, contracted=contracted,
            positions=(pos1, pos2)
        )
        for group in groups:
            yield group
    return None


def _explore_group(
        seen_groups: set[tuple[tuple[int, ...], tuple[Index, ...]]],
        obj_indices: Sequence[tuple[Index, ...]],
        target_indices: Sequence[Index],
        max_group_size: int,
        idx_occurences: dict[Index, list[int]],
        contracted: Sequence[Index],
        positions: Sequence[int],
        forbidden_contracted: tuple[Index, ...] = tuple()
        ) -> Generator[tuple[tuple[int, ...], tuple[Index, ...]], None, None]:
    """
    Recursively explores the group by expanding the number of
    contracted indices and the group size.

    Parameters
    ----------
    seen_groups: set[tuple[tuple[int, ...], tuple[Index, ...]]]
        Cache to store already encountered groups to avoid duplications.
    obj_indices: Sequence[tuple[Index, ...]]
        The indices of all objects.
    target_indices: Sequence[Index]
        The target indices of the term the objects describe.
    max_group_size: int
        Upper limit for the allowed size of groups to consider.
    idx_occurences: dict[Index, list[int]]
        Map to connect an index to the objects (by position) it appears on
    contracted: Sequence[Inde]
        The common contracted indices the objects at ``positions`` share.
    positions: Sequence[int]
        The positions defining the group to further explore and expand.
    forbidden_contracted: tuple[Index, ...], optional
        Indices that are not allowed to be considered as contracted indices
        during the exploration of the given group.
    """
    # Iterate over all sensible subsets of contracted indices.
    # For instance 2 objects might share the indices ijkl.
    # However, k and l appear on 2 distinct other objects.
    # Therefore, we should always contract over ij but the
    # contraction over k and l should be optional since the
    # group size has to grow for those contractions
    contracted_variants = _contracted_variants(
        contracted, positions, idx_occurences
    )
    for contracted_indices in contracted_variants:
        # update the positions including all objects that hold any of the
        # contracted indices while checking the groups size
        new_positions = tuple(sorted({
            pos for idx in contracted_indices
            for pos in idx_occurences[idx]
        }))
        if len(new_positions) > max_group_size:
            continue
        # - try to update the contracted indices covering all indices
        # that only appear on tensors already in the group.
        # It does not make any sense to not contract over any of them
        # since we can safely do so using the current group
        # -> contracted_indices has to be a subset of new_contracted
        indices = tuple(obj_indices[p] for p in new_positions)
        new_contracted, _ = Contraction._split_contracted_and_target(
            indices=indices, term_target_indices=target_indices
        )
        new_contracted = [
            idx for idx in new_contracted
            if all(pos in new_positions for pos in idx_occurences[idx])
        ]
        assert all(s in new_contracted for s in contracted_indices)
        # - however, if any of the safely contractable indices is marked
        # as forbidden, we have to skip to avoid duplications
        # since the combination will then be explored later
        # -> new_contracted can not contain forbidden indices
        if any(idx in forbidden_contracted for idx in new_contracted):
            continue
        new_contracted = tuple(sorted(new_contracted, key=sort_idx_canonical))
        # - avoid duplications. For instance:
        # 0, 1 and 2 are connected by a common index
        # -> the pair 0,1 and 0,2 will both give the triple 0,1,2
        # which will then grow in the same way independent of the starting
        # pair.
        if (new_positions, new_contracted) in seen_groups:
            continue
        # - current group is not a duplicate and can safely be returned while
        # marking the group as explored.
        seen_groups.add((new_positions, new_contracted))
        yield (new_positions, new_contracted)
        # - To prevent duplications we don't want to contract over indices
        # that will be covered in another iteration of contracted_variants.
        # Also we don't want to mark any safely contractable indices
        # as forbidden to avoid exploring stupid groups.
        # -> mark missing optionaly contracted indices as forbidden
        # (all indices in new_contracted not forbidden and safely contractable
        # for the current group)
        new_forbidden_contracted = forbidden_contracted + tuple(
            s for s in contracted if s not in new_contracted
        )
        # - See if there are any other contracted indices that repeat
        # on new_positions that are not forbidden (will be explored later).
        # -> new_contracted has logically to be a subset of
        # available_contracted, since it is not possible for any index in
        # new_contracted to appear in new_forbidden_contracted!!
        available_contracted, _ = Contraction._split_contracted_and_target(
            indices=indices, term_target_indices=target_indices
        )
        available_contracted = [
            idx for idx in available_contracted
            if idx not in new_forbidden_contracted
        ]
        available_contracted = sorted(
            available_contracted, key=sort_idx_canonical
        )
        assert all(s in available_contracted for s in new_contracted)
        if len(available_contracted) > len(new_contracted):
            child_groups = _explore_group(
                seen_groups=seen_groups, obj_indices=obj_indices,
                target_indices=target_indices, max_group_size=max_group_size,
                idx_occurences=idx_occurences, contracted=available_contracted,
                positions=new_positions,
                forbidden_contracted=new_forbidden_contracted
            )
            for group in child_groups:
                yield group


def _contracted_variants(contracted: Sequence[Index],
                         positions: Sequence[int],
                         idx_occurences: dict[Index, list[int]]
                         ) -> Generator[tuple[Index, ...], None, None]:
    """
    Generates all sensible subsets of contracted indices for the
    given ``contracted`` indices generated by a contraction of objects
    at ``positions``. Thereby, a map connecting an index
    to the objects (by position) they appear on is required
    (``idx_occurences``) to avoid bad subsets.
    """
    # we can always safely contract over indices that only appear on the
    # already included positions (not contracting any of those would be
    # stupid since the scaling remains the same but the memory scaling
    # would increase)
    safe_contracted = []  # those should always be contracted
    optional_contracted = []  # contracting those will grow the group
    for idx in contracted:
        if all(pos in positions for pos in idx_occurences[idx]):
            safe_contracted.append(idx)
        else:
            optional_contracted.append(idx)
    safe_contracted = tuple(safe_contracted)
    if safe_contracted:
        yield safe_contracted

    if not optional_contracted:
        return
    # try to form all possible combinations for the optional contracted indices
    combinations = itertools.chain.from_iterable(
        itertools.combinations(optional_contracted, n)
        for n in range(1, len(optional_contracted) + 1)
    )
    for addition in combinations:
        yield safe_contracted + addition


def unoptimized_contraction(term: TermContainer,
                            target_indices: str | None = None,
                            target_spin: str | None = None
                            ) -> list[Contraction]:
    """
    Determines the unoptimized contraction for the given term, i.e.,
    a simultaneous hyper-contraction of all tensors and deltas.

    Parameters
    ----------
    term: TermContainer
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
        target_symbols = term.target
    else:
        target_symbols = tuple(get_symbols(target_indices, target_spin))
    # extract the relevant part of the term
    relevant_obj_names: list[str] = []
    relevant_obj_indices: list[tuple[Index, ...]] = []
    for obj in term.objects:
        base, exp = obj.base_and_exponent
        if obj.inner.is_number:  # skip number prefactor
            continue
        elif exp < S.Zero:
            raise NotImplementedError(f"Found object {obj} with exponent "
                                      f"{exp} < 0. Contractions not "
                                      "implemented for divisions.")
        elif isinstance(base, Symbol):  # skip symbolic prefactor
            continue
        elif not isinstance(base, (SymbolicTensor, KroneckerDelta)):
            raise NotImplementedError("Contractions only implemented for "
                                      "tensors and KroneckerDeltas.")
        name, indices = obj.longname(), obj.idx
        assert name is not None
        assert exp.is_Integer
        relevant_obj_names.extend(name for _ in range(int(exp)))
        relevant_obj_indices.extend(indices for _ in range(int(exp)))
    assert len(relevant_obj_indices) == len(relevant_obj_names)
    return [Contraction(indices=relevant_obj_indices, names=relevant_obj_names,
                        term_target_indices=target_symbols)]
