from collections import Counter
from itertools import product

from .expression import ExprContainer
from .logger import logger
from .misc import Inputerror
from .indices import (
    Index, get_symbols, order_substitutions, sort_idx_canonical,
    _is_str_sequence
)
from .simplify import simplify


def transform_to_spatial_orbitals(expr: ExprContainer, target_idx: str,
                                  target_spin: str,
                                  restricted: bool = False,
                                  expand_eri: bool = True) -> ExprContainer:
    """
    Transforms an expression to a spatial orbital basis by integrating over
    the spin of the spin orbitals, i.e., a spin is attached to all indices.
    Furthermore, the antisymmetric ERI's are replaced by the in this context
    more commonly used coulomb integrals in chemist notation.
    Target indices of the expression are updated if necessary.

    Parameters
    ----------
    expr : ExprContainer
        Expression to express in terms of spatial orbitals.
    target_idx : str
        The names of target indices of the expression. Needs to be provided,
        because the target indices in the expression are stored in canonical
        order, which might not be correct.
    target_spin : str
        The spin of the target indices, e.g., 'aa' for 2 alpha orbitals.
    restricted : bool, optional
        Whether a restricted reference (equal alpha and beta orbitals)
        should be assumed. In case of a restricted reference, only alpha
        orbitals will be present in the returned expression.
        (default: False)
    expand_eri : bool, optional
        If set, the antisymmetric ERI (in physicist notation) are expanded
        to coulomb integrals using chemist notation
        <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr),
        where by default a SymmetricTensor 'v' is used to represent the
        coulomb integrals.
    """

    # perform the integration first, since the intermediates are defined
    # in terms of the antisymmetric ERI
    expr = integrate_spin(expr, target_idx, target_spin)
    if expand_eri:
        expr.expand_antisym_eri().expand()
    if not restricted:
        return expr
    # in the restricted case we can replace all beta orbitals by the
    # corresponding alpha orbitals.
    # It should be fine to keep the name and only adjust the spin of the
    # indices:
    # - in the input expression we only have spin orbitals
    # - during the integration we generate multiple terms mapping each index
    #   to a spin
    #  -> the names are still unique, i.e., at this point each term might only
    #     hold an index of a certain name with either alpha or beta spin but
    #     not both of them simultaneously
    restricted_expr: ExprContainer = ExprContainer(0, **expr.assumptions)
    if expr.provided_target_idx is not None:
        # update the target indices
        restricted_target = get_symbols(target_idx, "a" * len(target_spin))
        restricted_expr.set_target_idx(restricted_target)
    for term in expr.terms:
        idx = set(term.idx)
        beta_idx = [i for i in idx if i.spin == "b"]
        if not beta_idx:
            restricted_expr += term.inner
            continue
        new_idx = get_symbols([i.name for i in beta_idx], "a"*len(beta_idx))
        sub: dict[Index, Index] = {}
        for old, new in zip(beta_idx, new_idx):
            # conststruct the alpha index
            if new in idx:
                raise RuntimeError("It is not safe to replace the beta index "
                                   f"{old} with the corresponding alpha index,"
                                   " because the index with alpha spin is "
                                   f"already used in the term: {term}.")
            sub[old] = new
        restricted_expr += term.inner.subs(order_substitutions(sub))
    assert isinstance(restricted_expr, ExprContainer)
    return restricted_expr


def integrate_spin(expr: ExprContainer, target_idx: str,
                   target_spin: str) -> ExprContainer:
    """
    Integrates over the spin of the spin orbitals to transform an expression
    to a spatial orbital basis, i.e, a spin is attached to all indices.
    Target indices in the expression will be updated if necessary.

    Parameters
    ----------
    expr : ExprContainer
        Expression where the spin is integrated.
    target_idx : str
        Names of target indices of the expression.
    target_spin : str
        Spin of target indices of the expression.
    """
    assert isinstance(expr, ExprContainer)
    # - validate the target indices and target spin
    target_symbols = get_symbols(target_idx)
    if len(target_symbols) != len(target_spin):
        raise Inputerror(f"Spin {target_spin} and indices {target_symbols} are"
                         " not compatible.")
    target_idx_spins: dict[Index, str] = {}
    for idx, spin in zip(target_symbols, target_spin):
        if idx in target_idx_spins and target_idx_spins[idx] != spin:
            raise ValueError(f"The index {idx} can not be assigned to alpha "
                             "and beta spin simultaneously.")
        target_idx_spins[idx] = spin
    # - sort the target indices to validate that the terms have the correct
    #   target indices and build the target spins
    sorted_target = tuple(sorted(target_idx_spins, key=sort_idx_canonical))
    target_spins = [target_idx_spins[idx] for idx in sorted_target]
    del target_idx_spins
    # - generate the new target indices of the resulting expression to set
    #   them if needed
    result_target = get_symbols([s.name for s in target_symbols], target_spin)

    result: ExprContainer = ExprContainer(0, **expr.assumptions)
    if expr.provided_target_idx is not None:
        result.set_target_idx(result_target)

    for term in expr.terms:
        logger.debug(f"Integrating spin in term {term}")
        # - ensure that the term has matching target indices
        term_target = term.target
        if term_target != sorted_target:
            raise ValueError(f"Target indices {term_target} of term {term} "
                             "don't match the desired target indices "
                             f"{target_symbols}")
        # - ensure that no index in the term is holding a spin
        if any(s.spin for s in term.idx):
            raise ValueError("The function assumes that the input expression "
                             "is expressed in terms of spin orbitals. Found "
                             f"a spatial orbital in term {term}.")
        # we have no indices (the term is a number) we don't have anything
        # to do
        if not term.idx:
            logger.debug(f"Result = {term}")
            result += term.inner
            continue
        # - build a list of indices and base map for the spins of the indices
        #   starting with the target indices
        term_contracted = term.contracted
        term_indices = (*term_target, *term_contracted)
        assert all(v == 1 for v in Counter(term_indices).values())
        base_spins: list[str | None] = [spin for spin in target_spins]
        base_spins.extend(None for _ in range(len(term_contracted)))
        # - for each object in the term: go through the allowed spin_blocks and
        #   try to add them to the base spins (target spins) in order to form a
        #   valid variants where all indices are assigned to a spin.
        spin_variants: list[list[str | None]] = [base_spins]
        term_vanishes: bool = False
        for obj in term.objects:
            allowed_blocks = obj.allowed_spin_blocks
            # hit a Polynom, Prefactor or unknown tensor
            if allowed_blocks is None:
                continue
            # we have some allowed blocks to check
            # -> try to form valid combinations assigning all indices to a spin
            indices: tuple[int, ...] = tuple(
                term_indices.index(idx) for idx in obj.idx
            )
            old_spin_variants = spin_variants.copy()
            spin_variants.clear()
            for block in allowed_blocks:
                # - ensure that the block is valid: a index can not be
                #   assigned to alpha and beta at the same time
                addition: list[str | None] = [
                    None for _ in range(len(term_indices))
                ]
                for spin, idx in zip(block, indices):
                    if addition[idx] is not None and addition[idx] != spin:
                        raise ValueError("Found invalid allowed spin block "
                                         f"{block} for {obj}.")
                    addition[idx] = spin
                # check for contracdictions with the target_spin and skip the
                # block if this is the case
                if any(sp1 != sp2 for sp1, sp2 in
                       zip(target_spins, addition[:len(term_target)])
                       if sp2 is not None):
                    continue
                # iterate over the existing variants and try to add the
                # addition
                for old_variant in old_spin_variants:
                    # check for any contradiction
                    if any(sp1 != sp2 for sp1, sp2 in
                           zip(old_variant, addition)
                           if sp1 is not None and sp2 is not None):
                        continue
                    # add the addition to the old variant
                    combination = [sp1 if sp2 is None else sp2
                                   for sp1, sp2 in zip(old_variant, addition)]
                    # we only need unique variants -> remove duplicates
                    if any(comb == combination for comb in spin_variants):
                        continue
                    spin_variants.append(combination)
            # we could not find a single valid combination for the given
            # object -> the term has to vanish
            if not spin_variants:
                term_vanishes = True
                break
        if term_vanishes:
            logger.debug("Result = 0")
            continue
        # collect the result in a separate expression such that we can call
        # simplify before adding the contribution to the result
        contribution: ExprContainer = ExprContainer(0, **expr.assumptions)
        if expr.provided_target_idx is not None:  # if necessary update target
            contribution.set_target_idx(result_target)
        # - iterate over the unique combinations, replace the spin orbitals
        #   by the corresponding spatial orbitals (assign a spin to the
        #   indices) and add the corresponding terms to the result.
        #   Thereby, ensure that all indices have a spin assigned and
        #   try to assign a spin for not yet assigned indices:
        #   since all variants are initialized with the target spins
        #   set, only contracted indices can not be assigned
        #  -> generate a variant for alpha and beta since both are allowed
        for spin_var in spin_variants:
            missing_contracted = [
                idx for idx, spin in enumerate(spin_var)
                if idx >= len(target_spin) and spin is None
            ]
            # construct variants for missing contracted indices assuming that
            # alpha and beta spin is allowed.
            if missing_contracted:
                variants: list[list[str | None]] = []
                for spins in product("ab", repeat=len(missing_contracted)):
                    complete_variant = spin_var.copy()
                    for spin, idx in zip(spins, missing_contracted):
                        complete_variant[idx] = spin
                    variants.append(complete_variant)
            else:
                variants: list[list[str | None]] = [spin_var]
            # go through the variants and perform the actual substitutions
            for variant in variants:
                # ensure that we indeed assigned all spins
                assert _is_str_sequence(variant)

                new_indices = get_symbols(
                    indices=[s.name for s in term_indices],
                    spins="".join(variant)
                )
                sub = {
                    old: new for old, new in zip(term_indices, new_indices)
                }
                contrib = term.inner.subs(order_substitutions(sub))
                logger.debug(f"Found contribution {contrib}")
                contribution += contrib
        # TODO: if we simplify the result it will throw an error for any
        # polynoms or denominators. Should we skip the simplification altough
        # we currently don't treat polynoms correctly in this function
        # since their allowed_spin_blocks are not considered.
        assert isinstance(contribution, ExprContainer)
        result += simplify(contribution)
    return result


def allowed_spin_blocks(expr: ExprContainer,
                        target_idx: str) -> tuple[str, ...]:
    """
    Determines the allowed spin blocks of an expression. Thereby, it is assumed
    that the allowed spin blocks of tensors in the expression are either known
    or can be determined on the fly, i.e., this only works for closed
    expressions.

    Parameters
    ----------
    expr : ExprContainer
        The expression to check.
    target_idx : str
        The target indices of the expression.
    """

    assert isinstance(expr, ExprContainer)

    target_symbols = get_symbols(target_idx)
    sorted_target = tuple(sorted(target_symbols, key=sort_idx_canonical))

    # - determine all possible spin blocks
    spin_blocks: list[str] = [
        "".join(b) for b in product("ab", repeat=len(target_symbols))
    ]
    spin_blocks_to_check: list[int] = [i for i in range(len(spin_blocks))]

    allowed_blocks: set[str] = set()
    for term in expr.terms:
        # - ensure that the term has matching target indices
        if term.target != sorted_target:
            raise ValueError(f"Target indices {term.target} of {term} dont "
                             "match the provided target indices "
                             f"{target_symbols}")
        # - extract the allowed blocks for all tensors and initialize
        #   index maps to relate indices to a spin
        term_idx_maps: list[tuple[list[dict[Index, str]], int]] = []
        for obj in term.objects:
            allowed_object_blocks = obj.allowed_spin_blocks
            # hit a Polynom, Prefactor or unknown tensor
            if allowed_object_blocks is None:
                continue
            obj_indices = obj.idx
            n_target = len([
                idx for idx in obj_indices if idx in target_symbols
            ])
            object_idx_maps: list[dict[Index, str]] = []
            for block in allowed_object_blocks:
                idx_map = {}
                for spin, idx in zip(block, obj_indices):
                    if idx in idx_map and idx_map[idx] != spin:
                        raise ValueError("Found invalid allowed spin block "
                                         f"{block} for {obj}.")
                    idx_map[idx] = spin
                object_idx_maps.append(idx_map)
            term_idx_maps.append((object_idx_maps, n_target))
        # - sort the allowed_tensor_blocks such that tensors with a high
        #   number of target indices are preferred
        term_idx_maps = sorted(term_idx_maps,
                               key=lambda tpl: tpl[1], reverse=True)

        term_indices = set(term.idx)
        blocks_to_remove: set[int] = set()
        for block_i in spin_blocks_to_check:
            block = spin_blocks[block_i]
            if block in allowed_blocks:
                blocks_to_remove.add(block_i)
                continue
            valid_block = True

            # - assign the target indices to a spin
            target_spin: dict[Index, str] = {}
            for spin, idx in zip(block, target_symbols):
                # in case we have target indices iiab only spin blocks
                # aaxx or bbxx are valid
                if idx in target_spin and target_spin[idx] != spin:
                    valid_block = False
                    break
                target_spin[idx] = spin
            if not valid_block:
                continue

            # - remove all object spin blocks that are in contradiction to the
            #   current spin block
            relevant_term_spin_idx_maps: list[list[dict[str, set[Index]]]] = []
            for tensor_idx_maps, _ in term_idx_maps:
                relevant_object_spin_idx_maps: list[dict[str, set[Index]]] = []
                for idx_map in tensor_idx_maps:
                    # are all target idx compatible with the block?
                    if any(spin != idx_map[t_idx]
                           for t_idx, spin in target_spin.items()
                           if t_idx in idx_map):
                        continue
                    spin_idx_map: dict[str, set[Index]] = {
                        "a": set(), "b": set()
                    }
                    for idx, spin in idx_map.items():
                        spin_idx_map[spin].add(idx)
                    relevant_object_spin_idx_maps.append(spin_idx_map)
                # the object has not a single allowed spin block that is
                # compatible to the currently probed block
                if not relevant_object_spin_idx_maps:
                    valid_block = False
                    break
                relevant_term_spin_idx_maps.append(
                    relevant_object_spin_idx_maps
                )
            # at least 1 object has no compatible allowed spin block
            # -> the current term can not contribute to the current block
            if not valid_block:
                continue

            # - try to find a valid combination of the remaining spin blocks
            spin_idx_map: dict[str, set[Index]] = {"a": set(), "b": set()}
            if not _has_valid_combination(relevant_term_spin_idx_maps, 0,
                                          spin_idx_map):
                continue
            # - verify the result:
            #       ensure that all indices are assigned
            #       the target indices have the desired spin
            #       there is no intersection between the different spins
            if spin_idx_map["a"] & spin_idx_map["b"]:
                raise RuntimeError("Indices are assigned to alpha and beta "
                                   f"simultaneously in term {term}: ",
                                   spin_idx_map)
            if term_indices ^ (spin_idx_map["a"] | spin_idx_map["b"]):
                raise RuntimeError("Not all indices were assigned to a spin: "
                                   f"{term_indices} -> {spin_idx_map}")
            if any(idx not in spin_idx_map[spin]
                   for idx, spin in target_spin.items()):
                raise RuntimeError("Target index has wrong spin. Desired: "
                                   f"{target_spin}. Found: {spin_idx_map}.")
            # everything should be fine!
            # also add the 'inverse' block to the allowed blocks
            allowed_blocks.add(block)
            allowed_blocks.add("".join("a" if spin == "b" else "b"
                                       for spin in block))
            blocks_to_remove.add(block_i)
        # blocks that have been found dont need to be checked again
        spin_blocks_to_check = [i for i in spin_blocks_to_check
                                if i not in blocks_to_remove]
    return tuple(sorted(allowed_blocks))


def _has_valid_combination(tensor_idx_maps: list[list[dict[str, set[Index]]]],
                           current_pos: int, variant: dict[str, set[Index]]
                           ) -> bool:
    """
    Tries to recursively assign all indices to a spin without introducing
    contradictions. Returns immediately when all indices could be assigned
    successfully.
    """

    for idx_map in tensor_idx_maps[current_pos]:
        # look for any contradictions
        if idx_map["a"] & variant["b"] or idx_map["b"] & variant["a"]:
            continue
        # compute the indices which are added to remove them later again
        # if necessary
        addition: dict[str, tuple[Index, ...]] = {
            "a": tuple(idx for idx in idx_map["a"] if idx not in variant["a"]),
            "b": tuple(idx for idx in idx_map["b"] if idx not in variant["b"])
        }
        variant["a"].update(idx_map["a"])
        variant["b"].update(idx_map["b"])
        if len(tensor_idx_maps) == current_pos + 1:  # we are done!!
            return True
        # recurse further and try to complete
        if _has_valid_combination(tensor_idx_maps, current_pos+1, variant):
            return True
        # could not complete -> revert the addition and continue looping
        variant["a"].difference_update(addition["a"])
        variant["b"].difference_update(addition["b"])
    # could not add anything to the variant
    return False
