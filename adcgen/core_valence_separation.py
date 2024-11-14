from .expr_container import Expr, Obj, Polynom, NormalOrdered
from .indices import (
    Index, get_symbols, order_substitutions, sort_idx_canonical
)
from .logger import logger
from .misc import Inputerror
from .sympy_objects import SymbolicTensor
from .tensor_names import tensor_names

from sympy.physics.secondquant import FermionicOperator
from sympy import S

import itertools


def apply_cvs_approximation(expr: Expr, core_indices: str,
                            spin: str | None = None) -> Expr:
    """
    Apply the core-valence approximation to the given expression by
    splitting the occupied space into core and valence space.
    Furthermore certain ERI blocks are assumed to vanish.

    Parameters
    ----------
    expr: Expr
        Expression the CVS approximation should be applied to.
    core_indices: str
        The names of the core target indices to introduce assuming we currently
        have occupied target indices with matching names in the expression,
        e.g., "IJ" will transform the occupied target indices "ij" to the core
        target indices "IJ".
    spin: str | None, optional
        The spin of the core indices, e.g., "aa" for two core indices with
        alpha spin.
    """
    if not isinstance(expr, Expr):
        raise Inputerror("Expression needs to be provided as Expr instance.")
    expr = introduce_core_target_indices(
        expr, core_target_indices=core_indices, spin=spin
    )
    return expand_occupied_indices(expr, is_allowed_cvs_block)


def introduce_core_target_indices(expr: Expr, core_target_indices: str,
                                  spin: str | None = None):
    """
    Replaces certain occupied target indices in the expression by the
    corresponding core indices.

    Parameters
    ----------
    Expr : Expr
        The expression where to introduce the core indices
    core_target_indices : str
        The names of the core target indices to introduce assuming we currently
        have occupied target indices with matching names in the expression,
        e.g., "IJ" will transform the occupied target indices "ij" to the core
        target indices "IJ".
    spin: str | None, optional
        The spin of the core indices, e.g., "aa" for two core indices with
        alpha spin.
    """
    if not isinstance(expr, Expr):
        raise Inputerror("Expression needs to be provided as Expr instance.")
    if not core_target_indices:  # nothing to do
        return expr
    # ensure that the provided core indices are valid core indices
    core_indices: list[Index] = get_symbols(core_target_indices, spin)
    if not all(idx.space == "core" for idx in core_indices):
        raise Inputerror(f"The provided core indices {core_indices} are no "
                         "valid core indices, i.e., they do not belong to the"
                         " core space.")
    # for each core index build the corresponding occupied index
    occ_indices = get_occ_indices(core_indices)
    # get the target indices of the expression
    terms = expr.terms
    target_indices = terms[0].target
    assert all(term.target == target_indices for term in terms)
    # try to find the occupied target index for each provided core index
    subs: dict[Index, Index] = {}
    for core_idx, occ_idx in zip(core_indices, occ_indices):
        found = False
        for target_idx in target_indices:
            if occ_idx is target_idx:
                found = True
                subs[target_idx] = core_idx
                break
        if not found:
            raise ValueError("Could not find a matching occupied target index "
                             f"for the core index {core_idx}. Looked for "
                             f"{occ_idx} in {target_indices}.")
    # apply the target index substitutions
    expr = expr.subs(order_substitutions(subs))
    # and update the provided target indices
    if expr.provided_target_idx is not None:
        provided_target = [
            subs.get(idx, idx) for idx in expr.provided_target_idx
        ]
        expr.set_target_idx(provided_target)
    return expr


def expand_occupied_indices(expr: Expr,
                            is_allowed_cvs_block: callable = None
                            ) -> Expr:
    """
    Expands the contracted occupied indices into a core and valence index,
    where the valence index will be denoted as occupied index in the result.

    Parameters
    ----------
    expr : Expr
        The expression in which to expand the contracted occupied indices.
    is_allowed_cvs_block : callable | None, optional
        Callable that takes an expr_container.Obj instance and returns a bool
        indicating whether the object is valid within the CVS approximation,
        i.e., whether the tensor block does vanish or not.
        If no callable is provided no blocks are assumed to vanish.
    """
    if is_allowed_cvs_block is None:
        is_allowed_cvs_block = allow_all_cvs_blocks

    result = Expr(0, **expr.assumptions)
    for term in expr.terms:
        # - we have a number -> no indices -> nothing to do
        if not term.idx:
            result += term
        # get the contracted indices and
        # ensure that we have no contracted core indices
        contracted = term.contracted
        if any(idx.space == "core" for idx in contracted):
            raise ValueError(f"Found a contracted core index in term {term}. "
                             "Can not safely expand the occupied contracted "
                             "indices.")
        # get the occupied contracted indices
        # and build core indices for all occupied indices
        occupied_indices = [idx for idx in contracted if idx.space == "occ"]
        core_indices = get_core_indices(occupied_indices)
        # build all combinations of the core and valence (occ) space
        expansion_variants = itertools.product(
            ["occ", "core"], repeat=len(occupied_indices)
        )
        for variant in expansion_variants:
            # for ech combination/variant build a substitution dict
            constracted_subs: dict[Index, Index] = {}
            for space, occ_idx, core_idx in \
                    zip(variant, occupied_indices, core_indices):
                if space != "core":
                    continue
                constracted_subs[occ_idx] = core_idx
            # apply the substituions to the term
            if constracted_subs:
                sub_term = term.subs(
                    order_substitutions(constracted_subs)
                ).terms[0]
                # maybe we created delta_co -> invalid variant
                if sub_term.sympy is S.Zero:
                    continue
            else:
                sub_term = term
            # go through the objects and check whether they are valid within
            # the CVS approximation
            if all(is_allowed_cvs_block(obj) for obj in sub_term.objects):
                result += sub_term
    return result


def allow_all_cvs_blocks(obj: Obj) -> bool:
    return True


def is_allowed_cvs_block(obj: Obj) -> bool:
    """
    Whether the object is allowed within the CVS approximation.
    """
    from .intermediates import Intermediates, RegisteredIntermediate
    if not obj.idx:  # prefactor or symbol
        return True
    # skip Polynoms for now.
    # The MP orbital energy denoms should not be important
    if isinstance(obj, Polynom):
        return True
    elif isinstance(obj, NormalOrdered):
        return all(is_allowed_cvs_block(o) for o in obj.objects)

    sympy_obj = obj.base
    if isinstance(sympy_obj, SymbolicTensor):
        name = sympy_obj.name
        if name == tensor_names.eri:
            return is_allowed_cvs_eri_block(obj)
        elif name == tensor_names.coulomb:
            return is_allowed_cvs_coulomb_block(obj)
        # TODO: For t-amplitudes we seem to only have the ov/oovv/ooovvv/...
        # blocks as allowed blocks.
        # -> all core orbitals can be ignored
    elif isinstance(sympy_obj, FermionicOperator):
        return True
    # deltas should be handled automatically within the class

    # check if the obj is a known intermediate
    itmd: RegisteredIntermediate = (
        Intermediates().available.get(obj.longname(True), None)
    )
    if itmd is None:
        # the object is no intermediate
        # assume that all blocks are valid in this case
        logger.warning(
            f"Could not determine whether {obj} is valid within the CVS "
            "approximation."
        )
        return True
    # the object is a known intermediate:
    # expand the intermediate, and determine the allowed spin blocks
    return obj.space in itmd.allowed_cvs_blocks(is_allowed_cvs_block)


def is_allowed_cvs_coulomb_block(coulomb: Obj) -> bool:
    """
    Whether the given Coulomb integral (in chemist notation)
    is allowed within the CVS approximation
    """
    # NOTE: according to 10.1063/1.453424 (from 1987) coulomb integrals with
    # 1 and 3 core indices vanish. Furthermore, the Coulomb integrals
    # <cc|oo>, <cc|vv>, <oo|cc>, <vv|cc>
    # vanish, i.e., all integrals co/cv vanish.
    # However, in a later paper 10.1063/1.1418437 (from 2001) the integrals
    # <cc|oo>, <cc|vv>, <oo|cc>, <vv|cc>
    # = (co|co), (cv|cv), (oc|oc), (vc|vc)
    # only vanish when arising from different core-level occupations (DCO),
    # i.e., when they appear in matrix blocks that we are neglecting anyway.
    # In the current implementation in adcman/adcc those blocks are assumed
    # to vanish following the earlier paper.
    # The current implementation follows the implementation in adcman/adcc.
    block = coulomb.space
    assert len(block) == 4
    assert not block.count("g")  # no general indices
    if "c" in block and \
            (block[:2].count("c") == 1 or block[2:].count("c") == 1):
        return False
    return True


def is_allowed_cvs_eri_block(eri: Obj) -> bool:
    """
    Whether the given anti-symmetric ERI block (in physicist notation)
    is allowed within the CVS approximation.
    """
    block = eri.space
    assert len(block) == 4
    assert not block.count("g")  # no general indices
    n_core = block.count("c")
    if n_core == 1 or n_core == 3:
        return False
    # additionally, the blocks ccxx and xxcc are not allowed
    # (see comment in is_allowed_cvs_coulomb_block)
    elif n_core == 2 and (block[:2] == "cc" or block[2:] == "cc"):
        return False
    return True


def allowed_cvs_blocks(expr: Expr, target_idx: str, spin: str | None = None,
                       is_allowed_cvs_block: callable = None) -> tuple[str]:
    """
    Determines all allowed blocks for the given expression
    within the CVS approximation by expanding the occupied indices into
    core and valence indices.

    Parameters
    ----------
    expr: Expr
        The expression in which the allowed cvs blocks should be determined.
    target_idx: str
        The target indices of the expression.
    is_allowed_cvs_block : callable | None, optional
        Callable that takes an expr_container.Obj instance and returns a bool
        indicating whether the object is valid within the CVS approximation,
        i.e., whether the tensor block does vanish or not.
        If no callable is provided no blocks are assumed to vanish.
    """
    if is_allowed_cvs_block is None:
        is_allowed_cvs_block = allow_all_cvs_blocks

    target_idx: list[Index] = get_symbols(target_idx, spin)
    sorted_target = tuple(sorted(target_idx, key=sort_idx_canonical))
    # identify all occupied target indices
    # and build the corresponding core indices
    occupied_target = [idx for idx in target_idx if idx.space == "occ"]
    core_target = get_core_indices(occupied_target)
    # determine the possible cvs variants (part of the block string)
    cvs_variants = tuple(itertools.product("oc", repeat=len(occupied_target)))
    cvs_variants_to_check = [i for i in range(len(cvs_variants))]
    allowed_blocks = []
    # go through all terms and check each for the invalid cvs blocks
    for term in expr.terms:
        if term.target != sorted_target:
            raise ValueError(f"Target indices {term.target} of {term} dont "
                             f"match the provided target indices {target_idx}")

        variants_to_remove = set()
        for variant_i in cvs_variants_to_check:
            variant = cvs_variants[variant_i]
            # get the core indices for the current variant
            # and substitute the target indices in the term
            variant_core_indices = [
                core for core, sp in zip(core_target, variant)
                if sp == "c"
            ]
            sub_term = Expr(term.sympy, **term.assumptions)
            sub_term = introduce_core_target_indices(
                sub_term, variant_core_indices
            )
            # invalid substitutions -> invalid variant
            if sub_term.sympy is S.Zero:
                continue
            # update the asusmptions if necessary
            # expand the occupied contracted indices and check if we
            # get some contribution
            sub_term = expand_occupied_indices(
                sub_term, is_allowed_cvs_block=is_allowed_cvs_block
            )
            if sub_term.sympy is S.Zero:  # no valid contribution
                continue
            # build the full block string
            variant = list(reversed(variant))
            block = "".join(
                idx.space[0] if idx.space != "occ" else variant.pop()
                for idx in target_idx
            )
            assert not variant
            allowed_blocks.append(block)
            variants_to_remove.add(variant_i)
        cvs_variants_to_check = [i for i in cvs_variants_to_check
                                 if i not in variants_to_remove]
    return tuple(allowed_blocks)


def get_core_indices(occupied_indices: list[Index]) -> list[Index]:
    """
    Builds core indices for the given occupied indices, i.e.,
    I for the occupied index i.
    """
    assert all(idx.space == "occ" for idx in occupied_indices)
    names = []
    spins = []
    for idx in occupied_indices:
        names.append(idx.name.upper())
        spins.append(idx.spin)
    return get_symbols(names, spins)


def get_occ_indices(core_indices: list[Index]) -> list[Index]:
    """
    Builds the occupied/valence indices for the given core indices, i.e.,
    i for the core index I.
    """
    assert all(idx.space == "core" for idx in core_indices)
    names = []
    spins = []
    for idx in core_indices:
        names.append(idx.name.lower())
        spins.append(idx.spin)
    return get_symbols(names, spins)
