from .expr_container import Expr, Term, Obj, Polynom, NormalOrdered
from .indices import Index, get_symbols, sort_idx_canonical
from .logger import logger
from .misc import Inputerror
from .sympy_objects import SymbolicTensor, KroneckerDelta
from .tensor_names import tensor_names, is_t_amplitude

from sympy.physics.secondquant import FermionicOperator
from sympy import S

import itertools


def apply_cvs_approximation(expr: Expr, core_indices: str,
                            spin: str | None = None) -> Expr:
    """
    Apply the core-valence approximation to the given expression by
    splitting the occupied space into core and valence space.
    Furthermore certain ERI/Coulomb blocks are assumed to vanish.

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
    # NOTE: Index substitutions have to be performed for all indices
    # simultaneously, to avoid creating an intermediate delta_cx that is then
    # further substituted to a delta_cc for instance. However, the delta_cx
    # will be evalauted to zero upon creation and therefore some terms might
    # vanish by accident.
    if not isinstance(expr, Expr):
        raise Inputerror("Expression needs to be provided as Expr instance.")
    core_indices = get_symbols(core_indices, spin)
    # ensure that the provided core indices are valid
    if not all(idx.space == "core" for idx in core_indices):
        raise Inputerror(f"The provided core indices {core_indices} are no "
                         "valid core indices, i.e., they do not belong to the"
                         " core space.")
    # get the target indices of the expression
    terms = expr.terms
    target_indices = terms[0].target
    assert all(term.target == target_indices for term in terms)
    # for each occupied target index build the corresponding core index
    occupied_target_indices = tuple(
        idx for idx in target_indices if idx.space == "occ"
    )
    occ_target_as_core = get_core_indices(occupied_target_indices)
    # build the substitution dict for the occupied target indices
    target_subs = {occ: core for occ, core in
                   zip(occupied_target_indices, occ_target_as_core)
                   if core in core_indices}

    result = Expr(0, **expr.assumptions)
    for term in terms:
        result += expand_contracted_indices(
            term, target_subs=target_subs,
            is_allowed_cvs_block=is_allowed_cvs_block
        )
    # update the set target indices if necessary
    if result.provided_target_idx is not None:
        result_target = tuple(target_subs.get(s, s) for s in target_indices)
        result.set_target_idx(result_target)
    return result


def expand_contracted_indices(term: Term, target_subs: dict[Index, Index],
                              is_allowed_cvs_block: callable = None) -> Expr:
    """
    Expands the contracted occupied indices in the given term into core
    and valence indices. Note that valence indices are denoted as occupied
    in the result.

    Parameters
    ----------
    term: Term
        Term in which to expand the occupied contracted indices
    target_subs: dict[Index, Index]
        The substitution dict containing the necessary occ -> core
        substitutions for the target indices. Will not be modified in this
        function!
    is_allowed_cvs_block : callable | None, optional
        Callable that takes an expr_container.Obj instance and a space string
        (e.g. 'covv'). It returns a bool indicating whether the block of the
        object described by the space string is valid within the CVS
        approximation, i.e., whether the block is neglected or not.
        If no callable is provided no blocks are neglected.
    """
    if not term.idx:  # term is a number -> nothing to do
        return Expr(term.sympy, **term.assumptions)

    if is_allowed_cvs_block is None:
        is_allowed_cvs_block = allow_all_cvs_blocks
    # get the contracted occupied indices
    # and build the corresponding core indices
    contracted = term.contracted
    occupied_contracted = tuple(
        idx for idx in contracted if idx.space == "occ"
    )
    core_contracted = get_core_indices(occupied_contracted)
    result = Expr(0, **term.assumptions)
    # go through all variants of valence and core indices
    for variant in itertools.product("oc", repeat=len(occupied_contracted)):
        # finish the substitution dict
        subs = target_subs.copy()
        for space, occ, core in \
                zip(variant, occupied_contracted, core_contracted):
            if space != "c":
                continue
            # check for contradictions in the full substitutions dict
            if occ in subs and subs[occ] is not core:
                raise RuntimeError("Found contradiction in substitution dict. "
                                   f"The occ index {occ} can not be mapped "
                                   f"onto {subs[occ]} and {core} at the "
                                   "same time.")
            subs[occ] = core
        # go through the objects and check if there is a block that is
        # neglected within the CVS approximation
        is_valid_variant = True
        for obj in term.objects:
            cvs_block = "".join(
                subs.get(idx, idx).space[0] for idx in obj.idx
            )
            if not is_allowed_cvs_block(obj, cvs_block):
                is_valid_variant = False
                break
        if not is_valid_variant:  # variant generates a neglected block
            continue
        # apply the substitutions to the term. This has to happen
        # simultaneously in order to avoid intermediates delta_cx which
        # evaluate to zero.
        sub_term = term.subs(subs, simultaneous=True)
        result += sub_term
    return result


def allow_all_cvs_blocks(obj: Obj, cvs_block: str) -> bool:
    return True


def is_allowed_cvs_block(obj: Obj, cvs_block: str) -> bool:
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
        return all(is_allowed_cvs_block(o, b)
                   for o, b in zip(obj.objects, cvs_block))

    sympy_obj = obj.base
    if isinstance(sympy_obj, SymbolicTensor):
        name = sympy_obj.name
        if name == tensor_names.eri:
            return is_allowed_cvs_eri_block(cvs_block)
        elif name == tensor_names.coulomb:
            return is_allowed_cvs_coulomb_block(cvs_block)
        elif is_t_amplitude(name):
            return is_allowed_cvs_t_amplitude_block(cvs_block)
        elif name == tensor_names.fock:
            return is_allowed_cvs_fock_block(cvs_block)
    elif isinstance(sympy_obj, KroneckerDelta):
        return is_allowed_cvs_delta_block(cvs_block)
    elif isinstance(sympy_obj, FermionicOperator):
        return True

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
    return cvs_block in itmd.allowed_cvs_blocks(is_allowed_cvs_block)


def is_allowed_cvs_coulomb_block(coulomb_block: str) -> bool:
    """
    Whether the given Coulomb integral (in chemist notation) block
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
    assert len(coulomb_block) == 4
    assert "g" not in coulomb_block  # no general indices
    if "c" in coulomb_block and (coulomb_block[:2].count("c") == 1 or
                                 coulomb_block[2:].count("c") == 1):
        return False
    return True


def is_allowed_cvs_eri_block(eri_block: str) -> bool:
    """
    Whether the given anti-symmetric ERI block (in physicist notation)
    is allowed within the CVS approximation.
    """
    assert len(eri_block) == 4
    assert "g" not in eri_block  # no general indices
    n_core = eri_block.count("c")
    if n_core == 1 or n_core == 3:
        return False
    # additionally, the blocks ccxx and xxcc are not allowed
    # (see comment in is_allowed_cvs_coulomb_block)
    elif n_core == 2 and (eri_block[:2] == "cc" or eri_block[2:] == "cc"):
        return False
    return True


def is_allowed_cvs_fock_block(fock_block: str) -> bool:
    """
    Whether the given Fock matrix block is allowed within the CVS
    approximation.
    """
    assert len(fock_block) == 2
    assert "g" not in fock_block  # no general indices
    if fock_block.count("c") == 1:  # f_cx / f_xc
        return False
    return True  # f_cc / f_xx


def is_allowed_cvs_t_amplitude_block(amplitude_block: str) -> bool:
    """
    Whether the given block of a ground state t-amplitude is valid within
    the CVS approximation
    """
    # t-amplitudes seem to follow the rule that only the valence space
    # has to be considered, i.e., all core orbitals can simply
    # be neglected.
    # t2_1: oovv   t1_2: ov   t2_2: oovv   t3_2: ooovvv   t4_2: oooovvvv
    assert not len(amplitude_block) % 2
    assert all(sp == "v" for sp in amplitude_block[len(amplitude_block)//2:])
    if amplitude_block.count("c"):
        return False
    assert all(sp == "o" for sp in amplitude_block[:len(amplitude_block)//2])
    return True


def is_allowed_cvs_delta_block(delta_block: str) -> bool:
    """
    Whether the given delta block is allowed within the CVS approximation.
    """
    assert len(delta_block) == 2
    assert "g" not in delta_block
    return delta_block[0] == delta_block[1]


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
        Callable that takes an expr_container.Obj instance and a space string
        (e.g. 'covv'). It returns a bool indicating whether the block of the
        object described by the space string is valid within the CVS
        approximation, i.e., whether the block is neglected or not.
        If no callable is provided no blocks are neglected.
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
            # build the target index occ -> core substitution dict
            target_subs = {occ: core for space, occ, core in
                           zip(variant, occupied_target, core_target)
                           if space == "c"}
            # expand the occupied contracted indices
            sub_term = expand_contracted_indices(
                term, target_subs=target_subs,
                is_allowed_cvs_block=is_allowed_cvs_block
            )
            # invalid substitutions -> invalid variant
            if sub_term.sympy is S.Zero:
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
