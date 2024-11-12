from .expr_container import Expr, Obj
from .indices import Index, get_symbols, order_substitutions
from .misc import Inputerror
from .sympy_objects import SymbolicTensor, KroneckerDelta
from .tensor_names import tensor_names

from sympy.physics.secondquant import FermionicOperator


def apply_cvs_approximation(expr: Expr, core_indices: str,
                            spin: str | None = None) -> Expr:
    """
    Apply the core-valence approximation to the given expression by
    splitting the occupied space into core and valence space.
    Furthermore certain ERI blocks are assumed to vanish.

    Parameters
    ----------
    expr : Expr
        Expression the CVS approximation should be applied to.
    core_indices : str
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
    # ensure that the provided core indices are valid core indices
    core_indices: list[Index] = get_symbols(core_target_indices, spin)
    if not all(idx.space == "core" for idx in core_indices):
        raise Inputerror(f"The provided core indices {core_indices} are no "
                         "valid core indices, i.e., they do not belong to the"
                         " core space.")
    # for each core index build the corresponding occupied index
    occ_indices = get_symbols([s.name.lower() for s in core_indices],
                              [s.spin for s in core_indices])
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


def allowed_cvs_blocks(object: Obj) -> tuple[str] | None:
    # prefactor or symbol have no indices -> no allowed cvs blocks
    if not object.idx:
        return None

    sympy_obj = object.base
    # antisym-, sym-, nonsymtensor and amplitude
    if isinstance(sympy_obj, SymbolicTensor):
        name = sympy_obj.name
        if name == tensor_names.eri:
            return allowed_cvs_eri_blocks(object)
    elif isinstance(sympy_obj, KroneckerDelta):
        pass
    elif isinstance(sympy_obj, FermionicOperator):
        pass
    # we might have some intermediate for which we can determine the allowed
    # blocks on the fly
    raise NotImplementedError()


def allowed_cvs_eri_blocks(eri: Obj) -> tuple[str] | None:
    # NOTE: according to 10.1063/1.1418437
    # the ERI blocks: ccoo, oocc, ccvv, vvcc
    # only vanish if they occur in a block that couples configurations with
    # different core level occupations (DCO), which should be neglected
    # automatically by only considering non-DCO blocks. Therefore, the 4
    # ERI blocks in principle have to be considered.
    # However, in an earlier paper (10.1063/1.453424) those blocks were also
    # neglected and in the current CVS implementation in adcman/adcc those
    # blocks are neglected.
    # For now those blocks will be neglected!
    pass
