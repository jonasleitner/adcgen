from .expression import ExprContainer
from .tensor_names import tensor_names
from .misc import Inputerror
from sympy import Symbol


def apply_resolution_of_identity(expr: ExprContainer,
                                 symmetric: bool = True) -> ExprContainer:
    """
    Applies the Resolution of Identity approximation (RI, sometimes also
    called density fitting, DF) to an expression. This implies that every
    spatial ERI is replaced by its factorised form. Two types of factorisation
    are supported: symmetric and asymmetric. In the symmetric decomposition,
    a spatial ERI is approximated as:

    (pq | rs) ~ B^P_{pq} B^P_{rs}
    B^P_{pq} = (P | Q)^{-1/2} (Q | pq)

    This decomposition is the default. In the asymmetric factorisation, the
    same spatial ERI is approximated as:

    (pq | rs) ~ C^P_{pq} (P | rs)
    C^P_{pq} = (P | Q)^{-1} (Q | pq)

    Note that the RI approximation is only meaningful on spatial ERIs.
    Therefore, this routine will crash and exit if the given expression has
    not been spin-integrated before. All RI indices receive an alpha spin
    by default

    Args:
        expr : ExprContainer
            The expression to be spin-integrated.
        symmetric : bool, optional
            If true, the symmetric factorisation variant is employed.
            If false, the asymmetric factorisation variant is employed instead.
    """

    factorisation = 'asym'
    if symmetric:
        factorisation = 'sym'

    # Check whether the expression contains antisymmetric ERIs
    if Symbol(tensor_names.eri) in expr.inner.atoms(Symbol):
        raise Inputerror('Resolution of Identity requires that the ERIs'
                         ' be expanded first.')

    return expr.factorise_eri(factorisation=factorisation)
