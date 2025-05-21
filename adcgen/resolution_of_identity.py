from .expression import ExprContainer
from .tensor_names import tensor_names
from .misc import Inputerror
from sympy import Symbol


def apply_resolution_of_identity(expr: ExprContainer,
                                 factorisation: str = 'sym'
                                 ) -> ExprContainer:
    """
    Applies the Resolution of Identity approximation (RI, sometimes also
    called density fitting, DF) to an expression. This implies that every
    Coulomb operator is replaced by its factorised form. Two types of
    factorisation are supported: symmetric and asymmetric.
    In the symmetric decomposition, a Coulomb operator is approximated as:

    (pq | rs) ~ B^P_{pq} B^P_{rs}
    B^P_{pq} = (P | Q)^{-1/2} (Q | pq)

    This decomposition is the default. In the asymmetric factorisation, the
    same Coulomb operator is approximated as:

    (pq | rs) ~ C^P_{pq} (P | rs)
    C^P_{pq} = (P | Q)^{-1} (Q | pq)

    Note that the RI approximation is only meaningful on Coulomb operator.
    Therefore, this routine will crash and exit if the given expression has
    not been expanded before. All RI indices receive an alpha spin
    by default if the expression has been spin-integrated and no spin
    otherwise.

    Parameters
    ----------
    expr : ExprContainer
        The expression to be factorised into RI format.
    factorisation : str, optional
        Which type of factorisation to use.
        If 'sym', the symmetric factorisation variant is employed.
        If 'asym', the asymmetric factorisation variant is employed
        instead, by default 'sym'

    Returns
    -------
    ExprContainer
        The factorised expression

    Raises
    ------
    Inputerror
        If a factorisation other than 'sym' or 'asym' is provided
    Inputerror
        If the expression still contains antisymmetric ERIs.
    """

    # Check if a valid factorisation is given
    if factorisation not in ('sym', 'asym'):
        raise Inputerror('Only symmetric (sym) and asymmetric (asym) '
                         'factorisation modes are supported. '
                         f'Received: {factorisation}')

    # Check whether the expression contains antisymmetric ERIs
    if Symbol(tensor_names.eri) in expr.inner.atoms(Symbol):
        raise Inputerror('Resolution of Identity requires that the ERIs'
                         ' be expanded first.')

    return expr.expand_coulomb_ri(factorisation=factorisation)
