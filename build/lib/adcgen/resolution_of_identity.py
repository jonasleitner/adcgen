from .expression import ExprContainer
from .sympy_objects import SymmetricTensor
from .tensor_names import tensor_names
from .indices import Indices


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

    resolved_expr = 0

    # We iterate over all terms in the expression and apply RI individually
    for term in expr.terms:
        # Check if the term is spin-integrated
        assert ("n" not in "".join([o.spin for o in term.objects]))
        # Check that no antisymmetric ERIs remain
        assert (tensor_names.eri not in
                ",".join([o.name for o in term.objects]))
        idx_cls = Indices()

        for object in term.objects:
            # Replace spatial ERIs
            if object.name == tensor_names.coulomb:
                # Extract indices
                lower = object.idx[0:2]
                upper = object.idx[2:4]
                ri_idx = idx_cls.get_generic_indices(ri_a=1)[("ri", "a")]

                if symmetric:
                    # v_pqrs = B^P_pq B^P_rs
                    ri_expr = (SymmetricTensor(tensor_names.ri_sym,
                                               (ri_idx,), lower)
                               * SymmetricTensor(tensor_names.ri_sym,
                                                 (ri_idx,), upper))
                else:
                    ri_expr = (SymmetricTensor(tensor_names.ri_asym_eri,
                                               (ri_idx,), upper)
                               * SymmetricTensor(tensor_names.ri_asym_factor,
                                                 (ri_idx,), lower))
                term.subs(object, ri_expr)

        resolved_expr += term
    return resolved_expr
