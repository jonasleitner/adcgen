from sympy import Mul, Rational, S, diff

from .expression import ExprContainer, ObjectContainer
from .indices import minimize_tensor_indices, Index
from .sympy_objects import SymbolicTensor


def derivative(expr: ExprContainer, t_string: str
               ) -> dict[tuple[str, str], ExprContainer]:
    """Computes the derivative of an expression with respect to a tensor.
       The derivative is separated block whise, e.g, terms that contribute to
       the derivative w.r.t. the oooo ERI block are separated from terms that
       contribute to the ooov block.
       Assumptions of the input expression are NOT updated or modified.
       The derivative is NOT simplified."""
    assert isinstance(t_string, str)
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    # create some Dummy Symbol. Replace the tensor with the Symbol and
    # compute the derivative with respect to the Symbol. Afterwards
    # resubstitute the Tensor for the Dummy Symbol.
    x = Index('x')

    derivative = {}
    for term in expr.terms:
        assumptions = term.assumptions
        objects = term.objects
        # - find all occurences of the desired tensor
        tensor_obj: list[ObjectContainer] = []
        remaining_obj = ExprContainer(1, **term.assumptions)
        for obj in objects:
            if obj.name == t_string:
                tensor_obj.append(obj)
            else:
                remaining_obj *= obj

        # - extract the names of target indices of the term
        target_names_by_space: dict[tuple[str, str], set[str]] = {}
        for s in term.target:
            if (key := s.space_and_spin) not in target_names_by_space:
                target_names_by_space[key] = set()
            target_names_by_space[key].add(s.name)

        # 2) go through the tensor_obj list and compute the derivative
        #    for all found occurences one after another (product rule)
        for i, obj in enumerate(tensor_obj):
            # - extract the exponent of the tensor
            exponent = obj.exponent
            # - rebuild the term without the current occurence of the
            #   tensor obj
            deriv_contrib = remaining_obj.copy()
            for other_i, other_obj in enumerate(tensor_obj):
                if i != other_i:
                    deriv_contrib *= other_obj
            # - minimize the indices of the removed tensor
            _, perms = minimize_tensor_indices(obj.idx, target_names_by_space)
            # - apply the permutations to the remaining term
            deriv_contrib = deriv_contrib.permute(*perms)
            if deriv_contrib.inner is S.Zero:
                raise RuntimeError(f"Mnimization permutations {perms} let "
                                   f"the remaining term {deriv_contrib} "
                                   "vanish.")
            # - Apply the permutations to the object. Might introduce
            #   a prefactor of -1 that we need to move to the deriv_contrib.
            #   Also the indices might be further minimized due to the
            #   symmetry of the tensor obj
            obj = obj.permute(*perms).terms[0]
            if (factor := obj.prefactor) < S.Zero:
                deriv_contrib *= factor
            # - Apply the symmetry of the removed tensor to the remaining
            #   term to ensure that the result has the correct symmetry.
            #   Also replace the removed tensor by a Dummy Variable x.
            #   This allows to compute the symbolic derivative with diff.
            tensor_sym = obj.symmetry()
            deriv_contrib *= Rational(1, len(tensor_sym) + 1)
            symmetrized_deriv_contrib = S.Zero
            symmetrized_deriv_contrib += Mul(deriv_contrib.inner, x**exponent)
            for perms, factor in tensor_sym.items():
                symmetrized_deriv_contrib += Mul(
                    deriv_contrib.copy().permute(*perms).inner,
                    factor,
                    x**exponent
                )
            # - compute the derivative with respect to x
            symmetrized_deriv_contrib = diff(symmetrized_deriv_contrib, x)
            # - replace x by the removed tensor (due to diff the exponent
            #   is lowered by 1)
            obj = [
                o for o in obj.objects if isinstance(o.base, SymbolicTensor)
            ]
            assert len(obj) == 1
            obj = obj[0]
            symmetrized_deriv_contrib = (
                symmetrized_deriv_contrib.subs(x, obj.base)
            )
            # - sort the derivative according to the space of the minimal
            #   tensor indices
            #   -> sort the derivative block whise.
            key = (obj.space, obj.spin)
            if key not in derivative:
                derivative[key] = ExprContainer(0, **assumptions)
            derivative[key] += symmetrized_deriv_contrib
    return derivative
