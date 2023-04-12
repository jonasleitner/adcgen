from . import expr_container as e
from .indices import index_space, minimize_tensor_indices
from sympy import Dummy, Rational, diff, S


def derivative(expr: e.expr, t_string: str):
    """Computes the derivative of an expression with respect to a tensor.
       The derivative is separated block whise, e.g, terms that contribute to
       the derivative w.r.t. the oooo ERI block are separated from terms that
       contribute to the ooov block.
       Assumptions of the input expression are NOT updated or modified.
       The derivative is NOT simplified."""

    if not isinstance(t_string, str):
        raise TypeError("Tensor name needs to be provided as str.")

    expr = expr.expand()
    if not isinstance(expr, e.expr):  # ensure expr is in a container
        expr = e.expr(expr)

    # create some Dummy Symbol. Replace the tensor with the Symbol and
    # compute the derivative with respect to the Symbol. Afterwards
    # resubstitute the Tensor for the Dummy Symbol.
    x = Dummy('x')

    derivative = {}
    for term in expr.terms:
        assumptions = term.assumptions
        objects = term.objects
        # - find all occurences of the desired tensor
        tensor_obj = []
        remaining_obj = e.expr(1, **term.assumptions)
        for obj in objects:
            if obj.name == t_string:
                tensor_obj.append(obj)
            else:
                remaining_obj *= obj

        # - extract the names of target indices of the term
        target_names_by_space = {}
        for s in term.target:
            if (sp := index_space(s.name)) not in target_names_by_space:
                target_names_by_space[sp] = set()
            target_names_by_space[sp].add(s.name)

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
            if deriv_contrib.sympy is S.Zero:
                raise RuntimeError(f"Mnimization permutations {perms} let "
                                   f"the remaining term {deriv_contrib} "
                                   "vanish.")
            # - Apply the permutations to the object. Might introduce
            #   a prefactor of -1 that we need to move to the deriv_contrib.
            #   Also the indices might be further minimized due to the
            #   symmetry of the tensor obj
            obj: e.term = obj.permute(*perms).terms[0]
            if (factor := obj.prefactor) < 0:
                deriv_contrib *= factor
            # - Apply the symmetry of the removed tensor to the remaining
            #   term to ensure that the result has the correct symmetry.
            #   Also replace the removed tensor by a Dummy Variable x.
            #   This allows to compute the symbolic derivative with diff.
            tensor_sym = obj.symmetry()
            deriv_contrib *= Rational(1, len(tensor_sym) + 1)
            symmetrized_deriv_contrib = deriv_contrib.sympy * x**exponent
            for perms, factor in tensor_sym.items():
                symmetrized_deriv_contrib += (
                    deriv_contrib.copy().permute(*perms).sympy *
                    factor * x**exponent
                )
            # - compute the derivative with respect to x
            symmetrized_deriv_contrib = diff(symmetrized_deriv_contrib, x)
            # - replace x by the removed tensor (due to diff the exponent
            #   is lowered by 1)
            obj = obj.tensors
            assert len(obj) == 1
            obj = obj[0]
            symmetrized_deriv_contrib = (
                symmetrized_deriv_contrib.subs(x, obj)
            )
            # - sort the derivative according to the space of the minimal
            #   tensor indices
            #   -> sort the derivative block whise.
            space = obj.space
            if space not in derivative:
                derivative[space] = e.expr(0, **assumptions)
            derivative[space] += symmetrized_deriv_contrib
    return derivative
