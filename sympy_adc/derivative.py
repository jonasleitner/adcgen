from . import expr_container as e
from .misc import Inputerror
from .indices import index_space, get_symbols
from sympy import Dummy, Rational, diff


def derivative(expr: e.expr, t_string: str, indices: str):
    """Computes the derivative of an expression with respect to a specific
       tensor element, e.g., with respect to the t2-amplitude ijab."""

    if not isinstance(t_string, str) or not isinstance(indices, str):
        raise Inputerror("Tensor name and indices of the tensor element "
                         "need to be provided as str.")

    expr = expr.expand()
    if not isinstance(expr, e.expr):  # ensure expr is in a container
        expr = e.expr(expr)

    # get Dummies and generate the space of the indices
    indices = get_symbols(indices)
    space = "".join(index_space(s.name)[0] for s in indices)

    # create some Dummy Symbol. Replace the tensor with the Symbol and
    # compute the derivative with respect to the Symbol. Afterwards
    # resubstitute the Tensor for the Dummy Symbol.
    x = Dummy('x')

    derivative = e.expr(0, **expr.assumptions)
    for term in expr.terms:
        print(f"\nTERM: {term}")
        # assume we do not have any target indices in the term
        if term.target:
            raise NotImplementedError("Gradient only implemented for "
                                      "expectation values - expr without "
                                      f"target indices. Found: {term.target}.")

        obj = term.objects
        # 1) find all occurences of the relevant tensor and create
        #    multiple variants with where one of the occurences is removed
        #    from the term.
        # 2) Permute the indices such that the removed tensor holds the
        #    desired target indices of the derivative
        # 3) symmetrize the term with respect to the symmetry of the
        #    removed tensor that holds the desired indices
        # 4) multiply the remaining term by some Dummy x to represent the
        #    removed tensor (preserve exponent!)
        # 5) compute the derivative with respect to x
        # 6) replace x by the removed tensor
        term_deriv_contribs = e.expr(0, **term.assumptions)
        for i, o in enumerate(obj):
            if 'tensor' in o.type and o.name == t_string and o.space == space:
                exponent = o.exponent
                # found a matching tensor
                # -> remove the tensor from the term
                deriv_contrib = e.expr(1, **term.assumptions)
                for other_i, other_o in enumerate(obj):
                    if i == other_i:
                        continue
                    deriv_contrib *= other_o
                # -> substitute the indices
                sub = {}
                for old, new in zip(o.idx, indices):
                    if old == new:
                        continue
                    additional_sub = {old: new, new: old}
                    for other_old, other_new in sub.items():
                        if other_new == old:
                            sub[other_old] = new
                            del additional_sub[old]
                        elif other_new == new:
                            sub[other_old] = old
                            del additional_sub[new]
                    sub.update(additional_sub)
                deriv_contrib: e.expr = \
                    deriv_contrib.subs(sub, simultaneous=True)
                o: e.term = o.subs(sub, simultaneous=True).terms[0]
                # -> symmetrize the term and replace the removed tensor by
                #    the Dummy x
                sym = o.symmetry()
                deriv_contrib *= Rational(1, len(sym) + 1)
                symmetrized_deriv_contrib = deriv_contrib.sympy * x**exponent
                for perms, factor in sym.items():
                    symmetrized_deriv_contrib += (
                        deriv_contrib.copy().permute(*perms).sympy *
                        factor * x**exponent
                    )
                # compute the derivative wrt to x
                symmetrized_deriv_contrib = diff(symmetrized_deriv_contrib, x)
                # resubstitute the removed tensor for x
                symmetrized_deriv_contrib = \
                    symmetrized_deriv_contrib.subs(x, o.objects[0].extract_pow)
                term_deriv_contribs += symmetrized_deriv_contrib
        print(f"DERIVATIVE: {term_deriv_contribs}\n")
        derivative += term_deriv_contribs
    return derivative
