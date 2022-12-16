from . import expr_container as e
from .misc import Inputerror
from sympy import Add


def evaluate_deltas(expr):
    """Basically identical to the evaluate delta function of sympy. The only
       difference being how preferred and killable index of the KroneckerDelta
       are determined. Sympy just sorts them alphabetically -> a1 < b.
       This function also takes the numbering into account -> b < a1."""
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    # the expr consists of multiple terms
    if len(expr) > 1:
        return e.expr(
            Add(*[evaluate_deltas(term).sympy for term in expr.terms]),
            **expr.assumptions
        )
    # just a single term or object
    expr = expr.terms[0]
    if len(expr.polynoms) != 0:
        raise NotImplementedError("evaluate_delta is not implemented "
                                  f"for denominators. Found the term: {expr}.")

    # can simplify substitute the indices, because delta_ij -> delta_ii = 1
    # we can't hit any delta_ia, because delta_ia = 0, which is automatically
    # evaluated
    # -> preferred and killable need to belong to the same space
    target = expr.target
    for d in expr.deltas:
        # idx for deltas are sorted as: (preferred, killable)
        idx = d.idx
        # try to kill killable
        if idx[1] not in target:
            expr = expr.subs({idx[1]: idx[0]})
            return evaluate_deltas(expr)
        # try instead to kill preferred
        elif idx[0] not in target:
            expr = expr.subs({idx[0]: idx[1]})
            return evaluate_deltas(expr)
        # can kill neither of them without loosing a target index
        else:
            pass
    return expr


def gen_term_orders(order, term_length, min_order):
    """Generates all combinations that contribute to the n'th order
       contribution of a term x*x*x*..., where x is expanded in a perturbation
       expansion.

       :param order: The desired order
       :type order: int
       :param term_length: The number of objects in the term to expand in
            perturbation theory.
       :type term_length: int
       :param min_order: The minimum order that should be considered
       :type min_order: int
       :return: All possible combinations of a given order
       :rtype: list
       """
    from itertools import product

    if not all(isinstance(n, int) and n >= 0
               for n in [order, term_length, min_order]):
        raise Inputerror("Order, term_length and min_order need to be "
                         "non-negative integers.")

    orders = (o for o in range(min_order, order + 1))
    combinations = product(orders, repeat=term_length)
    return [comb for comb in combinations if sum(comb) == order]
