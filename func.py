import expr_container as e

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
            expr.real, expr.sym_tensors, expr.provided_target_idx
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
        # but need to resort, because I need another criterion
        # 0 is still preferred and 1 is still killable
        idx = tuple(sorted(
            [s for s in d.idx],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))
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
