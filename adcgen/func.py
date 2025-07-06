from collections.abc import Sequence
import itertools

from sympy import Add, Expr, Mul

from .misc import Inputerror
from .indices import Index, get_symbols
from .sympy_objects import KroneckerDelta


def gen_term_orders(order: int, term_length: int, min_order: int
                    ) -> list[tuple[int, ...]]:
    """
    Generate all combinations of orders that contribute to the n'th-order
    contribution of a term of the given length
    (a * b * c * ...)^{(n)},
    where a, b and c are each subject of a perturbation expansion.

    Parameters
    ----------
    order : int
        The perturbation theoretical order n.
    term_length : int
        The number of objects in the term.
    min_order : int
        The minimum perturbation theoretical order of the objects in the
        term to consider. For instance, 2 if the first and zeroth order
        contributions are not relevant, because they vanish or are considered
        separately.
    """

    if not all(isinstance(n, int) and n >= 0
               for n in [order, term_length, min_order]):
        raise Inputerror("Order, term_length and min_order need to be "
                         "non-negative integers.")

    orders = (o for o in range(min_order, order + 1))
    combinations = itertools.product(orders, repeat=term_length)
    return [comb for comb in combinations if sum(comb) == order]


def evaluate_deltas(
        expr: Expr,
        target_idx: Sequence[str] | Index | Sequence[Index] | None = None
        ) -> Expr:
    """
    Evaluates the KroneckerDeltas in an expression.
    The function only removes contracted indices from the expression and
    ensures that no information is lost if an index is removed.
    Adapted from the implementation in 'sympy.physics.secondquant'.
    Note that KroneckerDeltas in a Polynom (a*b + c*d)^n will not be evaluated.
    However, in most cases the expression can simply be expanded before
    calling this function.

    Parameters
    ----------
    expr: Expr
        Expression containing the KroneckerDeltas to evaluate. This function
        expects a plain object from sympy (Add/Mul/...) and no container class.
    target_idx : Sequence[str] | Sequence[Index] | None, optional
        Optionally, target indices can be provided if they can not be
        determined from the expression using the Einstein sum convention.
    """
    assert isinstance(expr, Expr)

    if isinstance(expr, Add):
        return Add(*(
            evaluate_deltas(arg, target_idx) for arg in expr.args
        ))
    elif isinstance(expr, Mul):
        if target_idx is None:
            # for determining the target indices it is sufficient to use
            # atoms, which lists every index only once per object, i.e.,
            # (f_ii).atoms(Index) -> i.
            # We are only interested in indices on deltas
            # -> it is sufficient to know that an index occurs on another
            #    object. (twice on the same delta is not possible)
            deltas: list[KroneckerDelta] = []
            indices: dict[Index, int] = {}
            for obj in expr.args:
                for s in obj.atoms(Index):
                    if s in indices:
                        indices[s] += 1
                    else:
                        indices[s] = 0
                if isinstance(obj, KroneckerDelta):
                    deltas.append(obj)
            # extract the target indices and use them in next recursion
            # so they only need to be determined once
            target_idx = [s for s, n in indices.items() if not n]
        else:
            # find all occurrences of kronecker delta
            deltas = [d for d in expr.args if isinstance(d, KroneckerDelta)]
            target_idx = get_symbols(target_idx)

        for d in deltas:
            # determine the killable and preferred index
            # in the case we have delta_{i p_alpha} we want to keep i_alpha
            # -> a new index is required. But for now just don't evaluate
            #    these deltas
            idx = d.preferred_and_killable
            if idx is None:  # delta_{i p_alpha}
                continue
            preferred, killable = idx
            # try to remove killable
            if killable not in target_idx:
                res = expr.subs(killable, preferred)
                assert isinstance(res, Expr)
                expr = res
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
            # try to remove preferred.
            # But only if no information is lost if doing so
            # -> killable has to be of length 1
            elif preferred not in target_idx \
                    and d.indices_contain_equal_information:
                res = expr.subs(preferred, killable)
                assert isinstance(res, Expr)
                expr = res
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
        return expr
    else:
        return expr
