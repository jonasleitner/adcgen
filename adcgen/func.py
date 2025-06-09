from collections.abc import Sequence
import itertools

from sympy.physics.secondquant import (
    F, Fd, FermionicOperator, NO
)
from sympy import S, Add, Expr, Mul

from .expression import ExprContainer
from .misc import Inputerror
from .rules import Rules
from .indices import Index, Indices, get_symbols
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


def wicks(expr: Expr, rules: Rules | None = None,
          simplify_kronecker_deltas: bool = False) -> Expr:
    """
    Evaluates Wicks theorem in the provided expression only returning fully
    contracted contributions.
    Adapted from the implementation in 'sympy.physics.secondquant'.

    Parameters
    ----------
    expr: Expr
        Expression containing the second quantized operator strings to
        evaluate. This function expects plain sympy objects (Add/Mul/...)
        and no container class.
    rules : Rules, optional
        Rules that are applied to the result before returning, e.g., in the
        context of RE not all tensor blocks might be allowed in the result.
    simplify_kronecker_deltas : bool, optional
        If set, the KroneckerDeltas generated through the contractions
        will be evaluated before returning.
    """
    assert isinstance(expr, Expr)
    # normal ordered operator string has to evaluate to zero
    # and a single second quantized operator can not be contracted
    if isinstance(expr, (NO, FermionicOperator)):
        return S.Zero

    # break up any NO-objects, and evaluate commutators
    expr = expr.doit(wicks=True).expand()
    assert isinstance(expr, Expr)

    if isinstance(expr, Add):
        return Add(*(
            wicks(term, rules=rules,
                  simplify_kronecker_deltas=simplify_kronecker_deltas)
            for term in expr.args
        ))
    elif not isinstance(expr, Mul):
        # nether Add, Mul, NO, F, Fd -> maybe a number or tensor
        return expr
    # -> we have a Mul object
    # we don't want to mess around with commuting part of Mul
    # so we factorize it out before starting recursion
    c_part: list[Expr] = []
    op_string: list[FermionicOperator] = []
    for factor in expr.args:
        if factor.is_commutative:
            c_part.append(factor)
        else:
            assert isinstance(factor, FermionicOperator)
            op_string.append(factor)

    if (n := len(op_string)) == 0:  # no operators
        result = expr
    elif n == 1:  # a single operator
        return S.Zero
    else:  # at least 2 operators
        result = _contract_operator_string(op_string)
        result = Mul(*c_part, result).expand()
        assert isinstance(result, Expr)
        if simplify_kronecker_deltas:
            result = evaluate_deltas(result)

    # apply rules to the result
    if rules is None:
        return result
    assert isinstance(rules, Rules)
    return rules.apply(ExprContainer(result)).inner


def _contract_operator_string(op_string: list[FermionicOperator]) -> Expr:
    """
    Contracts the operator string only returning fully contracted
    contritbutions.
    Adapted from 'sympy.physics.secondquant'.
    """
    # check that we can get a fully contracted contribution
    if not _has_fully_contracted_contribution(op_string):
        return S.Zero

    result = []
    for i in range(1, len(op_string)):
        c = _contraction(op_string[0], op_string[i])
        if c is S.Zero:
            continue
        if not i % 2:  # introduce -1 for swapping operators
            c *= S.NegativeOne

        if len(op_string) - 2 > 0:  # at least one operator left
            # remove the contracted operators from the string and recurse
            remaining = op_string[1:i] + op_string[i+1:]
            result.append(Mul(c, _contract_operator_string(remaining)))
        else:  # no operators left
            result.append(c)
    return Add(*result)


def _contraction(p: FermionicOperator, q: FermionicOperator) -> Expr:
    """
    Evaluates the contraction between two sqcond quantized fermionic
    operators.
    Adapted from 'sympy.physics.secondquant'.
    """
    assert isinstance(p, FermionicOperator)
    assert isinstance(q, FermionicOperator)

    p_idx, q_idx = p.args[0], q.args[0]
    assert isinstance(p_idx, Index) and isinstance(q_idx, Index)
    if p_idx.spin or q_idx.spin:
        raise NotImplementedError("Contraction not implemented for indices "
                                  "with spin.")
    # get the space and ensure we have no unexpected space
    space_p, space_q = p_idx.space[0], q_idx.space[0]
    assert space_p in ["o", "v", "g"] and space_q in ["o", "v", "g"]

    if isinstance(p, F) and isinstance(q, Fd):
        if space_p == "o" or space_q == "o":
            res = S.Zero
        elif space_p == "v" or space_q == "v":
            res = KroneckerDelta(p_idx, q_idx)
        else:
            res = Mul(
                KroneckerDelta(p_idx, q_idx),
                KroneckerDelta(q_idx, Index('a', above_fermi=True))
            )
    elif isinstance(p, Fd) and isinstance(q, F):
        if space_p == "v" or space_q == "v":
            res = S.Zero
        elif space_p == "o" or space_q == "o":
            res = KroneckerDelta(p_idx, q_idx)
        else:
            res = Mul(
                KroneckerDelta(p_idx, q_idx),
                KroneckerDelta(q_idx, Index('i', below_fermi=True))
            )
    else:  # vanish if 2xAnnihilator or 2xCreator
        res = S.Zero
    assert isinstance(res, Expr)
    return res


def _has_fully_contracted_contribution(op_string: list[FermionicOperator]
                                       ) -> bool:
    """
    Takes a list of second quantized operators and checks whether a
    non-vanishing fully contracted contribution can exist.
    """
    if len(op_string) % 2:  # odd number of operators
        return False
    # count the number of creation and annihilation operators per space
    create = {space: 0 for space in Indices.base}
    annihilate = {space: 0 for space in Indices.base}
    for op in op_string:
        if isinstance(op, Fd):
            counter = create
        else:
            counter = annihilate
        idx = op.args[0]
        assert isinstance(idx, Index)
        counter[idx.space] += 1
    # check that we have a matching amount of creation and annihilation
    # operators
    for space, n_create in create.items():
        if space == "general":
            continue
        n_annihilate = annihilate[space] + annihilate["general"]
        if n_create - n_annihilate > 0:
            return False
    return True
