from collections.abc import Sequence
import itertools
import math

from sympy.physics.secondquant import F, Fd, FermionicOperator, NO
from sympy import Add, Expr, Mul, S

from .expression import ExprContainer
from .func import evaluate_deltas
from .indices import Index, Indices
from .rules import Rules
from .sympy_objects import KroneckerDelta


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
        result = _contract_operator_string(
            op_string=tuple(enumerate(op_string))
        )
        result = Mul(*c_part, result).expand()
        assert isinstance(result, Expr)
        if simplify_kronecker_deltas:
            result = evaluate_deltas(result)

    # apply rules to the result
    if rules is None:
        return result
    assert isinstance(rules, Rules)
    return rules.apply(ExprContainer(result)).inner


def _contract_operator_string(
        op_string: Sequence[tuple[int, FermionicOperator]],
        contractions: Sequence[Expr] | None = None) -> Expr:
    """
    Contracts the operator string only returning fully contracted
    contritbutions.
    """
    # check that we can get a fully contracted contribution
    if not _has_fully_contracted_contribution(op_string):
        return S.Zero
    # we can precompute all relevant contractions as they can
    # be reused often in an depth first graph traversal
    # It is sufficient to precompute the upper triangle
    #   1  2  3
    # 1    x  x
    # 2       x
    # 3
    # the element (i, j) that is part of the upper triangle
    # (excluding diagonal elements) of a n x n matrix
    # can be computed accoding to
    # idx = (2*i*n - i**2 + 2*j - 3*i - 2) / 2
    if contractions is None:
        contractions = tuple(
            _contraction(op1, op2) for (_, op1), (_, op2) in
            itertools.combinations(op_string, 2)
        )
        n_operators = len(op_string)
    else:
        # calculate the number of operators from the length of the
        # contraction cache: n(n-1)/2 elements are in the cache
        # required for the calculation of the flattened cache index
        n_operators = int(0.5 + math.sqrt(0.25 + 2 * len(contractions)))

    result = []
    left_pos, _ = op_string[0]
    for i in range(1, len(op_string)):
        right_pos, _ = op_string[i]
        # compute the index in the flattened cache of the upper triangular
        # matrix and load the contraction result
        flattened_idx = (
            2 * left_pos * n_operators - left_pos * left_pos
            + 2 * right_pos - 3 * left_pos - 2
        ) // 2
        c = contractions[flattened_idx]
        if c is S.Zero:
            continue
        if not i % 2:  # introduce -1 for swapping operators
            c *= S.NegativeOne

        if len(op_string) - 2 > 0:  # at least one operator left
            # remove the contracted operators from the string and recurse
            remaining = tuple(
                ele for j, ele in
                enumerate(op_string) if j != 0 and j != i
            )
            result.append(
                c * _contract_operator_string(op_string=remaining,
                                              contractions=contractions)
            )
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
            res = (
                KroneckerDelta(p_idx, q_idx) *
                KroneckerDelta(q_idx, Index('a', above_fermi=True))
            )
    elif isinstance(p, Fd) and isinstance(q, F):
        if space_p == "v" or space_q == "v":
            res = S.Zero
        elif space_p == "o" or space_q == "o":
            res = KroneckerDelta(p_idx, q_idx)
        else:
            res = (
                KroneckerDelta(p_idx, q_idx) *
                KroneckerDelta(q_idx, Index('i', below_fermi=True))
            )
    else:  # vanish if 2xAnnihilator or 2xCreator
        res = S.Zero
    return res


def _has_fully_contracted_contribution(
        op_string: Sequence[tuple[int, FermionicOperator]]) -> bool:
    """
    Takes a list of second quantized operators and checks whether a
    non-vanishing fully contracted contribution can exist.
    """
    if len(op_string) % 2:  # odd number of operators
        return False
    # count the number of creation and annihilation operators per space
    idx_spaces: tuple[str, ...] = tuple(Indices.base.keys())
    creation: list[int] = [0 for _ in idx_spaces]
    annihilation: list[int] = [0 for _ in idx_spaces]
    for _, op in op_string:
        if isinstance(op, Fd):
            counter = creation
        else:
            counter = annihilation
        idx = op.args[0]
        assert isinstance(idx, Index)
        counter[idx_spaces.index(idx.space)] += 1
    # check that we have a matching amount of creation and annihilation
    # operators
    if sum(creation) != sum(annihilation):
        return False
    # remove the general operators from the counters
    n_general_creation = creation.pop(idx_spaces.index("general"))
    n_general_annihilation = annihilation.pop(idx_spaces.index("general"))
    # ensure that we have a matching amount of creation and annihilation
    # operators for each space accounting for general operators
    for n_create, n_annihilate in zip(creation, annihilation, strict=True):
        if n_create == n_annihilate:
            continue
        elif n_create > n_annihilate:
            n_general_annihilation -= n_create - n_annihilate
            if n_general_annihilation < 0:
                return False
        else:  # n_create < n_annihilate
            n_general_creation -= n_annihilate - n_create
            if n_general_creation < 0:
                return False
    # we have to have the same number of general creation and annihilation
    # operators at this point. Otherwise sum(creation) != sum(annihilation)
    # and we would have failed above
    return True
