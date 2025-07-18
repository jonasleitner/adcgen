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
        If set, the :py:class:`KroneckerDelta` set generated through the
        contractions will be evaluated before returning.
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
    elif isinstance(expr, Mul):
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
            if simplify_kronecker_deltas:
                result = evaluate_deltas(result)
    else:  # neither Add, Mul, NO, F, Fd -> maybe a number or tensor
        result = expr

    # apply rules to the result
    if rules is not None:
        assert isinstance(rules, Rules)
        result = rules.apply(ExprContainer(result)).inner
    return result


def _contract_operator_string(
        op_string: Sequence[tuple[int, FermionicOperator]],
        n_total_operators: int | None = None,
        contractions: Sequence[Expr] | None = None) -> Expr:
    """
    Contracts the operator string only returning fully contracted
    contritbutions.

    Parameters
    ----------
    op_string: Sequence[tuple[int, FermionicOperator]]
        The list/tuple of second quantized operators to contract
        along with their positions in the operator string
    n_total_operators: int | None, optional
        The total number of second quantized operators in the
        operator string. By default, it is calculated from the
        length of the contraction cache.
    contractions: Sequence[Expr] | None, optional
        Precomputed flattened array of contractions of operator
        pairs. Only the upper triangular part
        (excluding diagonal elements) of the pair matrix
        is expected to be present in the cache: for N operators
        a cache with N(N-1)/2 elements is expected. If not given
        it will be calculated on the fly for the current
        operator string.
    """
    # This function implements recursive depth first tree traversal

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
    # the flattened index of the element (i, j) that is part of
    # the upper triangle (excluding diagonal elements) of a n x n matrix
    # can be computed accoding to
    # idx = (2*i*n - i**2 + 2*j - 3*i - 2) / 2
    # using row index i, column index j and the number of elements
    # along each dimension n.
    # for a derivation see:
    # math.stackexchange.com/questions/646117/how-to-find-a-function-mapping-matrix-indices
    if contractions is None:
        contractions = tuple(
            _contraction(op1, op2) for (_, op1), (_, op2) in
            itertools.combinations(op_string, 2)
        )
    # calculate the number of operators from the length of the
    # contraction cache: n(n-1)/2 elements are in the cache
    # required for the calculation of the flattened cache index
    if n_total_operators is None:
        n = 0.5 + math.sqrt(0.25 + 2 * len(contractions))
        # if n is not integer, the number of entries in the
        # contraction cache is not valid
        assert n.is_integer()
        n_total_operators = int(n)
        del n
    # left_pos and right_pos denote the positions in the
    # original full operator string (before contracting any
    # operators). This is required for the cache lookup.
    # The loop index i denotes the position in the
    # current version of the operator string potentially
    # with previous contraction of other operators. Required to
    # to determine the sign of a contraction
    result = []
    left_pos, _ = op_string[0]
    for i in range(1, len(op_string)):
        right_pos, _ = op_string[i]
        # compute the flattened index for the (left_pos, right_pos)
        # element of the upper triangular matrix as explained above
        flattened_idx = (
            2 * left_pos * n_total_operators - left_pos * left_pos
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
            c *= _contract_operator_string(
                op_string=remaining, n_total_operators=n_total_operators,
                contractions=contractions
            )
            result.append(c)
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
    # in principle this also works for indices with spin since the
    # KroneckerDelta handles spin correctly. However, mixed deltas
    # \delta_{i j_{\alpha}} might not be evaluated by evaluate_deltas,
    # because the preferred index would be i_{\alpha} - a new index
    # that is not present in the expression.
    # To avoid unevaluated deltas this check was introduced
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
    Takes a list of second quantized operators and their respective positions
    in the operator string and checks whether a non-vanishing fully contracted
    contribution can exist.
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
