from adcgen.expression import ExprContainer
from adcgen.indices import Index, get_symbols
from adcgen.sympy_objects import KroneckerDelta
from adcgen.wicks import _contraction, _contract_operator_string, wicks

from sympy.physics.secondquant import F, Fd
from sympy import S

import pytest


class TestWicks:
    def test_contraction(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        p, q = Index("p"), Index("q")

        assert _contraction(Fd(i), F(j)) == KroneckerDelta(i, j)
        assert _contraction(F(i), Fd(j)) is S.Zero
        assert _contraction(F(i), F(j)) is S.Zero
        assert _contraction(Fd(i), Fd(j)) is S.Zero

        assert _contraction(Fd(a), F(b)) is S.Zero
        assert _contraction(F(a), Fd(b)) == KroneckerDelta(a, b)
        assert _contraction(F(a), F(b)) is S.Zero
        assert _contraction(Fd(a), Fd(b)) is S.Zero

        zero = (
            _contraction(Fd(p), F(q))
            - (KroneckerDelta(p, q) *
               KroneckerDelta(q, Index("i", below_fermi=True)))
        )
        zero = ExprContainer(zero, target_idx="").substitute_contracted()
        assert zero.inner is S.Zero
        zero = (
            _contraction(F(p), Fd(q))
            - (KroneckerDelta(p, q) *
               KroneckerDelta(q, Index("a", above_fermi=True)))
        )
        zero = ExprContainer(zero, target_idx="").substitute_contracted()
        assert zero.inner is S.Zero

        assert _contraction(Fd(i), F(p)) == KroneckerDelta(i, p)
        assert _contraction(F(p), Fd(i)) is S.Zero
        assert _contraction(F(i), Fd(p)) is S.Zero
        assert _contraction(Fd(p), F(i)) == KroneckerDelta(i, p)

        assert _contraction(Fd(a), F(p)) is S.Zero
        assert _contraction(F(p), Fd(a)) == KroneckerDelta(a, p)
        assert _contraction(F(p), Fd(a)) == KroneckerDelta(a, p)
        assert _contraction(Fd(a), F(p)) is S.Zero

        assert _contraction(Fd(a), F(i)) is S.Zero
        assert _contraction(F(i), Fd(a)) is S.Zero
        assert _contraction(F(a), Fd(i)) is S.Zero
        assert _contraction(Fd(i), F(a)) is S.Zero

    def test_contraction_spin(self):
        i, j = get_symbols("ij", "ab")
        with pytest.raises(NotImplementedError):
            _contraction(Fd(i), F(j))

    def test_contract_operator_string(self):
        i, j, a, b, p, q = get_symbols("ijabpq")

        op_string = [Fd(i), F(a), Fd(b), F(j)]
        ref = KroneckerDelta(i, j) * KroneckerDelta(a, b)
        assert _contract_operator_string(tuple(enumerate(op_string))) == ref

        op_string = [Fd(i), F(a), Fd(p), F(q), Fd(b), F(j)]
        ref = (
            - (KroneckerDelta(i, q) * KroneckerDelta(a, b) *
               KroneckerDelta(p, j))
            + (KroneckerDelta(i, j) * KroneckerDelta(a, p) *
               KroneckerDelta(q, b))
            + (KroneckerDelta(i, j) * KroneckerDelta(a, b) *
               KroneckerDelta(p, q) *
               KroneckerDelta(q, Index("i", below_fermi=True)))
        )
        res = _contract_operator_string(tuple(enumerate(op_string))).expand()
        zero = ExprContainer(
            ref - res, target_idx="ijab"
        ).substitute_contracted()
        assert zero.inner is S.Zero

    def test_wicks(self):
        i, j, a, b, p, q = get_symbols("ijabpq")

        expr = (
            Fd(i) * F(a) * Fd(p) * F(q) * Fd(b) * F(j) * 2 *
            KroneckerDelta(i, j)
        )
        ref = S.Zero
        ref -= (
            KroneckerDelta(i, q) * KroneckerDelta(a, b) * KroneckerDelta(p, j)
        )
        ref += (
            KroneckerDelta(i, j) * KroneckerDelta(a, p) * KroneckerDelta(q, b)
        )
        ref += (
            KroneckerDelta(i, j) * KroneckerDelta(a, b) * KroneckerDelta(p, q)
            * KroneckerDelta(q, Index("i", below_fermi=True))
        )
        ref *= (2 * KroneckerDelta(i, j))
        res = wicks(expr)
        zero = ExprContainer(
            ref.expand() - res, target_idx="ijab"
        ).substitute_contracted()
        assert zero.inner is S.Zero
