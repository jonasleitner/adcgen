from sympy_adc.func import (
    _contraction, _contract_operator_string, wicks, evaluate_deltas
)
from sympy_adc.indices import Index

from sympy import S
from sympy.physics.secondquant import F, Fd, KroneckerDelta, substitute_dummies


class TestEvaluateDeltas:
    def test_ev_deltas(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p = Index("p")

        test = KroneckerDelta(i, j) * F(j)
        assert evaluate_deltas(test) == F(i)
        test = KroneckerDelta(i, j) * F(i)
        assert evaluate_deltas(test) == F(j)
        test = KroneckerDelta(i, p) * F(p)
        assert evaluate_deltas(test) == F(i)
        test = KroneckerDelta(i, p) * F(i)  # don't remove i!
        assert evaluate_deltas(test) == test
        test = KroneckerDelta(i, j) * F(p)
        assert evaluate_deltas(test) == test

    def test_with_target_idx(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p = Index("p")

        test = KroneckerDelta(i, j) * F(j)
        assert evaluate_deltas(test, j) == F(j)
        assert evaluate_deltas(test, p) == F(i)
        assert evaluate_deltas(test, (i, j)) == test


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

        zero = (_contraction(Fd(p), F(q)) -
                KroneckerDelta(p, q) *
                KroneckerDelta(q, Index("i", below_fermi=True)))
        assert substitute_dummies(zero, new_indices=True) is S.Zero
        zero = (_contraction(F(p), Fd(q)) -
                KroneckerDelta(p, q) *
                KroneckerDelta(q, Index("a", above_fermi=True)))
        assert substitute_dummies(zero, new_indices=True) is S.Zero

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

    def test_contract_operator_string(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        p, q = Index("p"), Index("q")

        op_string = Fd(i) * F(a) * Fd(b) * F(j)
        ref = KroneckerDelta(i, j) * KroneckerDelta(a, b)
        assert _contract_operator_string(op_string.args) == ref

        op_string = Fd(i) * F(a) * Fd(p) * F(q) * Fd(b) * F(j)
        ref = (-KroneckerDelta(i, q) * KroneckerDelta(a, b)
               * KroneckerDelta(p, j)
               + KroneckerDelta(i, j) * KroneckerDelta(a, p)
               * KroneckerDelta(q, b)
               + KroneckerDelta(i, j) * KroneckerDelta(a, b)
               * KroneckerDelta(p, q)
               * KroneckerDelta(q, Index("i", below_fermi=True)))
        res = _contract_operator_string(op_string.args)
        assert substitute_dummies(ref - res) is S.Zero

    def test_wicks(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        p, q = Index("p"), Index("q")

        expr = (Fd(i) * F(a) * Fd(p) * F(q) * Fd(b) * F(j) * 2
                * KroneckerDelta(i, j))
        ref = (-KroneckerDelta(i, q) * KroneckerDelta(a, b)
               * KroneckerDelta(p, j)
               + KroneckerDelta(i, j) * KroneckerDelta(a, p)
               * KroneckerDelta(q, b)
               + KroneckerDelta(i, j) * KroneckerDelta(a, b)
               * KroneckerDelta(p, q)
               * KroneckerDelta(q, Index("i", below_fermi=True)))
        ref *= 2 * KroneckerDelta(i, j)
        res = wicks(expr)
        assert substitute_dummies(ref - res) is S.Zero
