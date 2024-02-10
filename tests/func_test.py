from sympy_adc.expr_container import Expr
from sympy_adc.func import (
    _contraction, _contract_operator_string, wicks, evaluate_deltas
)
from sympy_adc.indices import Index
from sympy_adc.sympy_objects import Delta

from sympy import S
from sympy.physics.secondquant import F, Fd


class TestEvaluateDeltas:
    def test_ev_deltas(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p = Index("p")

        test = Delta(i, j) * F(j)
        assert evaluate_deltas(test) == F(i)
        test = Delta(i, j) * F(i)
        assert evaluate_deltas(test) == F(j)
        test = Delta(i, p) * F(p)
        assert evaluate_deltas(test) == F(i)
        test = Delta(i, p) * F(i)  # don't remove i!
        assert evaluate_deltas(test) == test
        test = Delta(i, j) * F(p)
        assert evaluate_deltas(test) == test

    def test_with_target_idx(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p = Index("p")

        test = Delta(i, j) * F(j)
        assert evaluate_deltas(test, j) == F(j)
        assert evaluate_deltas(test, p) == F(i)
        assert evaluate_deltas(test, (i, j)) == test


class TestWicks:
    def test_contraction(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        p, q = Index("p"), Index("q")

        assert _contraction(Fd(i), F(j)) == Delta(i, j)
        assert _contraction(F(i), Fd(j)) is S.Zero
        assert _contraction(F(i), F(j)) is S.Zero
        assert _contraction(Fd(i), Fd(j)) is S.Zero

        assert _contraction(Fd(a), F(b)) is S.Zero
        assert _contraction(F(a), Fd(b)) == Delta(a, b)
        assert _contraction(F(a), F(b)) is S.Zero
        assert _contraction(Fd(a), Fd(b)) is S.Zero

        zero = (_contraction(Fd(p), F(q)) -
                Delta(p, q) *
                Delta(q, Index("i", below_fermi=True)))
        assert Expr(zero, target_idx="").substitute_contracted().sympy \
            is S.Zero
        zero = (_contraction(F(p), Fd(q)) -
                Delta(p, q) *
                Delta(q, Index("a", above_fermi=True)))
        assert Expr(zero, target_idx="").substitute_contracted().sympy \
            is S.Zero

        assert _contraction(Fd(i), F(p)) == Delta(i, p)
        assert _contraction(F(p), Fd(i)) is S.Zero
        assert _contraction(F(i), Fd(p)) is S.Zero
        assert _contraction(Fd(p), F(i)) == Delta(i, p)

        assert _contraction(Fd(a), F(p)) is S.Zero
        assert _contraction(F(p), Fd(a)) == Delta(a, p)
        assert _contraction(F(p), Fd(a)) == Delta(a, p)
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
        ref = Delta(i, j) * Delta(a, b)
        assert _contract_operator_string(op_string.args) == ref

        op_string = Fd(i) * F(a) * Fd(p) * F(q) * Fd(b) * F(j)
        ref = (-Delta(i, q) * Delta(a, b)
               * Delta(p, j)
               + Delta(i, j) * Delta(a, p)
               * Delta(q, b)
               + Delta(i, j) * Delta(a, b)
               * Delta(p, q)
               * Delta(q, Index("i", below_fermi=True)))
        res = _contract_operator_string(op_string.args)
        assert Expr(ref - res, target_idx="").substitute_contracted().sympy \
            is S.Zero

    def test_wicks(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        p, q = Index("p"), Index("q")

        expr = (Fd(i) * F(a) * Fd(p) * F(q) * Fd(b) * F(j) * 2
                * Delta(i, j))
        ref = (-Delta(i, q) * Delta(a, b)
               * Delta(p, j)
               + Delta(i, j) * Delta(a, p)
               * Delta(q, b)
               + Delta(i, j) * Delta(a, b)
               * Delta(p, q)
               * Delta(q, Index("i", below_fermi=True)))
        ref *= 2 * Delta(i, j)
        res = wicks(expr)
        assert Expr(ref - res, target_idx="").substitute_contracted().sympy \
            is S.Zero
