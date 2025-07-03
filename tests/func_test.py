from adcgen.expression import ExprContainer
from adcgen.func import evaluate_deltas
from adcgen.indices import Index, get_symbols
from adcgen.sympy_objects import KroneckerDelta
from adcgen.wicks import wicks, _contract_operator_string, _contraction

from sympy import Add, Mul, S
from sympy.physics.secondquant import F, Fd


class TestEvaluateDeltas:
    def test_ev_deltas(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p, pa = Index("p"), Index("p", alpha=True)

        test = Mul(KroneckerDelta(i, j), F(j))
        assert evaluate_deltas(test) == F(i)
        test = Mul(KroneckerDelta(i, j), F(i))
        assert evaluate_deltas(test) == F(j)
        test = Mul(KroneckerDelta(i, p), F(p))
        assert evaluate_deltas(test) == F(i)
        test = Mul(KroneckerDelta(i, p), F(i))  # don't remove i!
        assert evaluate_deltas(test) == test
        test = Mul(KroneckerDelta(i, j), F(p))
        assert evaluate_deltas(test) == test
        test = Mul(KroneckerDelta(i, pa), F(i))
        assert evaluate_deltas(test) == test

    def test_with_target_idx(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        p = Index("p")

        test = Mul(KroneckerDelta(i, j), F(j))
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

        zero = Add(
            _contraction(Fd(p), F(q)),
            -Mul(KroneckerDelta(p, q),
                 KroneckerDelta(q, Index("i", below_fermi=True)))
        )
        zero = ExprContainer(zero, target_idx="").substitute_contracted()
        assert zero.inner is S.Zero
        zero = Add(
            _contraction(F(p), Fd(q)),
            -Mul(KroneckerDelta(p, q),
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

    def test_contract_operator_string(self):
        i, j, a, b, p, q = get_symbols("ijabpq")

        op_string = [Fd(i), F(a), Fd(b), F(j)]
        ref = Mul(KroneckerDelta(i, j), KroneckerDelta(a, b))
        assert _contract_operator_string(op_string) == ref

        op_string = [Fd(i), F(a), Fd(p), F(q), Fd(b), F(j)]
        ref = Add(
            -Mul(KroneckerDelta(i, q), KroneckerDelta(a, b),
                 KroneckerDelta(p, j)),
            Mul(KroneckerDelta(i, j), KroneckerDelta(a, p),
                KroneckerDelta(q, b)),
            Mul(KroneckerDelta(i, j), KroneckerDelta(a, b),
                KroneckerDelta(p, q),
                KroneckerDelta(q, Index("i", below_fermi=True)))
        )
        res = _contract_operator_string(op_string).expand()
        zero = ExprContainer(
            Add(ref, -res), target_idx="ijab"
        ).substitute_contracted()
        assert zero.inner is S.Zero

    def test_wicks(self):
        i, j, a, b, p, q = get_symbols("ijabpq")

        expr = Mul(
            Fd(i), F(a), Fd(p), F(q), Fd(b), F(j), 2, KroneckerDelta(i, j)
        )
        ref = S.Zero
        ref -= Mul(
            KroneckerDelta(i, q), KroneckerDelta(a, b), KroneckerDelta(p, j)
        )
        ref += Mul(
            KroneckerDelta(i, j), KroneckerDelta(a, p), KroneckerDelta(q, b)
        )
        ref += Mul(
            KroneckerDelta(i, j), KroneckerDelta(a, b), KroneckerDelta(p, q),
            KroneckerDelta(q, Index("i", below_fermi=True))
        )
        ref *= Mul(2, KroneckerDelta(i, j))
        res = wicks(expr)
        zero = ExprContainer(
            ref.expand() - res, target_idx="ijab"
        ).substitute_contracted()
        assert zero.inner is S.Zero
