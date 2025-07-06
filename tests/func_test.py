from adcgen.func import evaluate_deltas
from adcgen.indices import Index
from adcgen.sympy_objects import KroneckerDelta

from sympy import Mul
from sympy.physics.secondquant import F


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
