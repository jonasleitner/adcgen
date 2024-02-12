from sympy_adc.indices import Index
from sympy_adc.sympy_objects import KroneckerDelta
from sympy import S


class TestKroneckerDelta:
    def test_evaluation(self):
        i, j = Index("i"), Index("j")
        assert KroneckerDelta(i, j) - KroneckerDelta(j, i) is S.Zero
        assert KroneckerDelta(i, j) is not S.Zero

        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        assert KroneckerDelta(i, j) is not S.Zero
        a = Index("a", above_fermi=True)
        assert KroneckerDelta(i, a) is S.Zero
        p, q = Index("p", alpha=True), Index("q", beta=True)
        assert KroneckerDelta(p, q) is S.Zero

    def test_preferred_killable_index(self):
        p, q = Index("p"), Index("q")
        pa, qa = Index("p", alpha=True), Index("q", alpha=True)
        i = Index("i", below_fermi=True)
        ia = Index("i", below_fermi=True, alpha=True)

        assert KroneckerDelta(q, p).preferred_and_killable == (p, q)
        assert KroneckerDelta(pa, p).preferred_and_killable == (pa, p)
        assert KroneckerDelta(pa, qa).preferred_and_killable == (pa, qa)
        assert KroneckerDelta(i, p).preferred_and_killable == (i, p)
        assert KroneckerDelta(ia, p).preferred_and_killable == (ia, p)
        assert KroneckerDelta(ia, pa).preferred_and_killable == (ia, pa)
        assert KroneckerDelta(i, pa).preferred_and_killable is None
