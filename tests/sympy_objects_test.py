from sympy_adc.indices import Index
from sympy_adc.sympy_objects import Delta
from sympy import S


class TestDelta:
    def test_evaluation(self):
        i, j = Index("i"), Index("j")
        assert Delta(i, j) - Delta(j, i) is S.Zero
        assert Delta(i, j) is not S.Zero

        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        assert Delta(i, j) is not S.Zero
        a = Index("a", above_fermi=True)
        assert Delta(i, a) is S.Zero
        p, q = Index("p", alpha=True), Index("q", beta=True)
        assert Delta(p, q) is S.Zero

    def test_preferred_index(self):
        p, q = Index("p"), Index("q")
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)

        assert Delta(q, p).preferred_index is p
        assert Delta(j, i).preferred_index is i
        assert Delta(a, b).preferred_index is a
        assert Delta(i, p).preferred_index is i
        assert Delta(a, p).preferred_index is a
