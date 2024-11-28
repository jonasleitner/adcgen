from adcgen.indices import Index
from adcgen.sympy_objects import (
    KroneckerDelta, AntiSymmetricTensor, SymmetricTensor
)
from sympy import S


class TestAntiSymmetricTensor:
    def test_symmetry(self):
        p, q, r, s = Index("p"), Index("q"), Index("r"), Index("s")
        tensor = AntiSymmetricTensor("V", (p, q), (r, s))
        assert AntiSymmetricTensor("V", (p, p), (r, s)) is S.Zero
        assert tensor + AntiSymmetricTensor("V", (q, p), (r, s)) is S.Zero
        assert tensor + AntiSymmetricTensor("V", (p, q), (s, r)) is S.Zero
        assert tensor - AntiSymmetricTensor("V", (q, p), (s, r)) is S.Zero
        i = Index("i", below_fermi=True)
        ia = Index("i", below_fermi=True, alpha=True)
        ib = Index("i", below_fermi=True, beta=True)
        a = Index("a", above_fermi=True)
        aa = Index("a", above_fermi=True, alpha=True)
        ab = Index("a", above_fermi=True, beta=True)
        pa = Index("p", alpha=True)
        pb = Index("p", beta=True)
        I = Index("I", core=True)  # noqa E741
        Ia = Index("I", core=True, alpha=True)
        Ib = Index("I", core=True, beta=True)
        ref = AntiSymmetricTensor(
            "V", tuple(), (p, pa, pb, i, ia, ib, I, Ia, Ib, a, aa, ab))
        res = AntiSymmetricTensor(
            "V", tuple(), (Ib, ia, I, ib, ab, aa, pb, a, p, Ia, pa, i))
        assert ref - res is S.Zero

    def test_bra_ket_symmetry(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        test = AntiSymmetricTensor("V", (i, j), (a, b), 1)
        assert test - AntiSymmetricTensor("V", (a, b), (i, j), 1) is S.Zero
        assert test.idx == (i, j, a, b)
        test = AntiSymmetricTensor("V", (i, j), (a, b), -1)
        assert test + AntiSymmetricTensor("V", (a, b), (i, j), -1) is S.Zero
        assert test.idx == (i, j, a, b)
        k, l = Index("k", below_fermi=True), Index("l", below_fermi=True)  # noqa E741
        test = AntiSymmetricTensor("V", (i, j), (k, l), 1)
        assert test - AntiSymmetricTensor("V", (k, l), (i, j), 1) is S.Zero
        assert test.idx == (i, j, k, l)
        ia = Index("i", below_fermi=True, alpha=True)
        ib = Index("i", below_fermi=True, alpha=True)
        test = AntiSymmetricTensor("V", (i, ia), (k, l), 1)
        assert test - AntiSymmetricTensor("V", (k, l), (i, ia), 1) is S.Zero
        assert test.idx == (k, l, i, ia)
        test = AntiSymmetricTensor("V", (i, j), (ia, ib), 1)
        assert test - AntiSymmetricTensor("V", (ia, ib), (i, j), 1) is S.Zero
        assert test.idx == (i, j, ia, ib)
        test = AntiSymmetricTensor("V", (i, j, ib), (i, ia, ib), 1)
        assert (test -
                AntiSymmetricTensor("V", (i, ia, ib), (i, j, ib), 1) is S.Zero)
        assert test.idx == (i, j, ib, i, ia, ib)
        I = Index("I", core=True)  # noqa E741
        test = AntiSymmetricTensor("d", (I,), (i,), 1)  # co -> oc
        assert test - AntiSymmetricTensor("d", (i,), (I,), 1) is S.Zero
        assert test.idx == (i, I)


class TestSymmetricTensor:
    def test_symmetry(self):
        p, q, r, s = Index("p"), Index("q"), Index("r"), Index("s")
        tensor = SymmetricTensor("V", (p, q), (r, s))
        assert tensor - SymmetricTensor("V", (q, p), (r, s)) is S.Zero
        assert tensor - SymmetricTensor("V", (p, q), (s, r)) is S.Zero
        assert tensor - SymmetricTensor("V", (q, p), (s, r)) is S.Zero
        i = Index("i", below_fermi=True)
        ia = Index("i", below_fermi=True, alpha=True)
        ib = Index("i", below_fermi=True, beta=True)
        a = Index("a", above_fermi=True)
        aa = Index("a", above_fermi=True, alpha=True)
        ab = Index("a", above_fermi=True, beta=True)
        pa = Index("p", alpha=True)
        pb = Index("p", beta=True)
        ref = SymmetricTensor("V", tuple(), (p, pa, pb, i, ia, ib, a, aa, ab))
        res = SymmetricTensor("V", tuple(), (ia, ib, ab, aa, pb, a, p, pa, i))
        assert ref - res is S.Zero

    def test_bra_ket_symmetry(self):
        i, j = Index("i", below_fermi=True), Index("j", below_fermi=True)
        a, b = Index("a", above_fermi=True), Index("b", above_fermi=True)
        assert SymmetricTensor("V", (i, j), (a, b), 1) - \
            SymmetricTensor("V", (a, b), (i, j), 1) is S.Zero
        assert SymmetricTensor("V", (i, j), (a, b), -1) + \
            SymmetricTensor("V", (a, b), (i, j), -1) is S.Zero
        k, l = Index("k", below_fermi=True), Index("l", below_fermi=True)  # noqa E741
        assert SymmetricTensor("V", (i, j), (k, l), 1) - \
            SymmetricTensor("V", (k, l), (i, j), 1) is S.Zero
        ia = Index("i", below_fermi=True, alpha=True)
        ib = Index("i", below_fermi=True, alpha=True)
        assert SymmetricTensor("V", (i, ia), (k, l), 1) - \
            SymmetricTensor("V", (k, l), (i, ia), 1) is S.Zero
        assert SymmetricTensor("V", (i, j), (ia, ib), 1) - \
            SymmetricTensor("V", (ia, ib), (i, j), 1) is S.Zero


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
