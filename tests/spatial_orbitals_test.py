from sympy_adc.expr_container import Expr
from sympy_adc.indices import Index, get_symbols
from sympy_adc.intermediates import Intermediates
from sympy_adc.simplify import simplify
from sympy_adc.spatial_orbitals import integrate_spin
from sympy_adc.sympy_objects import AntiSymmetricTensor

from sympy import S, Rational


class TestIntegrateSpin:
    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        t2 = t2.expand_itmd(fully_expand=False)
        res = integrate_spin(t2, 'ijab', "aaaa")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "aaaa"))
        res = integrate_spin(t2, 'ijab', "aaab")
        assert res.sympy is S.Zero
        res = integrate_spin(t2, 'ijab', "aabb")
        assert res.sympy is S.Zero
        res = integrate_spin(t2, 'ijab', "abab")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "abab"))
        res = integrate_spin(t2, 'ijab', "baab")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "baab"))

    def test_t1_2(self):
        # use one of the 2 t1_2 terms as test case (without denominator)
        i, j, a, b, c = get_symbols('ijabc')
        term = Rational(1, 2) * AntiSymmetricTensor("t1", (b, c), (i, j)) * \
            AntiSymmetricTensor("V", (j, a), (b, c))
        term = Expr(term, real=True)
        # case 1: aa
        res = integrate_spin(term, 'ia', 'aa')
        ia, ja, aa, ba, ca = get_symbols('ijabc', 'aaaaa')
        jb, cb = get_symbols('jc', 'bb')
        ref = Expr(
            Rational(1, 2) * AntiSymmetricTensor("t1", (ba, ca), (ia, ja)) *
            AntiSymmetricTensor("V", (ja, aa), (ba, ca))
            + AntiSymmetricTensor("t1", (ba, cb), (ia, jb)) *
            AntiSymmetricTensor("V", (jb, aa), (ba, cb))
        ).make_real()
        assert simplify(res - ref).sympy is S.Zero
        # case 2: ab
        assert integrate_spin(term, 'ia', 'ab').sympy is S.Zero
        # case 3: ba
        assert integrate_spin(term, 'ia', 'ba').sympy is S.Zero
        # case 4: bb
        res = integrate_spin(term, 'ia', 'bb')
        ib, jb, ab, bb, cb = get_symbols('ijabc', 'bbbbb')
        ref = Expr(
            Rational(1, 2) * AntiSymmetricTensor("t1", (bb, cb), (ib, jb)) *
            AntiSymmetricTensor("V", (jb, ab), (bb, cb))
            + AntiSymmetricTensor("t1", (ba, cb), (ib, ja)) *
            AntiSymmetricTensor("V", (ja, ab), (ba, cb))
        ).make_real()
        assert simplify(res - ref).sympy is S.Zero


class TestAllowedSpinBlocks:
    def test_t2_1(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_1"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref

    def test_t1_2(self):
        ref = ("aa", "bb")
        t1 = Intermediates().available["t1_2"]
        assert t1.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t1.allowed_spin_blocks == ref

    def test_t2_2(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_2"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref

    def test_t3_2(self):
        ref = ("aaaaaa", "aabaab", "aababa", "aabbaa", "abaaab", "abaaba",
               "ababaa", "baaaab", "baaaba", "baabaa", "abbabb", "abbbab",
               "abbbba", "bbabba", "bbabab", "bbaabb", "babbba", "babbab",
               "bababb", "bbbbbb")
        ref = tuple(sorted(ref))
        t3 = Intermediates().available["t3_2"]
        assert t3.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t3.allowed_spin_blocks == ref

    def test_t4_2(self):
        t4 = Intermediates().available["t4_2"]
        sb1 = t4.tensor().terms[0].objects[0].allowed_spin_blocks
        sb2 = t4.allowed_spin_blocks
        assert sb1 == sb2

    def test_t1_3(self):
        ref = ("aa", "bb")
        t1 = Intermediates().available["t1_3"]
        assert t1.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t1.allowed_spin_blocks == ref

    def test_t2_3(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_3"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref
