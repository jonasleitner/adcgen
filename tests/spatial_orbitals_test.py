from adcgen.expr_container import Expr
from adcgen.indices import Index, get_symbols
from adcgen.intermediates import Intermediates
from adcgen.simplify import simplify
from adcgen.spatial_orbitals import (
    integrate_spin, transform_to_spatial_orbitals
)
from adcgen.sympy_objects import (
    AntiSymmetricTensor, SymmetricTensor, NonSymmetricTensor, KroneckerDelta,
    Amplitude
)
from adcgen.tensor_names import tensor_names

from sympy import S, Rational
from sympy.physics.secondquant import F, Fd


class TestExpandAntiSymEri:
    def test_no_eri(self):
        test = Expr(AntiSymmetricTensor("d", tuple(), tuple())).make_real()
        test = test.expand_antisym_eri()
        assert tensor_names.coulomb not in test.sym_tensors

    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        t2: Expr = t2.expand_itmd(fully_expand=False).make_real()
        res = t2.expand_antisym_eri()
        assert tensor_names.coulomb in res.sym_tensors
        i, j, a, b = get_symbols('ijab')
        ref = (SymmetricTensor(tensor_names.coulomb, (i, a), (j, b), 1)
               - SymmetricTensor(tensor_names.coulomb, (i, b), (j, a), 1))
        ref /= (NonSymmetricTensor(tensor_names.orb_energy, (a,))
                + NonSymmetricTensor(tensor_names.orb_energy, (b,))
                - NonSymmetricTensor(tensor_names.orb_energy, (i,))
                - NonSymmetricTensor(tensor_names.orb_energy, (j,)))
        assert ref - res.sympy is S.Zero

    def test_t1_2(self):
        t1 = Intermediates().available["t1_2"]
        t1: Expr = t1.expand_itmd(fully_expand=False).make_real()
        res = t1.expand_antisym_eri().substitute_contracted()
        i, j, k, a, b, c = get_symbols("ijkabc")
        ref = (Rational(1, 2) *
               Amplitude(f"{tensor_names.gs_amplitude}1", (b, c), (i, j)) * (
                   SymmetricTensor(tensor_names.coulomb, (j, b), (a, c), 1)
                   - SymmetricTensor(tensor_names.coulomb, (j, c), (a, b), 1)
               ))
        ref += (Rational(1, 2) *
                Amplitude(f"{tensor_names.gs_amplitude}1", (a, b), (j, k))
                * (SymmetricTensor(tensor_names.coulomb, (j, i), (k, b), 1)
                   - SymmetricTensor(tensor_names.coulomb, (j, b), (k, i), 1)))
        ref /= (NonSymmetricTensor(tensor_names.orb_energy, (i,))
                - NonSymmetricTensor(tensor_names.orb_energy, (a,)))
        assert res.sympy.expand() - ref.expand() is S.Zero


class TestIntegrateSpin:
    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        t2 = t2.expand_itmd(fully_expand=False)
        res = integrate_spin(t2, 'ijab', "aaaa")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "aaaa"))
        assert set(res.provided_target_idx) == set(get_symbols("ijab", "aaaa"))
        res = integrate_spin(t2, 'ijab', "aaab")
        assert res.sympy is S.Zero
        assert set(res.provided_target_idx) == set(get_symbols("ijab", "aaab"))
        res = integrate_spin(t2, 'ijab', "aabb")
        assert res.sympy is S.Zero
        assert set(res.provided_target_idx) == set(get_symbols("ijab", "aabb"))
        res = integrate_spin(t2, 'ijab', "abab")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "abab"))
        assert set(res.provided_target_idx) == set(get_symbols("ijab", "abab"))
        res = integrate_spin(t2, 'ijab', "baab")
        assert res.sympy.atoms(Index) == set(get_symbols("ijab", "baab"))
        assert set(res.provided_target_idx) == set(get_symbols("ijab", "baab"))

    def test_number(self):
        # ensure that numbers are pure number terms are not dropped
        # during the spin integration
        num = 42
        assert integrate_spin(Expr(num), "", "").sympy - num is S.Zero
        i, j, a, b = get_symbols('ijab')
        ia, ja, aa, ba = get_symbols('ijab', "aaaa")
        ib, jb, ab, bb = get_symbols('ijab', "bbbb")
        tensors = (Rational(1, 4) *
                   AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b)) *
                   Amplitude(f"{tensor_names.gs_amplitude}1", (a, b), (i, j)))
        test = Expr(tensors + num)
        res = integrate_spin(test, "", "")
        ref = tensors.subs({i: ia, j: ja, a: aa, b: ba})
        ref += 4 * tensors.subs({i: ia, j: jb, a: aa, b: bb})
        ref += tensors.subs({i: ib, j: jb, a: ab, b: bb})
        ref += num
        assert res.sympy - ref is S.Zero
        assert res.provided_target_idx is None

    def test_t1_2(self):
        # use one of the 2 t1_2 terms as test case (without denominator)
        i, j, a, b, c = get_symbols('ijabc')
        term = (Rational(1, 2) *
                Amplitude(f"{tensor_names.gs_amplitude}1", (b, c), (i, j)) *
                AntiSymmetricTensor(tensor_names.eri, (j, a), (b, c)))
        term = Expr(term, real=True)
        # case 1: aa
        res = integrate_spin(term, 'ia', 'aa')
        ia, ja, aa, ba, ca = get_symbols('ijabc', 'aaaaa')
        jb, cb = get_symbols('jc', 'bb')
        ref = Expr(
            Rational(1, 2) *
            Amplitude(f"{tensor_names.gs_amplitude}1", (ba, ca), (ia, ja)) *
            AntiSymmetricTensor(tensor_names.eri, (ja, aa), (ba, ca))
            + Amplitude(f"{tensor_names.gs_amplitude}1", (ba, cb), (ia, jb)) *
            AntiSymmetricTensor(tensor_names.eri, (jb, aa), (ba, cb))
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
            Rational(1, 2) *
            Amplitude(f"{tensor_names.gs_amplitude}1", (bb, cb), (ib, jb)) *
            AntiSymmetricTensor(tensor_names.eri, (jb, ab), (bb, cb))
            + Amplitude(f"{tensor_names.gs_amplitude}1", (ba, cb), (ib, ja)) *
            AntiSymmetricTensor(tensor_names.eri, (ja, ab), (ba, cb))
        ).make_real()
        assert simplify(res - ref).sympy is S.Zero


class TestAllowedSpinBlocks:
    def test_single_objects(self):
        p, q, r, s = get_symbols("pqrs")
        # ERI
        obj = Expr(AntiSymmetricTensor(tensor_names.eri, (p, q), (r, s)))
        obj = obj.terms[0].objects[0]
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        assert obj.allowed_spin_blocks == ref
        # ERI: chemist notation
        obj = Expr(SymmetricTensor(tensor_names.coulomb, (p, q), (r, s)))
        obj = obj.terms[0].objects[0]
        ref = ("aaaa", "aabb", "bbaa", "bbbb")
        assert obj.allowed_spin_blocks == ref
        # delta
        obj = Expr(KroneckerDelta(p, q)).terms[0].objects[0]
        ref = ("aa", "bb")
        assert obj.allowed_spin_blocks == ref
        # create / annihilate
        for obj in Expr(F(p) * Fd(q)).terms[0].objects:
            assert obj.allowed_spin_blocks == ("a", "b")

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


class TestTransformToSpatialOrbitals:
    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        t2: Expr = t2.expand_itmd(fully_expand=False).make_real()
        # unrestricted:
        res = transform_to_spatial_orbitals(t2, "ijab", "abab",
                                            restricted=False)
        i, j, a, b = get_symbols("ijab", "abab")
        ref = SymmetricTensor(tensor_names.coulomb, (i, a), (j, b), 1) / (
            NonSymmetricTensor(tensor_names.orb_energy, (a,))
            + NonSymmetricTensor(tensor_names.orb_energy, (b,))
            - NonSymmetricTensor(tensor_names.orb_energy, (i,))
            - NonSymmetricTensor(tensor_names.orb_energy, (j,))
        )
        assert res.sympy - ref is S.Zero
        assert set(res.provided_target_idx) == {i, j, a, b}
        # restricted
        res = transform_to_spatial_orbitals(t2, "ijab", "abab",
                                            restricted=True)
        i, j, a, b = get_symbols("ijab", "aaaa")
        ref = SymmetricTensor(tensor_names.coulomb, (i, a), (j, b), 1) / (
            NonSymmetricTensor(tensor_names.orb_energy, (a,))
            + NonSymmetricTensor(tensor_names.orb_energy, (b,))
            - NonSymmetricTensor(tensor_names.orb_energy, (i,))
            - NonSymmetricTensor(tensor_names.orb_energy, (j,))
        )
        assert res.sympy - ref is S.Zero
        assert set(res.provided_target_idx) == {i, j, a, b}

    def test_t1_2(self):
        t1 = Intermediates().available["t1_2"]
        t1: Expr = t1.expand_itmd(fully_expand=True).make_real()
        t1.substitute_contracted().use_symbolic_denominators()
        # unrestricted
        unrestricted = transform_to_spatial_orbitals(t1, "ia", "aa",
                                                     restricted=False)
        unrestricted = simplify(unrestricted)
        i, j, k, a, b, c = get_symbols("ijkabc", "aaaaaa")
        jb, kb, bb = get_symbols("jkb", "bbb")
        ref = (
            SymmetricTensor(tensor_names.sym_orb_denom, (b, c), (i, j), -1) *
            SymmetricTensor(tensor_names.coulomb, (j, b), (a, c), 1) * (
                SymmetricTensor(tensor_names.coulomb, (i, b), (j, c), 1)
                - SymmetricTensor(tensor_names.coulomb, (i, c), (j, b), 1)
            )
            - SymmetricTensor(tensor_names.sym_orb_denom, (bb, c), (i, jb),
                              -1) *
            SymmetricTensor(tensor_names.coulomb, (jb, bb), (a, c), 1) *
            SymmetricTensor(tensor_names.coulomb, (i, c), (jb, bb), 1)
            + SymmetricTensor(tensor_names.sym_orb_denom, (a, b), (j, k), -1) *
            SymmetricTensor(tensor_names.coulomb, (i, j), (k, b), 1) * (
                SymmetricTensor(tensor_names.coulomb, (j, a), (k, b), 1)
                - SymmetricTensor(tensor_names.coulomb, (j, b), (k, a), 1)
            )
            + SymmetricTensor(tensor_names.sym_orb_denom, (a, bb), (j, kb),
                              -1) *
            SymmetricTensor(tensor_names.coulomb, (i, j), (kb, bb), 1) *
            SymmetricTensor(tensor_names.coulomb, (j, a), (kb, bb), 1)
        )
        ref *= SymmetricTensor(tensor_names.sym_orb_denom, (i,), (a,), -1)
        assert simplify(unrestricted - ref.expand()).sympy is S.Zero
        assert set(unrestricted.provided_target_idx) == {i, a}
        # restricted
        restricted = transform_to_spatial_orbitals(t1, "ia", "aa",
                                                   restricted=True)
        restricted = simplify(restricted)
        ref = (
            SymmetricTensor(tensor_names.sym_orb_denom, (b, c), (i, j), -1) *
            SymmetricTensor(tensor_names.coulomb, (j, b), (a, c), 1) * (
                SymmetricTensor(tensor_names.coulomb, (i, b), (j, c), 1)
                - 2 * SymmetricTensor(tensor_names.coulomb, (i, c), (j, b), 1)
            )
            + SymmetricTensor(tensor_names.sym_orb_denom, (a, b), (j, k), -1) *
            SymmetricTensor(tensor_names.coulomb, (j, a), (k, b), 1) * (
                2 * SymmetricTensor(tensor_names.coulomb, (j, i), (k, b), 1)
                - SymmetricTensor(tensor_names.coulomb, (j, b), (k, i), 1)
            )) * SymmetricTensor(tensor_names.sym_orb_denom, (i,), (a,), -1)
        assert simplify(restricted - ref.expand()).sympy is S.Zero
        assert set(restricted.provided_target_idx) == {i, a}
