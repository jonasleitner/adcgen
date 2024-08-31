from adcgen import Expr, get_symbols, NonSymmetricTensor
from sympy import S


class TestSubstituteContracted:
    def test_no_contracted(self):
        i, j = get_symbols("ij")
        expr = NonSymmetricTensor("V", tuple())
        test = Expr(expr).substitute_contracted()
        assert test.sympy - expr is S.Zero
        expr = NonSymmetricTensor("V", (i, j))
        test = Expr(expr).substitute_contracted()
        assert test.sympy - expr is S.Zero
        expr = NonSymmetricTensor("V", (i, i, j))
        test = Expr(expr, target_idx="ij").substitute_contracted()
        assert test.sympy - expr is S.Zero

    def test_spin_orbitals(self):
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        # continuous indices
        expr = NonSymmetricTensor("V", (j, k, l))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (i, j, k)) is S.Zero
        # with a target index
        test = Expr(expr, target_idx="k").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (i, k, j)) is S.Zero
        # with a gap
        expr = NonSymmetricTensor("V", (j, l))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (i, j)) is S.Zero
        test = Expr(expr, target_idx="j").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (j, i)) is S.Zero
        # repeated indices
        expr = NonSymmetricTensor("V", (j, l, l, j))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (i, j, j, i)) is S.Zero
        # ill defined target indices
        expr = NonSymmetricTensor("V", (l,))
        test = Expr(expr, target_idx="ij").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (k,)) is S.Zero

    def test_spatial_orbitals(self):
        ia, ja, ka, ib, jb, kb = get_symbols("ijkijk", "aaabbb")
        # indices with the same spin should behave like spin orbitals
        expr = NonSymmetricTensor("V", (ia, ka))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (ia, ja)) is S.Zero
        expr = NonSymmetricTensor("V", (ja, ka))
        test = Expr(expr, target_idx=[ja]).substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (ja, ia)) is S.Zero
        # indices with different spin: each spin is minized independently
        expr = NonSymmetricTensor("V", (ia, ka, jb))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (ia, ja, ib)) is S.Zero
        expr = NonSymmetricTensor("V", (kb, ka, jb))
        test = Expr(expr, target_idx=[ka, jb]).substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (ib, ka, jb)) is S.Zero
        expr = NonSymmetricTensor("V", (kb, ka))
        test = Expr(expr, target_idx="").substitute_contracted()
        assert test.sympy - NonSymmetricTensor("V", (ib, ia)) is S.Zero
