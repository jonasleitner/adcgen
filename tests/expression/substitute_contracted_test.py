from adcgen import ExprContainer, get_symbols, NonSymmetricTensor
from sympy import S


class TestSubstituteContracted:
    def test_no_contracted(self):
        i, j = get_symbols("ij")
        expr = NonSymmetricTensor("V", tuple())
        test = ExprContainer(expr).substitute_contracted()
        assert test.inner - S.One * expr is S.Zero
        expr = NonSymmetricTensor("V", (i, j))
        test = ExprContainer(expr).substitute_contracted()
        assert test.inner - S.One * expr is S.Zero
        expr = NonSymmetricTensor("V", (i, i, j))
        test = ExprContainer(expr, target_idx="ij").substitute_contracted()
        assert test.inner - S.One * expr is S.Zero

    def test_spin_orbitals(self):
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        # continuous indices
        expr = NonSymmetricTensor("V", (j, k, l))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (i, j, k)) is S.Zero  # noqa E501
        # with a target index
        test = ExprContainer(expr, target_idx="k").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (i, k, j)) is S.Zero  # noqa E501
        # with a gap
        expr = NonSymmetricTensor("V", (j, l))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (i, j)) is S.Zero
        test = ExprContainer(expr, target_idx="j").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (j, i)) is S.Zero
        # repeated indices
        expr = NonSymmetricTensor("V", (j, l, l, j))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (i, j, j, i)) is S.Zero  # noqa E501
        # ill defined target indices
        expr = NonSymmetricTensor("V", (l,))
        test = ExprContainer(expr, target_idx="ij").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (k,)) is S.Zero

    def test_spatial_orbitals(self):
        ia, ja, ka, ib, jb, kb = get_symbols("ijkijk", "aaabbb")
        # indices with the same spin should behave like spin orbitals
        expr = NonSymmetricTensor("V", (ia, ka))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (ia, ja)) is S.Zero
        expr = NonSymmetricTensor("V", (ja, ka))
        test = ExprContainer(expr, target_idx=[ja]).substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (ja, ia)) is S.Zero
        # indices with different spin: each spin is minized independently
        expr = NonSymmetricTensor("V", (ia, ka, jb))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (ia, ja, ib)) is S.Zero  # noqa E501
        expr = NonSymmetricTensor("V", (kb, ka, jb))
        test = ExprContainer(expr, target_idx=[ka, jb]).substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (ib, ka, jb)) is S.Zero  # noqa E501
        expr = NonSymmetricTensor("V", (kb, ka))
        test = ExprContainer(expr, target_idx="").substitute_contracted()
        assert test.inner - S.One * NonSymmetricTensor("V", (ib, ia)) is S.Zero
