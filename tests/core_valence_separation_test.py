from adcgen.core_valence_separation import (
    introduce_core_target_indices, expand_occupied_indices,
    is_allowed_cvs_block, allowed_cvs_blocks
)
from adcgen.expr_container import Expr
from adcgen.indices import get_symbols
from adcgen.misc import Inputerror
from adcgen.sympy_objects import (
    AntiSymmetricTensor, SymmetricTensor, Amplitude
)
from adcgen.tensor_names import tensor_names

from sympy import S

import pytest


class TestCoreValenceSeparation:
    def test_introduce_core_target_indices(self):
        # without spin
        i, j, a, b = get_symbols("ijab")
        tensor = AntiSymmetricTensor("t", (i, j), (a, b))
        # no core indices to introduce -> leave input unchanged
        res = introduce_core_target_indices(Expr(tensor), "").sympy
        assert res - tensor is S.Zero
        res = introduce_core_target_indices(Expr(tensor), "I").sympy
        I, J = get_symbols("IJ")
        ref = AntiSymmetricTensor("t", (I, j), (a, b))
        assert res - ref is S.Zero
        res = introduce_core_target_indices(Expr(tensor), "IJ").sympy
        ref = AntiSymmetricTensor("t", (I, J), (a, b))
        assert res - ref is S.Zero
        # with spin
        i, j, a, b = get_symbols("ijab", "abab")
        tensor = AntiSymmetricTensor("t", (i, j), (a, b))
        with pytest.raises(ValueError):
            # can't find corresponding occ index -> different spin
            introduce_core_target_indices(Expr(tensor), "I", "b")
        I, J = get_symbols("IJ", "ab")
        # also ensure that the set target indices are updated
        test = Expr(tensor, target_idx=[i, a])
        res = introduce_core_target_indices(test.copy(), "I", spin="a")
        ref = AntiSymmetricTensor("t", (I, j), (a, b))
        assert res.sympy - ref is S.Zero
        assert res.provided_target_idx == (I, a)
        test.set_target_idx([i, j])
        res = introduce_core_target_indices(test.copy(), "IJ", spin="ab")
        ref = AntiSymmetricTensor("t", (I, J), (a, b))
        assert res.sympy - ref is S.Zero
        assert res.provided_target_idx == (I, J)

        with pytest.raises(ValueError):
            # can't find corresponding occ index -> no matching name
            introduce_core_target_indices(Expr(tensor), "K")
        with pytest.raises(Inputerror):  # invalid core index
            introduce_core_target_indices(Expr(tensor), "a")
        with pytest.raises(Inputerror):  # invalid core index
            introduce_core_target_indices(Expr(tensor), "i")
        with pytest.raises(Inputerror):  # invalid core index
            introduce_core_target_indices(Expr(tensor), "p")

    def test_expand_occupied_indices(self):
        # trivial case: allow all blocks (no CVS approximation)
        i, j, a, b = get_symbols("ijab")
        I, J = get_symbols("IJ")
        tensor = AntiSymmetricTensor("V", (i, j), (a, b))
        res = expand_occupied_indices(Expr(tensor, target_idx=""))
        ref = (AntiSymmetricTensor("V", (i, j), (a, b)) +
               AntiSymmetricTensor("V", (I, j), (a, b)) +
               AntiSymmetricTensor("V", (i, J), (a, b)) +
               AntiSymmetricTensor("V", (I, J), (a, b)))
        assert res.sympy - ref is S.Zero
        # even more trivial: no occupied contracted indices
        res = expand_occupied_indices(Expr(tensor, target_idx="ij"))
        assert res.sympy - tensor is S.Zero

    def test_is_allowed_cvs_block(self):
        i, j, k, l, I, J, K, L = get_symbols("ijklIJKL")
        coulomb = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        assert is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])
        coulomb = SymmetricTensor(tensor_names.coulomb, (I, j), (k, l))
        assert not is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])
        coulomb = SymmetricTensor(tensor_names.coulomb, (I, J), (k, l))
        assert is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])
        coulomb = SymmetricTensor(tensor_names.coulomb, (I, j), (K, l))
        assert not is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])
        coulomb = SymmetricTensor(tensor_names.coulomb, (I, J), (K, l))
        assert not is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])
        coulomb = SymmetricTensor(tensor_names.coulomb, (I, J), (K, L))
        assert is_allowed_cvs_block(Expr(coulomb).terms[0].objects[0])

        eri = AntiSymmetricTensor(tensor_names.eri, (i, j), (k, l))
        assert is_allowed_cvs_block(Expr(eri).terms[0].objects[0])
        eri = AntiSymmetricTensor(tensor_names.eri, (i, J), (k, l))
        assert not is_allowed_cvs_block(Expr(eri).terms[0].objects[0])
        eri = AntiSymmetricTensor(tensor_names.eri, (I, J), (k, l))
        assert not is_allowed_cvs_block(Expr(eri).terms[0].objects[0])
        eri = AntiSymmetricTensor(tensor_names.eri, (i, J), (k, L))
        assert is_allowed_cvs_block(Expr(eri).terms[0].objects[0])
        eri = AntiSymmetricTensor(tensor_names.eri, (I, J), (k, L))
        assert not is_allowed_cvs_block(Expr(eri).terms[0].objects[0])
        eri = AntiSymmetricTensor(tensor_names.eri, (I, J), (K, L))
        assert is_allowed_cvs_block(Expr(eri).terms[0].objects[0])

    def test_allowed_cvs_blocks(self):
        i, j, k, l, a, b = get_symbols("ijklab")  # noqa E741
        # Coulomb
        coulomb = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        res = allowed_cvs_blocks(Expr(coulomb), "ijkl",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oooo", "oocc", "ccoo", "cccc")
        # ERI
        eri = AntiSymmetricTensor(tensor_names.eri, (i, j), (k, l))
        res = allowed_cvs_blocks(Expr(eri), "ijkl",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oooo", "ococ", "occo", "cooc", "coco", "cccc")
        # MP amplitudes
        t2_1 = Amplitude("t1", (a, b), (i, j))
        res = allowed_cvs_blocks(Expr(t2_1), "ijab",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oovv",)
        t1_2 = Amplitude("t2", (a,), (i,))
        res = allowed_cvs_blocks(Expr(t1_2), "ia",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("ov",)
        t2_2 = Amplitude("t2", (a, b), (i, j))
        res = allowed_cvs_blocks(Expr(t2_2), "ijab",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oovv",)
