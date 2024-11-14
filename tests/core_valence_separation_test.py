from adcgen.core_valence_separation import (
    introduce_core_target_indices, expand_occupied_indices,
    is_allowed_cvs_block, allowed_cvs_blocks
)
from adcgen.expr_container import Expr
from adcgen.indices import get_symbols
from adcgen.intermediates import Intermediates, RegisteredIntermediate
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

    def test_allowed_cvs_blocks(self):
        i, j, k, l = get_symbols("ijkl")  # noqa E741
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

    @pytest.mark.parametrize("amplitude", ["t2_1", "t1_2", "t2_2", "t3_2",
                                           "t4_2", "t1_3", "t2_3"])
    def test_allowed_t_amplitude_cvs_blocks(self, amplitude):
        ampl: RegisteredIntermediate = Intermediates().available.get(amplitude)
        tensor: Expr = ampl.tensor()
        # ov/oovv/... is a valid block
        assert is_allowed_cvs_block(tensor.terms[0].objects[0])
        # ov/oovv/... is the only valid block
        target: Amplitude = tensor.sympy.idx
        ref = ("".join(s.space[0] for s in target),)  # = ov/oovv/...
        res = allowed_cvs_blocks(tensor, target,
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ref
        # ensure that we get the same result if we expand the amplitude
        res = ampl.allowed_cvs_blocks(is_allowed_cvs_block)
        assert res == ref
