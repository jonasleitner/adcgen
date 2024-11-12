from adcgen.core_valence_separation import (
    introduce_core_target_indices
)
from adcgen.expr_container import Expr
from adcgen.indices import get_symbols
from adcgen.misc import Inputerror
from adcgen.sympy_objects import AntiSymmetricTensor

from sympy import S

import pytest


class TestCoreValenceSeparation:
    def test_introduce_core_target_indices(self):
        # without spin
        i, j, a, b = get_symbols("ijab")
        tensor = AntiSymmetricTensor("t", (i, j), (a, b))
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
