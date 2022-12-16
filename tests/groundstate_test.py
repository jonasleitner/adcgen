from sympy_adc.expr_container import expr
from sympy_adc.simplify import simplify, extract_dm

from sympy import S

import pytest


@pytest.mark.parametrize('order', [0, 1, 2])
class TestGroundState():
    @pytest.mark.parametrize('operator', ['ca'])
    def test_expectation_value(self, order: int, operator: str, cls_instances,
                               reference_data):
        # load the reference data
        ref = reference_data["mp_1p_dm"][order]
        # compute the expectation value
        expec = cls_instances.gs.expectation_value(order, operator)
        expec = expr(expec)
        ref_expec = ref['expec_val']
        assert simplify(ref_expec - expec).sympy is S.Zero

        # assume a real basis and a symmetric operator/a symmetric dm
        expec = expec.substitute_contracted()
        expec = expr(expec.sympy, real=True, sym_tensors=['d'])
        expec = simplify(expec)
        ref_expec = ref['real_sym_expec_val']
        ref_expec = expr(ref_expec.sympy, **expec.assumptions)
        assert simplify(ref_expec - expec).sympy is S.Zero

        # extract all blocks of the symmetric dm
        density_matrix = extract_dm(expec)
        for block, dm in density_matrix.items():
            assert len(block) == 1
            ref_dm = ref["real_sym_dm"][block[0]]
            ref_dm = expr(ref_dm, **dm.assumptions)
            assert simplify(dm - ref_dm, True).sympy is S.Zero
