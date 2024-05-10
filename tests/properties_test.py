from adcgen import Expr, simplify, remove_tensor, factor_intermediates

from sympy import S

import pytest


class TestProperties():
    @pytest.mark.parametrize('adc_variant', ['pp'])
    @pytest.mark.parametrize('opstring', ['ca'])
    @pytest.mark.parametrize('adc_order', [0, 1, 2])
    def test_expectation_value(self, adc_variant, opstring, adc_order,
                               cls_instances, reference_data):
        # load the reference data
        ref = reference_data['properties_expectation_value']
        ref = ref[adc_variant][opstring][adc_order]

        # compute the complex non-symmetric s2s expec value
        res = cls_instances["mp"]["prop_pp"].expectation_value(adc_order,
                                                               opstring)
        res = Expr(res).substitute_contracted()
        ref_expec = ref["expectation_value"]
        assert simplify(res - ref_expec).sympy is S.Zero

        # real state expectation value for a symmetric operator matrix
        res.make_real()
        res.set_sym_tensors(["d"])
        res.rename_tensor("X", "Y")
        res = simplify(res)
        ref_expec = ref["real_symmetric_state_expectation_value"]
        assert simplify(res - ref_expec.sympy).sympy is S.Zero

        ref = ref["real_symmetric_state_dm"]
        # extract the state density matrix and factor intermediates
        res = factor_intermediates(res, ["t_amplitude", "mp_density"],
                                   adc_order)
        for block, block_expr in remove_tensor(res, "d").items():
            assert len(block) == 1
            block = block[0]
            assert block in ref

            ref_block = ref[block]
            assert simplify(block_expr - ref_block.sympy).sympy is S.Zero
