from adcgen import (
    ExprContainer, simplify, remove_tensor, factor_intermediates,
    tensor_names
)

from sympy import S

import pytest


@pytest.mark.parametrize('adc_variant', ['pp'])
class TestProperties():
    @pytest.mark.parametrize('n_particles', [1])
    @pytest.mark.parametrize('adc_order', [0, 1, 2])
    def test_expectation_value(self, adc_variant: str, n_particles: int,
                               adc_order: int, cls_instances: dict,
                               reference_data: dict):
        # load the reference data
        ref = reference_data['properties_expectation_value']
        ref = ref[adc_variant][n_particles][adc_order]
        prop = cls_instances["mp"]["prop_pp"]

        # compute the complex non-symmetric s2s expec value
        res = prop.expectation_value(adc_order=adc_order, n_particles=1)
        res = ExprContainer(res).substitute_contracted()
        ref_expec = ref["expectation_value"]
        assert simplify(res - ref_expec).inner is S.Zero

        # build the result by only computing contributions of a specific order
        res_by_order = 0
        for order in range(adc_order + 1):
            res_by_order += prop.expectation_value(
                adc_order=adc_order, n_particles=1, order=order
            )
        assert simplify(res - res_by_order).inner is S.Zero

        # real state expectation value for a symmetric operator matrix
        res.make_real()
        res.sym_tensors = [tensor_names.operator]
        res.rename_tensor(tensor_names.left_adc_amplitude,
                          tensor_names.right_adc_amplitude)
        res = simplify(res)
        ref_expec = ref["real_symmetric_state_expectation_value"]
        assert simplify(res - ref_expec.inner).inner is S.Zero

        ref = ref["real_symmetric_state_dm"]
        # extract the state density matrix and factor intermediates
        res = factor_intermediates(res, ["t_amplitude", "mp_density"],
                                   adc_order)
        for block, block_expr in \
                remove_tensor(res, tensor_names.operator).items():
            assert len(block) == 1
            block = block[0]
            assert block in ref

            ref_block = ref[block]
            assert simplify(block_expr - ref_block.inner).inner is S.Zero

    @pytest.mark.parametrize('n_create,n_annihilate', [(1, 1)])
    @pytest.mark.parametrize('adc_order', [0, 1, 2])
    def test_trans_moment(self, adc_variant: str, n_create: int,
                          n_annihilate: int, adc_order: int,
                          cls_instances: dict, reference_data: dict):
        # load the reference data
        ref = reference_data['properties_trans_moment']
        ref = ref[adc_variant][f"{n_create}_{n_annihilate}"][adc_order]
        prop = cls_instances["mp"]["prop_pp"]

        # compute the complex transition dm
        res = prop.trans_moment(
            adc_order=adc_order, n_create=n_create, n_annihilate=n_annihilate
        )
        res = ExprContainer(res).substitute_contracted()
        ref_expec = ref["expectation_value"]
        assert simplify(res - ref_expec).inner is S.Zero

        # build the result by only computing contributions of a specific order
        res_by_order = 0
        for order in range(adc_order + 1):
            res_by_order += prop.trans_moment(
                adc_order=adc_order, n_create=n_create,
                n_annihilate=n_annihilate, order=order
            )
        assert simplify(res - res_by_order).inner is S.Zero

        # real expectation value (non-symmetric oeprator)
        res.make_real()
        res = simplify(res)
        ref_expec = ref["real_expectation_value"]
        assert simplify(res - ref_expec.inner).inner is S.Zero

        ref = ref["real_transition_dm"]
        # extract the transition dm and factor intermediates
        res = factor_intermediates(res, ["t_amplitude", "mp_density"],
                                   adc_order)
        for block, block_expr in \
                remove_tensor(res, tensor_names.operator).items():
            assert len(block) == 1
            block = block[0]
            assert block in ref

            ref_tdm = ref[block]
            assert simplify(block_expr - ref_tdm.inner).inner is S.Zero
