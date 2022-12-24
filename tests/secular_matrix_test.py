from sympy_adc.expr_container import expr
from sympy_adc.simplify import simplify
import sympy_adc.sort_expr as sort
from sympy_adc.reduce_expr import reduce_expr
from sympy_adc.factor_intermediates import factor_intermediates

from sympy import S

import pytest


@pytest.mark.parametrize('order', [0, 1, 2, 3])
class TestSecularMatrix():
    @pytest.mark.parametrize('block,indices',
                             [('ph,ph', 'ia,jb')])
    def test_isr_matrix_block(self, order, block, indices, cls_instances,
                              reference_data):
        # load reference data
        block_key = "-".join(block.split(','))
        ref = reference_data[f"m_{block_key}_isr"][order]

        # compute the raw matrix block
        m = cls_instances.m_pp.isr_matrix_block(order, block, indices)
        m = expr(m)
        ref_m = ref['m']
        assert simplify(m - ref_m).sympy is S.Zero

        # assume a real orbital basis
        m = expr(m.sympy, real=True).substitute_contracted()
        ref_m = ref['real_m'].make_real()
        assert simplify(m - ref_m).sympy is S.Zero

        idx1, idx2 = indices.split(',')
        # sort the matrix block by delta space, reduce the terms and
        # factor intermediates
        for delta_sp, block_expr in sort.by_delta_types(m).items():
            block_expr = reduce_expr(block_expr.diagonalize_fock())
            block_expr = factor_intermediates(block_expr, max_order=order-1)
            ref_block_expr = ref["real_factored_m"]["_".join(delta_sp)]
            ref_block_expr = expr(ref_block_expr.sympy,
                                  **block_expr.assumptions)
            assert simplify(block_expr - ref_block_expr).sympy is S.Zero

            # exploit permutational symmetry and test that the result is
            # still identical
            exploited_perm_sym = sort.exploit_perm_sym(
                block_expr, target_upper=idx1, target_lower=idx2,
                target_bra_ket_sym=1
            )
            re_expanded_block = expr(0, **block_expr.assumptions)
            for perm_sym, sub_expr in exploited_perm_sym.items():
                # ensure that we can reexpand to the original expr
                re_expanded_block += sub_expr.copy()
                for perms, factor in perm_sym:
                    re_expanded_block += \
                        sub_expr.copy().permute(*perms) * factor
            assert simplify(re_expanded_block - block_expr).sympy is S.Zero
