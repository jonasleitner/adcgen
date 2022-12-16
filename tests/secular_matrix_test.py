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

        # sort the matrix block by delta space, reduce the terms and
        # factor intermediates
        for delta_sp, sub_expr in sort.by_delta_types(m).items():
            print(delta_sp)
            sub_expr = reduce_expr(sub_expr)
            sub_expr = factor_intermediates(sub_expr)
            ref_m = ref["real_factored_m"]["_".join(delta_sp)]
            ref_m = expr(ref_m.sympy, **sub_expr.assumptions)
            assert simplify(sub_expr - ref_m).sympy is S.Zero
