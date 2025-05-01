from adcgen.expression import ExprContainer
from adcgen.simplify import simplify
import adcgen.sort_expr as sort
from adcgen.reduce_expr import reduce_expr
from adcgen.factor_intermediates import factor_intermediates
from adcgen.eri_orbenergy import EriOrbenergy

from sympy import S

import pytest


@pytest.mark.parametrize('order', [0, 1, 2, 3])
class TestSecularMatrix():
    @pytest.mark.parametrize('block,indices',
                             [('ph,ph', 'ia,jb')])
    def test_isr_matrix_block(self, order, block, indices, cls_instances,
                              reference_data):
        # load reference data
        ref = reference_data["secular_matrix"]["pp"][block][order]

        # compute the raw matrix block
        m = cls_instances['mp']['m'].isr_matrix_block(order, block, indices)
        m = ExprContainer(m)
        ref_m = ref["complex"]
        assert simplify(m - ref_m).inner is S.Zero

        # assume a real orbital basis
        m = ExprContainer(m.inner, real=True).substitute_contracted()
        ref_m = ref['real'].make_real()
        assert simplify(m - ref_m).inner is S.Zero

        # expand itmds, cancel orbital energy fractions
        # and collect matching terms
        m = reduce_expr(m.diagonalize_fock())
        # check that we cancelled all orbital energy numerators
        for term in m.terms:
            term = EriOrbenergy(term)
            assert term.num.inner in [S.One, S.Zero]
        # factor intermediates
        m = factor_intermediates(m, max_order=order-1)
        # check that we removed all denominators
        for term in m.terms:
            term = EriOrbenergy(term)
            assert term.denom.inner is S.One
        # split according to the delta space and compare to reference data
        for delta_sp, sub_expr in sort.by_delta_types(m).items():
            # compare to reference
            ref_sub_expr = ref["real_factored"]["-".join(delta_sp)]
            ref_sub_expr = ExprContainer(
                ref_sub_expr.inner, **sub_expr.assumptions
            )
            assert simplify(sub_expr - ref_sub_expr).inner is S.Zero
            # exploit permutational symmetry and test that the result is
            # still identical
            # if we dont have a diagonal block we need to set
            # bra_ket_sym to 0
            sp1, sp2 = block.split(',')
            bra_ket_sym = 1 if sp1 == sp2 else 0
            exploited_perm_sym = sort.exploit_perm_sym(
                sub_expr, target_indices=indices, bra_ket_sym=bra_ket_sym
            )
            re_expanded_block = ExprContainer(0, **sub_expr.assumptions)
            for perm_sym, sub_sub_expr in exploited_perm_sym.items():
                # ensure that we can reexpand to the original expr
                re_expanded_block += sub_sub_expr.copy()
                for perms, factor in perm_sym:
                    re_expanded_block += \
                        sub_sub_expr.copy().permute(*perms) * factor
            assert simplify(re_expanded_block - sub_expr).inner is S.Zero
