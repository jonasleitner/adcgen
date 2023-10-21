from sympy_adc.expr_container import Expr
from sympy_adc.simplify import simplify, remove_tensor
from sympy_adc import sort_expr as sort

from sympy import S

import pytest


@pytest.mark.parametrize('order', [0, 1, 2])
class TestGroundState():
    @pytest.mark.parametrize('variant', ['mp', 're'])
    def test_energy(self, order, variant, cls_instances, reference_data):
        # load the reference data
        ref = reference_data[variant + "_energy"][order]

        # compute the energy
        e = cls_instances[variant]['gs'].energy(order)
        assert (ref - e).substitute_contracted().sympy is S.Zero

    @pytest.mark.parametrize('operator', ['ca'])
    def test_expectation_value(self, order: int, operator: str, cls_instances,
                               reference_data):
        # load the reference data
        ref = reference_data["mp_1p_dm"][order]

        # compute the expectation value
        expec = cls_instances['mp']['gs'].expectation_value(order, operator)
        expec = Expr(expec)
        ref_expec = ref['expec_val']
        assert simplify(ref_expec - expec).sympy is S.Zero

        # assume a real basis and a symmetric operator/ symmetric dm
        expec = expec.substitute_contracted()
        expec = Expr(expec.sympy, real=True, sym_tensors=['d'])
        expec = simplify(expec)
        ref_expec = ref['real_sym_expec_val']
        ref_expec = Expr(ref_expec.sympy, **expec.assumptions)
        assert simplify(ref_expec - expec).sympy is S.Zero

        # extract all blocks of the symmetric dm
        density_matrix = remove_tensor(expec, 'd')
        for block, block_expr in density_matrix.items():
            assert len(block) == 1
            ref_dm = ref["real_sym_dm"][block[0]]
            ref_dm = Expr(ref_dm, **block_expr.assumptions)
            assert simplify(block_expr - ref_dm).sympy is S.Zero

            # exploit permutational symmetry
            re_expanded_dm = Expr(0, **block_expr.assumptions)
            for perm_sym, sub_expr in \
                    sort.exploit_perm_sym(block_expr).items():
                re_expanded_dm += sub_expr.copy()
                for perms, factor in perm_sym:
                    re_expanded_dm += sub_expr.copy().permute(*perms) * factor
            assert simplify(re_expanded_dm - block_expr).sympy is S.Zero
