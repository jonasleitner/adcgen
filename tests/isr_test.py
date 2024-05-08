from adcgen.expr_container import Expr
from adcgen.simplify import simplify

from sympy import S

import pytest


@pytest.mark.parametrize('variant', ['pp'])
@pytest.mark.parametrize('order', [0, 1, 2])
class TestIsr:
    def test_precursor(self, variant, order, cls_instances,
                       reference_data):
        # load the reference_data
        reference = reference_data['isr_precursor'][variant]

        # isr based on re and mp ground state
        isr_mp = cls_instances['mp'][f"isr_{variant}"]
        isr_re = cls_instances['re'][f"isr_{variant}"]

        spaces = {'pp': [('ph', 'ia')]}

        for space, indices in spaces[variant]:
            ref = reference[space][order]

            # build precursor state with mp ground state
            ket = Expr(isr_mp.precursor(order, space, 'ket', indices))
            bra = Expr(isr_mp.precursor(order, space, 'bra', indices))

            # build precursor states with re ground state
            ket_re = Expr(isr_re.precursor(order, space, 'ket', indices))
            bra_re = Expr(isr_re.precursor(order, space, 'bra', indices))
            # ensure that the ground state does not influence the
            # precursor states (and therefore also not the IS)
            assert (ket - ket_re).substitute_contracted().sympy is S.Zero
            assert (bra - bra_re).substitute_contracted().sympy is S.Zero

            assert (ket - ref['ket']).substitute_contracted().sympy is S.Zero
            assert (bra - ref['bra']).substitute_contracted().sympy is S.Zero

    def test_overlap_precursor(self, variant, order, cls_instances,
                               reference_data):
        # load the reference data
        reference = reference_data['isr_precursor_overlap'][variant]

        # class to generate the overlap
        isr = cls_instances['mp'][f"isr_{variant}"]

        spaces = {'pp': [('ph,ph', 'ia,jb')]}

        for block, indices in spaces[variant]:
            ref = reference[block][order]

            # build the overlap and compare to the stored reference
            overlap = isr.overlap_precursor(order, block, indices)
            assert simplify(overlap - ref).sympy is S.Zero
