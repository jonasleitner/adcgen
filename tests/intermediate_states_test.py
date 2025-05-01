from adcgen.expression import ExprContainer
from adcgen.simplify import simplify

from sympy import S

import pytest


@pytest.mark.parametrize('order', [0, 1, 2])
class TestIntermediateStates:
    @pytest.mark.parametrize('variant,space,indices',
                             [("pp", "ph", "ia"), ("pp", "pphh", "ijab")])
    def test_precursor(self, variant: str, order: int, space: str,
                       indices: str, cls_instances: dict,
                       reference_data: dict):
        # load the reference_data
        reference = reference_data['isr_precursor'][variant]

        # isr based on re and mp ground state
        isr_mp = cls_instances['mp'][f"isr_{variant}"]
        isr_re = cls_instances['re'][f"isr_{variant}"]

        ref = reference[space][order]

        # build precursor state with mp ground state
        ket = ExprContainer(isr_mp.precursor(order, space, 'ket', indices))
        bra = ExprContainer(isr_mp.precursor(order, space, 'bra', indices))

        # build precursor states with re ground state
        ket_re = ExprContainer(isr_re.precursor(order, space, 'ket', indices))
        bra_re = ExprContainer(isr_re.precursor(order, space, 'bra', indices))
        # ensure that the ground state does not influence the
        # precursor states (and therefore also not the IS)
        assert (ket - ket_re).substitute_contracted().inner is S.Zero
        assert (bra - bra_re).substitute_contracted().inner is S.Zero

        assert (ket - ref['ket']).substitute_contracted().inner is S.Zero
        assert (bra - ref['bra']).substitute_contracted().inner is S.Zero

    @pytest.mark.parametrize('variant,block,indices',
                             [("pp", "ph,ph", "ia,jb")])
    def test_overlap_precursor(self, variant: str, order: int, block: str,
                               indices: str, cls_instances: dict,
                               reference_data: dict):
        # load the reference data
        reference = reference_data['isr_precursor_overlap'][variant]

        # class to generate the overlap
        isr = cls_instances['mp'][f"isr_{variant}"]

        ref = reference[block][order]

        # build the overlap and compare to the stored reference
        overlap = isr.overlap_precursor(order, block, indices)
        assert simplify(overlap - ref).inner is S.Zero
