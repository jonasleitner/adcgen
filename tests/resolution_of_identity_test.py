from adcgen.spatial_orbitals import transform_to_spatial_orbitals
from adcgen.resolution_of_identity import apply_resolution_of_identity
from adcgen.simplify import simplify
from adcgen.expression import ExprContainer

from sympy import S

import pytest


class TestResolutionOfIdentity():

    @pytest.mark.parametrize('variant', ['mp', 're'])
    @pytest.mark.parametrize('order', [0, 1, 2, 3])
    @pytest.mark.parametrize('restriction', ['r', 'u'])
    @pytest.mark.parametrize('symmetry', ['sym', 'asym'])
    def test_ri_gs_energy(self, variant, order, restriction, symmetry,
                          cls_instances, reference_data):
        # load the reference data
        ref = reference_data['ri_gs_energy'][variant][order]
        ref = ref[restriction][symmetry].inner
        # transform restriction and symmetry to bool
        restricted = restriction == 'r'
        # compute the energy
        e = cls_instances[variant]['gs'].energy(order)
        expr = ExprContainer(e, real=True)

        sp_expr = transform_to_spatial_orbitals(expr, '', '', restricted)
        ri_expr = apply_resolution_of_identity(sp_expr, symmetry)

        assert simplify(ri_expr - ref).substitute_contracted().inner is S.Zero
