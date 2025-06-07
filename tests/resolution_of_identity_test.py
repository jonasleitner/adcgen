from adcgen.expression import ExprContainer
from adcgen.indices import Index, get_symbols
from adcgen.misc import Inputerror
from adcgen.resolution_of_identity import apply_resolution_of_identity
from adcgen.simplify import simplify
from adcgen.spatial_orbitals import transform_to_spatial_orbitals
from adcgen import (
    AntiSymmetricTensor, SymmetricTensor, tensor_names,
)

from sympy import S

import pytest


class TestResolutionOfIdentity():
    def test_sanity_checks(self):
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        # forgot to expand the antisym ERI
        tensor = AntiSymmetricTensor(tensor_names.eri, (i, j), (k, l))
        with pytest.raises(Inputerror):
            apply_resolution_of_identity(ExprContainer(tensor))
        # implementation assumes braket symmetry
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        with pytest.raises(NotImplementedError):
            apply_resolution_of_identity(ExprContainer(tensor))
        # mix of spin and spatial orbitals is not allowed
        i, j = get_symbols("ij", "aa")
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        with pytest.raises(NotImplementedError):
            apply_resolution_of_identity(ExprContainer(tensor))

    def test_sym_factorisation(self):
        i, j, k, l, P = get_symbols("ijklP")  # noqa E741
        # expand a single coulomb integral in spin orbital basis
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="sym"
        )
        ref = (
            SymmetricTensor(tensor_names.ri_sym, (P,), (i, j))
            * SymmetricTensor(tensor_names.ri_sym, (P,), (k, l))
        )
        assert res.inner - ref is S.Zero
        # with spatial orbitals
        i, j, k, l, P = get_symbols("ijklP", "ababa")  # noqa E741
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="sym"
        )
        ref = (
            SymmetricTensor(tensor_names.ri_sym, (P,), (i, j))
            * SymmetricTensor(tensor_names.ri_sym, (P,), (k, l))
        )
        assert res.inner - ref is S.Zero

    def test_asym_factorisation(self):
        i, j, k, l, P = get_symbols("ijklP")  # noqa E741
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="asym"
        )
        ref = (
            SymmetricTensor(tensor_names.ri_asym_factor, (P,), (i, j))
            * SymmetricTensor(tensor_names.ri_asym_eri, (P,), (k, l))
        )
        assert res.inner - ref is S.Zero
        # with spatial orbitals
        i, j, k, l, P = get_symbols("ijklP", "ababa")  # noqa E741
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="asym"
        )
        ref = (
            SymmetricTensor(tensor_names.ri_asym_factor, (P,), (i, j))
            * SymmetricTensor(tensor_names.ri_asym_eri, (P,), (k, l))
        )
        assert res.inner - ref is S.Zero

    def test_resolve_indices(self):
        i, j, k, l, P, Q = get_symbols("ijklPQ")
        # without resolve indices a unknown P should be in the res
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="sym", resolve_indices=False
        )
        ref = (
            SymmetricTensor(tensor_names.ri_sym, (P,), (i, j))
            * SymmetricTensor(tensor_names.ri_sym, (P,), (k, l))
        )
        assert P not in res.inner.atoms(Index)
        # subsitute contracted should resolve the unknown P
        res.substitute_contracted()
        assert P in res.inner.atoms(Index)
        assert res.inner - ref is S.Zero
        # add another tensor and try with resolve_indices to immediately
        # get a good result
        tensor *= SymmetricTensor(tensor_names.coulomb, (i, k), (j, l), 1)
        res = apply_resolution_of_identity(
            ExprContainer(tensor), factorisation="sym", resolve_indices=True
        )
        ref *= (
            SymmetricTensor(tensor_names.ri_sym, (Q,), (i, k))
            * SymmetricTensor(tensor_names.ri_sym, (Q,), (j, l))
        )
        assert P in res.inner.atoms(Index) and Q in res.inner.atoms(Index)
        assert simplify(res - ref).inner is S.Zero

    @pytest.mark.parametrize('variant', ['mp', 're'])
    @pytest.mark.parametrize('order', [0, 1, 2, 3])
    @pytest.mark.parametrize('restriction', ['r', 'u'])
    @pytest.mark.parametrize('symmetry', ['sym', 'asym'])
    def test_ri_gs_energy(self, variant, order, restriction, symmetry,
                          cls_instances, reference_data):
        # load the reference data
        ref = reference_data['ri_gs_energy'][variant][order]
        ref = ref[restriction][symmetry]
        ref.make_real()
        # transform restriction and symmetry to bool
        restricted = restriction == 'r'
        # compute the energy
        e = cls_instances[variant]['gs'].energy(order)
        expr = ExprContainer(e, real=True)

        sp_expr = transform_to_spatial_orbitals(expr, '', '', restricted)
        ri_expr = apply_resolution_of_identity(sp_expr, symmetry)

        assert simplify(ri_expr - ref).substitute_contracted().inner is S.Zero
