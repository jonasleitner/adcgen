from adcgen.expr_container import Expr
from adcgen.simplify import simplify, remove_tensor
from adcgen import sort_expr as sort
from adcgen.reduce_expr import factor_eri_parts, factor_denom

from sympy import S

import itertools
import pytest


class TestGroundState():
    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('variant', ['mp', 're'])
    def test_energy(self, order, variant, cls_instances, reference_data):
        # load the reference data
        ref = reference_data["gs_energy"][variant][order]

        # compute the energy
        e = cls_instances[variant]['gs'].energy(order)
        assert (ref - e).substitute_contracted().sympy is S.Zero

    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('braket', ['bra', 'ket'])
    def test_psi(self, order, braket, cls_instances, reference_data):
        # load the reference data
        ref = reference_data['gs_psi'][order][braket]
        if order == 1:
            ref, ref_with_singles = ref['no_singles'], ref['with_singles']
            # additionaly compare the wfn with singles
            mp_psi = Expr(
                cls_instances['mp']['gs_with_singles'].psi(order, braket)
            )
            re_psi = Expr(
                cls_instances['re']['gs_with_singles'].psi(order, braket)
            )
            assert (mp_psi - re_psi).substitute_contracted().sympy is S.Zero
            assert ((mp_psi - ref_with_singles).substitute_contracted().sympy
                    is S.Zero)

        # the wfn should not depend on the variant
        mp_psi = Expr(cls_instances['mp']['gs'].psi(order, braket))
        re_psi = Expr(cls_instances['re']['gs'].psi(order, braket))
        assert (mp_psi - re_psi).substitute_contracted().sympy is S.Zero
        assert (mp_psi - ref).substitute_contracted().sympy is S.Zero

    @pytest.mark.parametrize('order', [1, 2])
    @pytest.mark.parametrize('variant', ['mp', 're'])
    def test_amplitude(self, order, variant, cls_instances, reference_data):

        def simplify_mp(ampl):
            res = 0
            for term in itertools.chain.from_iterable(
                            factor_denom(sub_expr)
                            for sub_expr in factor_eri_parts(ampl)):
                res += term.factor()
            return res

        spaces = {1: [('ph', 'ia'), ('pphh', 'ijab')],
                  2: [('ph', 'ia'), ('pphh', 'ijab'), ('ppphhh', 'ijkabc'),
                      ('pppphhhh', 'ijklabcd')]}

        # load the reference data
        ref = reference_data['gs_amplitude'][variant][order]

        for sp, idx in spaces[order]:
            # skip 2nd order re triples and quadruples, since I don't know how
            # the equations should loook and the generation takes a very long
            # time.
            if variant == "re" and sp in ["ppphhh", "pppphhhh"]:
                continue
            # compute the amplitude
            ampl = cls_instances[variant]['gs'].amplitude(order, sp, idx)
            # no einstein sum convention (for mp) -> set target idx
            ampl = Expr(ampl, target_idx=idx)
            if variant == 'mp':
                ampl = simplify_mp(
                    (ampl - ref[sp].sympy).substitute_contracted()
                )
            elif variant == 're':
                ampl = simplify(
                    (ampl - ref[sp].sympy).substitute_contracted()
                )
            else:
                raise NotImplementedError()
            assert ampl.sympy is S.Zero

    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('n_particles', [1])
    def test_expectation_value(self, order: int, n_particles: int,
                               cls_instances, reference_data):
        # load the reference data
        ref = reference_data["gs_expectation_value"]
        re_ref = ref["re"][n_particles][order]
        ref = ref["mp"][n_particles][order]

        # compute the expectation value
        expec = cls_instances["mp"]['gs'].expectation_value(
            order=order, n_particles=n_particles
        )
        expec = Expr(expec)
        ref_expec = ref['expectation_value']
        assert simplify(ref_expec - expec).sympy is S.Zero

        # the result for mp and re should be the same!
        assert simplify(expec - re_ref["expectation_value"]).sympy is S.Zero

        # assume a real basis and a symmetric operator/ symmetric dm
        expec.substitute_contracted()
        expec.make_real()
        expec.set_sym_tensors(["d"])
        expec = simplify(expec)
        ref_expec = ref['real_symmetric_expectation_value']
        assert simplify(expec - ref_expec.sympy).sympy is S.Zero

        # extract all blocks of the symmetric dm
        density_matrix = remove_tensor(expec, 'd')
        for block, block_expr in density_matrix.items():
            assert len(block) == 1
            ref_dm = ref["real_symmetric_dm"][block[0]]
            assert simplify(block_expr - ref_dm.sympy).sympy is S.Zero

            # exploit permutational symmetry and collec the terms again
            re_expanded_dm = Expr(0, **block_expr.assumptions)
            for perm_sym, sub_expr in \
                    sort.exploit_perm_sym(block_expr).items():
                re_expanded_dm += sub_expr.copy()
                for perms, factor in perm_sym:
                    re_expanded_dm += sub_expr.copy().permute(*perms) * factor
            assert simplify(re_expanded_dm - block_expr).sympy is S.Zero
