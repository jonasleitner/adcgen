from adcgen.core_valence_separation import (
    expand_contracted_indices, is_allowed_cvs_block, allowed_cvs_blocks,
    apply_cvs_approximation
)
from adcgen.expr_container import Expr, Obj
from adcgen.indices import get_symbols
from adcgen.intermediates import Intermediates, RegisteredIntermediate
from adcgen.simplify import simplify
from adcgen.sympy_objects import (
    AntiSymmetricTensor, SymmetricTensor, Amplitude
)
from adcgen.tensor_names import tensor_names
from adcgen import sort

from sympy import S

import pytest


class TestCoreValenceSeparation:
    def test_expand_contracted_indices(self):
        # trivial case: allow all blocks (no CVS approximation)
        i, j, a, b = get_symbols("ijab")
        I, J = get_symbols("IJ")
        tensor = AntiSymmetricTensor("V", (i, j), (a, b))
        term = Expr(tensor, target_idx="").terms[0]
        res = expand_contracted_indices(term, target_subs={})
        ref = (AntiSymmetricTensor("V", (i, j), (a, b)) +
               AntiSymmetricTensor("V", (I, j), (a, b)) +
               AntiSymmetricTensor("V", (i, J), (a, b)) +
               AntiSymmetricTensor("V", (I, J), (a, b)))
        assert res.sympy - ref is S.Zero
        # even more trivial: no occupied contracted indices
        term = Expr(tensor, target_idx="ij").terms[0]
        res = expand_contracted_indices(term, target_subs={})
        assert res.sympy - tensor is S.Zero
        res = expand_contracted_indices(term, target_subs={i: I})
        ref = AntiSymmetricTensor("V", (I, j), (a, b))
        assert res.sympy - ref is S.Zero
        res = expand_contracted_indices(term, target_subs={i: I, j: J})
        ref = AntiSymmetricTensor("V", (I, J), (a, b))
        assert res.sympy - ref is S.Zero

    def test_allowed_cvs_blocks(self):
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        # Coulomb
        coulomb = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        res = allowed_cvs_blocks(Expr(coulomb), "ijkl",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oooo", "oocc", "ccoo", "cccc")
        # ERI
        eri = AntiSymmetricTensor(tensor_names.eri, (i, j), (k, l))
        res = allowed_cvs_blocks(Expr(eri), "ijkl",
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == ("oooo", "ococ", "occo", "cooc", "coco", "cccc")

    @pytest.mark.parametrize("amplitude", ["t2_1", "t1_2", "t2_2", "t3_2",
                                           "t4_2", "t1_3", "t2_3"])
    def test_allowed_t_amplitude_cvs_blocks(self, amplitude):
        ampl: RegisteredIntermediate = Intermediates().available.get(amplitude)
        expr: Expr = ampl.tensor()
        tensor_obj: Obj = expr.terms[0].objects[0]
        # ov/oovv/... is a valid block
        assert is_allowed_cvs_block(tensor_obj, tensor_obj.space)
        # ov/oovv/... is the only valid block
        target: Amplitude = expr.sympy.idx
        res = allowed_cvs_blocks(expr, target,
                                 is_allowed_cvs_block=is_allowed_cvs_block)
        assert res == (tensor_obj.space,)
        # ensure that we get the same result if we expand the amplitude
        res = ampl.allowed_cvs_blocks(is_allowed_cvs_block)
        assert res == (tensor_obj.space,)

    def test_apply_cvs_approximation(self):
        # case1: t2_1
        t2_1: Expr = Intermediates().available["t2_1"].expand_itmd()
        res = apply_cvs_approximation(t2_1.copy(), "")
        assert res.sympy is t2_1.sympy
        res = apply_cvs_approximation(t2_1.copy(), "I")
        assert res.sympy is S.Zero
        res = apply_cvs_approximation(t2_1.copy(), "IJ")
        assert res.sympy is S.Zero
        # case2: t2_2
        t2_2: Expr = Intermediates().available["t2_2"].expand_itmd().expand()
        res = apply_cvs_approximation(t2_2.copy(), "")
        res.use_symbolic_denominators()
        ref = t2_2.use_symbolic_denominators()
        assert simplify(res - ref).sympy is S.Zero
        res = apply_cvs_approximation(t2_2.copy(), "I")
        assert res.sympy is S.Zero
        res = apply_cvs_approximation(t2_2.copy(), "IJ")
        assert res.sympy is S.Zero
        # case3: p0_2_oo = t^ab_ik t^ab_jk
        p2_oo: Expr = Intermediates().available["p0_2_oo"].expand_itmd()
        p2_oo.expand().use_symbolic_denominators()
        res = apply_cvs_approximation(p2_oo, "")
        assert (res - p2_oo).sympy is S.Zero
        res = apply_cvs_approximation(p2_oo, "I")
        assert res.sympy is S.Zero
        res = apply_cvs_approximation(p2_oo, "IJ")
        assert res.sympy is S.Zero

    @pytest.mark.parametrize("order", [0])
    def test_apply_cvs_approximation_ph_ph(self, order, reference_data):
        # load the simplified and factorized equations secular matrix
        # equations from the reference data.
        ref_data = reference_data["secular_matrix"]["pp"]["ph,ph"][order]
        m_contribs = ref_data["real_factored"]
        m_expr: Expr = sum(m_contribs.values())
        m_expr.make_real()
        m_expr.set_sym_tensors(("p2", "t2sq"))  # fine through 3rd order
        print(m_expr)
        # build the valence-valence block (no core orbitals)
        res = apply_cvs_approximation(m_expr.copy(), "")
        print(res)
        assert simplify(res - m_expr).sympy is S.Zero
        # build the valence-core block
        res = apply_cvs_approximation(m_expr.copy(), "J")
        assert res.sympy is S.Zero
        # build the core-valence block
        res = apply_cvs_approximation(m_expr.copy(), "I")
        assert res.sympy is S.Zero
        # build the core-core block
        res = apply_cvs_approximation(m_expr.copy(), "IJ")
        for delta_sp, sub_expr in sort.by_delta_types(res).items():
            ref = ref_data["real_factored_cvs"]["-".join(delta_sp)]
            ref = Expr(ref.sympy, **sub_expr.assumptions)
            assert simplify(sub_expr - ref).sympy is S.Zero
