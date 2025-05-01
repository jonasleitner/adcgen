from adcgen.expression import ExprContainer
from adcgen.eri_orbenergy import EriOrbenergy
from adcgen.indices import get_symbols
from adcgen.intermediates import Intermediates
from adcgen.sympy_objects import (
    AntiSymmetricTensor, SymmetricTensor, NonSymmetricTensor, Amplitude
)
from adcgen.tensor_names import tensor_names

from sympy import S, Rational


class TestSymbolicDenominators:
    def test_no_denom(self):
        i, j, a, b = get_symbols("ijab")
        original = AntiSymmetricTensor("V", (i, j), (a, b))
        original = ExprContainer(original, real=True)
        symbolic = original.copy().use_symbolic_denominators()
        assert symbolic.inner - original.inner is S.Zero  # type: ignore
        assert not symbolic.antisym_tensors  # D not set

    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        original = t2.expand_itmd(fully_expand=False)
        assert isinstance(original, ExprContainer)
        original.make_real()
        symbolic = original.copy().use_symbolic_denominators()
        assert tensor_names.sym_orb_denom in symbolic.antisym_tensors
        i, j, a, b = get_symbols("ijab")
        ref = (AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b), 1) *  # type: ignore  # noqa E501
               SymmetricTensor(tensor_names.sym_orb_denom, (a, b), (i, j), -1))
        assert symbolic.inner - ref is S.Zero
        # reintroduce the explicit denominators
        explicit = symbolic.use_explicit_denominators()
        assert tensor_names.sym_orb_denom not in explicit.antisym_tensors
        # need to fix the sign in the denominator for the test to pass
        explicit = EriOrbenergy(explicit).canonicalize_sign().expr
        original = EriOrbenergy(original).canonicalize_sign().expr
        assert explicit.inner - original.inner is S.Zero  # type: ignore

    def test_t1_2(self):
        t1 = Intermediates().available["t1_2"]
        # first without fully expanding -> only 1 Denominator
        original = t1.expand_itmd(fully_expand=False)
        assert isinstance(original, ExprContainer)
        original.make_real()
        original.substitute_contracted()
        symbolic = original.copy().use_symbolic_denominators()
        assert tensor_names.sym_orb_denom in symbolic.antisym_tensors
        i, j, k, a, b, c = get_symbols("ijkabc")
        ref = (Rational(1, 2) *
               Amplitude(f"{tensor_names.gs_amplitude}1", (b, c), (i, j)) *
               AntiSymmetricTensor(tensor_names.eri, (j, a), (b, c), 1)
               + Rational(1, 2) *
               Amplitude(f"{tensor_names.gs_amplitude}1", (a, b), (j, k)) *
               AntiSymmetricTensor(tensor_names.eri, (j, k), (i, b), 1))
        ref *= SymmetricTensor(tensor_names.sym_orb_denom, (i,), (a,), -1)  # type: ignore # noqa E501
        assert symbolic.inner - ref.expand() is S.Zero  # type: ignore
        explicit = symbolic.use_explicit_denominators().expand()
        assert tensor_names.sym_orb_denom not in explicit.antisym_tensors
        assert explicit.inner - original.inner is S.Zero  # type: ignore

        # fully expand the itmd -> 2 denominators
        original = t1.expand_itmd(fully_expand=True)
        assert isinstance(original, ExprContainer)
        original.make_real()
        original.substitute_contracted()
        symbolic = original.copy().use_symbolic_denominators()
        assert tensor_names.sym_orb_denom in symbolic.antisym_tensors
        ref = (Rational(1, 2) *
               AntiSymmetricTensor(tensor_names.eri, (b, c), (i, j), 1) *
               AntiSymmetricTensor(tensor_names.eri, (j, a), (b, c), 1) *
               SymmetricTensor(tensor_names.sym_orb_denom, (b, c), (i, j), -1)
               + Rational(1, 2) *
               AntiSymmetricTensor(tensor_names.eri, (a, b), (j, k), 1) *
               AntiSymmetricTensor(tensor_names.eri, (j, k), (i, b), 1) *
               SymmetricTensor(tensor_names.sym_orb_denom, (a, b), (j, k), -1))
        ref *= SymmetricTensor(tensor_names.sym_orb_denom, (i,), (a,), -1)  # type: ignore # noqa E501
        assert symbolic.inner - ref.expand() is S.Zero  # type: ignore
        # need to fix the sign of the denominators!
        explicit = 0
        for term in symbolic.use_explicit_denominators().terms:
            explicit += EriOrbenergy(term).canonicalize_sign().expr
        assert isinstance(explicit, ExprContainer)
        assert tensor_names.sym_orb_denom not in explicit.antisym_tensors
        ref = 0
        for term in original.terms:
            ref += EriOrbenergy(term).canonicalize_sign().expr
        assert isinstance(ref, ExprContainer)
        assert explicit.inner - ref.inner is S.Zero  # type: ignore

    def test_squared_denom(self):
        i, j, a, b = get_symbols("ijab")
        original = AntiSymmetricTensor("V", (i, j), (a, b), 1)
        original /= (
            NonSymmetricTensor(tensor_names.orb_energy, (i,))
            + NonSymmetricTensor(tensor_names.orb_energy, (j,))  # type: ignore
            - NonSymmetricTensor(tensor_names.orb_energy, (a,))
            - NonSymmetricTensor(tensor_names.orb_energy, (b,))
        )**2
        original = ExprContainer(original, real=True)
        symbolic = original.copy().use_symbolic_denominators()
        assert tensor_names.sym_orb_denom in symbolic.antisym_tensors
        ref = (  # type: ignore
            AntiSymmetricTensor("V", (i, j), (a, b), 1) *
            SymmetricTensor(tensor_names.sym_orb_denom, (i, j), (a, b), -1)**2
        )
        assert symbolic.inner - ref is S.Zero
        explicit = symbolic.use_explicit_denominators()
        assert explicit.inner - original.inner is S.Zero  # type: ignore

    def test_some_denom_terms(self):
        t2 = Intermediates().available["t2_1"]
        i, j, a, b = get_symbols("ijab")
        original = (
            t2.expand_itmd(fully_expand=False).make_real()  # type: ignore
            + AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b), 1)
        )
        symbolic = original.copy().use_symbolic_denominators()
        assert tensor_names.sym_orb_denom in symbolic.antisym_tensors
        ref = (AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b), 1) *  # type: ignore # noqa E501
               SymmetricTensor(tensor_names.sym_orb_denom, (a, b), (i, j), -1)
               + AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b), 1))
        assert symbolic.inner - ref is S.Zero
