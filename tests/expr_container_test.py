from adcgen.expr_container import Expr
from adcgen.eri_orbenergy import EriOrbenergy
from adcgen.indices import get_symbols
from adcgen.intermediates import Intermediates
from adcgen.sympy_objects import (
    AntiSymmetricTensor, SymmetricTensor, NonSymmetricTensor, Amplitude
)

from sympy import S, Rational


class TestSymbolicDenominators:
    def test_no_denom(self):
        i, j, a, b = get_symbols("ijab")
        original = AntiSymmetricTensor("V", (i, j), (a, b))
        original = Expr(original, real=True)
        symbolic = original.copy().use_symbolic_denominators()
        assert symbolic.sympy - original.sympy is S.Zero
        assert len(symbolic.antisym_tensors) == 0  # D not set

    def test_t2_1(self):
        t2 = Intermediates().available["t2_1"]
        original = t2.expand_itmd(fully_expand=False).make_real()
        symbolic = original.copy().use_symbolic_denominators()
        assert "D" in symbolic.antisym_tensors
        i, j, a, b = get_symbols("ijab")
        ref = (AntiSymmetricTensor("V", (i, j), (a, b), 1) *
               SymmetricTensor("D", (a, b), (i, j), -1))
        assert symbolic.sympy - ref is S.Zero
        # reintroduce the explicit denominators
        explicit = symbolic.use_explicit_denominators()
        # need to fix the sign in the denominator for the test to pass
        explicit = EriOrbenergy(explicit).canonicalize_sign().expr
        original = EriOrbenergy(original).canonicalize_sign().expr
        assert explicit.sympy - original.sympy is S.Zero

    def test_t1_2(self):
        t1 = Intermediates().available["t1_2"]
        # first without fully expanding -> only 1 Denominator
        original = t1.expand_itmd(fully_expand=False).make_real()
        original.substitute_contracted()
        symbolic = original.copy().use_symbolic_denominators()
        assert "D" in symbolic.antisym_tensors
        i, j, k, a, b, c = get_symbols("ijkabc")
        ref = (Rational(1, 2) * Amplitude("t1", (b, c), (i, j)) *
               AntiSymmetricTensor("V", (j, a), (b, c), 1)
               + Rational(1, 2) * Amplitude("t1", (a, b), (j, k)) *
               AntiSymmetricTensor("V", (j, k), (i, b), 1))
        ref *= SymmetricTensor("D", (i,), (a,), -1)
        assert symbolic.sympy - ref.expand() is S.Zero
        explicit = symbolic.use_explicit_denominators().expand()
        assert explicit.sympy - original.sympy is S.Zero

        # fully expand the itmd -> 2 denominators
        original = t1.expand_itmd(fully_expand=True).make_real()
        original.substitute_contracted()
        symbolic = original.copy().use_symbolic_denominators()
        assert "D" in symbolic.antisym_tensors
        ref = (Rational(1, 2) * AntiSymmetricTensor("V", (b, c), (i, j), 1) *
               AntiSymmetricTensor("V", (j, a), (b, c), 1) *
               SymmetricTensor("D", (b, c), (i, j), -1)
               + Rational(1, 2) * AntiSymmetricTensor("V", (a, b), (j, k), 1) *
               AntiSymmetricTensor("V", (j, k), (i, b), 1) *
               SymmetricTensor("D", (a, b), (j, k), -1))
        ref *= SymmetricTensor("D", (i,), (a,), -1)
        assert symbolic.sympy - ref.expand() is S.Zero
        # need to fix the sign of the denominators!
        explicit = 0
        for term in symbolic.use_explicit_denominators().terms:
            explicit += EriOrbenergy(term).canonicalize_sign().expr
        ref = 0
        for term in original.terms:
            ref += EriOrbenergy(term).canonicalize_sign().expr
        assert explicit.sympy - ref.sympy is S.Zero

    def test_squared_denom(self):
        i, j, a, b = get_symbols("ijab")
        original = AntiSymmetricTensor("V", (i, j), (a, b), 1)
        original /= (
            NonSymmetricTensor("e", (i,)) + NonSymmetricTensor("e", (j,)) -
            NonSymmetricTensor("e", (a,)) - NonSymmetricTensor("e", (b,))
        )**2
        original = Expr(original, real=True)
        symbolic = original.copy().use_symbolic_denominators()
        assert "D" in symbolic.antisym_tensors
        ref = (AntiSymmetricTensor("V", (i, j), (a, b), 1) *
               SymmetricTensor("D", (i, j), (a, b), -1)**2)
        assert symbolic.sympy - ref is S.Zero
        explicit = symbolic.use_explicit_denominators()
        assert explicit.sympy - original.sympy is S.Zero

    def test_some_denom_terms(self):
        t2 = Intermediates().available["t2_1"]
        i, j, a, b = get_symbols("ijab")
        original: Expr = (t2.expand_itmd(fully_expand=False).make_real()
                          + AntiSymmetricTensor("V", (i, j), (a, b), 1))
        symbolic = original.copy().use_symbolic_denominators()
        assert "D" in symbolic.antisym_tensors
        ref = (AntiSymmetricTensor("V", (i, j), (a, b), 1) *
               SymmetricTensor("D", (a, b), (i, j), -1)
               + AntiSymmetricTensor("V", (i, j), (a, b), 1))
        assert symbolic.sympy - ref is S.Zero
