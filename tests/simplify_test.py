from adcgen.simplify import simplify_unitary, remove_tensor, simplify
from adcgen.sympy_objects import (
    NonSymmetricTensor, AntiSymmetricTensor, KroneckerDelta, Amplitude
)
from adcgen.expr_container import Expr
from adcgen.indices import get_symbols
from adcgen.tensor_names import tensor_names

from sympy import S, Rational


class TestRemoveTensor:
    def test_trivial(self):
        i, j, k, a, b, c = get_symbols("ijkabc")

        test = Expr(
            Rational(1, 2) *
            Amplitude(tensor_names.left_adc_amplitude, (a, b), (i, j)) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k)) *
            AntiSymmetricTensor(tensor_names.operator, (k,), (c,))
        )
        ref = Expr(
            Rational(1, 2) *
            Amplitude(tensor_names.left_adc_amplitude, (b, c), (j, k)) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k))
        )
        res = remove_tensor(test, tensor_names.operator)
        assert len(res) == 1
        block, res = res.popitem()
        assert block == ("ov",)
        assert simplify(res - ref).sympy is S.Zero

    def test_with_braketsym(self):
        i, j, k, a, b, c = get_symbols("ijkabc")
        # we will get an additional factor of 1/2, because the non-canonical
        # d_vo block is also included in the test term.

        test = Expr(
            Rational(1, 2) *
            Amplitude(tensor_names.left_adc_amplitude, (a, b), (i, j)) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k)) *
            AntiSymmetricTensor(tensor_names.operator, (k,), (c,), 1)
        )
        ref = Expr(
            Rational(1, 4) *
            Amplitude(tensor_names.left_adc_amplitude, (b, c), (j, k)) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k))
        )
        res = remove_tensor(test, tensor_names.operator)
        assert len(res) == 1
        block, res = res.popitem()
        assert block == ("ov",)
        assert simplify(res - ref).sympy is S.Zero

    def test_adc_amplitude(self):
        i, j, k, a, b, c = get_symbols("ijkabc")
        # due to the perm symmetry of X, the term will be expanded in 4 terms
        # that can be collected into a single term. Since X is an adc amplitude
        # we will get a special prefactor of 1/2
        # -> final pref: 1/2 * 4 * 1/2 = 1

        test = Expr(
            Rational(1, 2) *
            Amplitude(tensor_names.left_adc_amplitude, (a, b), (i, j)) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k)) *
            AntiSymmetricTensor(tensor_names.operator, (k,), (c,), 1)
        )
        ref = Expr(
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k)) *
            AntiSymmetricTensor(tensor_names.operator, (k,), (c,), 1)
        )
        res = remove_tensor(test, tensor_names.left_adc_amplitude)
        assert len(res) == 1
        block, res = res.popitem()
        assert block == ("oovv",)
        assert simplify(res - ref).sympy is S.Zero

    def test_multiple_tensors(self):
        i, j, k, a, b, c = get_symbols("ijkabc")
        # 1/2 t_ijkabc d_kc d_ijab -> 1/4 t_ijkabc d_ijab
        # (because d is symmetric)
        # 1/4 t_ijkabc d_ijab -> 1/2 t_ijkabc
        # 1/4 * 1/2 (d is symmetric) * 4 (we can collect the 4 terms) = 1/2

        test = Expr(
            Rational(1, 2) *
            AntiSymmetricTensor(tensor_names.operator, (i, j), (a, b), 1) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k)) *
            AntiSymmetricTensor(tensor_names.operator, (k,), (c,), 1)
        )
        ref = Expr(
            Rational(1, 2) *
            Amplitude(tensor_names.gs_amplitude, (a, b, c), (i, j, k))
        )
        res = remove_tensor(test, tensor_names.operator)
        assert len(res) == 1
        block, res = res.popitem()
        assert block == ("oovv", "ov")
        assert simplify(res - ref).sympy is S.Zero


class TestSimplify:
    def test_simplify_unitary(self):
        i, j, k = get_symbols('ijk')
        a, b, c = get_symbols('abc')

        # trivial positive: non-symmetric
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (k, j))
        )
        res = KroneckerDelta(i, k)
        assert simplify_unitary(expr, 'U').sympy == res
        # trivial positive: anti-symmetric
        expr = Expr(AntiSymmetricTensor('A', (i,), (j,))
                    * AntiSymmetricTensor('A', (k,), (j,)))
        assert simplify_unitary(expr, 'A').sympy - res is S.Zero

        # with remainder:
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (k, j))
            * NonSymmetricTensor('U', (a, b))
        )
        res = NonSymmetricTensor('U', (a, b)) * KroneckerDelta(i, k)
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero

        # U_ij U_ik U_ab U_ac = delta_jk * delta_bc
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, k))
            * NonSymmetricTensor('U', (a, b)) * NonSymmetricTensor('U', (a, c))
        )
        res = KroneckerDelta(j, k) * KroneckerDelta(b, c)
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero

        # switch index positions
        expr = Expr(
            NonSymmetricTensor('U', (j, i)) * NonSymmetricTensor('U', (k, i))
            * NonSymmetricTensor('U', (a, b)) * NonSymmetricTensor('U', (a, c))
        )
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero

        # index occurs at 3 objects
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, k))
            * NonSymmetricTensor('V', (i, j, k))
        )
        assert (simplify_unitary(expr, 'U') - expr).sympy is S.Zero

        # exponent > 1 and multiple occurences of index
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, j))
            * NonSymmetricTensor('U', (i, k))
        )
        res = NonSymmetricTensor('U', (i, k))
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero
