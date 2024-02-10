from sympy_adc.simplify import simplify_unitary
from sympy_adc.sympy_objects import NonSymmetricTensor, AntiSymmetricTensor, \
    Delta
from sympy_adc.expr_container import Expr
from sympy_adc.indices import get_symbols

from sympy import S


class TestSimplify:
    def test_simplify_unitary(self):
        i, j, k = get_symbols('ijk')
        a, b, c = get_symbols('abc')

        # trivial positive: non-symmetric
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (k, j))
        )
        res = Delta(i, k)
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
        res = NonSymmetricTensor('U', (a, b)) * Delta(i, k)
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero

        # U_ij U_ik U_ab U_ac = delta_jk * delta_bc
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, k))
            * NonSymmetricTensor('U', (a, b)) * NonSymmetricTensor('U', (a, c))
        )
        res = Delta(j, k) * Delta(b, c)
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
        print(simplify_unitary(expr, 'U'))
        assert simplify_unitary(expr, 'U').sympy - res is S.Zero
