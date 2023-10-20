from sympy_adc.rules import Rules
from sympy_adc.expr_container import expr
from sympy_adc.sympy_objects import AntiSymmetricTensor
from sympy_adc.indices import get_symbols

from sympy import S


class TestRules:

    # some indices to use in all tests
    i, j = get_symbols('ij')
    a, b = get_symbols('ab')

    def test_empty(self):
        term = AntiSymmetricTensor('f', (self.i,), (self.j,), 0)
        term = expr(term)
        r = Rules(forbidden_tensor_blocks={})
        new_term = r.apply(term)
        assert (new_term - term).sympy is S.Zero

        r = Rules()
        new_term = r.apply(term)
        assert (new_term - term).sympy is S.Zero

    def test_negative(self):
        term = AntiSymmetricTensor('f', (self.i,), (self.a,), 0)
        term = expr(term)

        r = Rules(forbidden_tensor_blocks={'V': ['ov', 'oo']})
        new_term = r.apply(term)
        assert (new_term - term).sympy is S.Zero

        r = Rules(forbidden_tensor_blocks={'f': ['vo', 'oo', 'vv']})
        assert (new_term - term).sympy is S.Zero

    def test_positive(self):
        term1 = AntiSymmetricTensor('V', (self.i, self.j), (self.a, self.b))
        term1 *= AntiSymmetricTensor('f', (self.j,), (self.b,))
        term2 = AntiSymmetricTensor('V', (self.i, self.a), (self.j, self.b))
        term2 *= AntiSymmetricTensor('f', (self.i,), (self.j,))
        terms = expr(term1 + term2)

        # remove both terms
        r = Rules(forbidden_tensor_blocks={'V': ['oovv'], 'f': ['oo']})
        new_term = r.apply(terms)
        assert (new_term).sympy is S.Zero

        # remove 1 term
        r = Rules(forbidden_tensor_blocks={'V': ['ooov'], 'f': ['ov']})
        new_term = r.apply(terms)
        assert new_term.sympy - term2 is S.Zero

        r = Rules(forbidden_tensor_blocks={'V': ['ovov'], 'f': ['oo']})
        new_term = r.apply(terms)
        assert new_term.sympy - term1 is S.Zero