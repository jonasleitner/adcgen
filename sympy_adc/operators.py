from .misc import (cached_property, process_arguments, cached_member,
                   validate_input)
from .indices import indices
from .rules import Rules
from .sympy_objects import AntiSymmetricTensor

from sympy import Rational, factorial, Mul, latex
from sympy.physics.secondquant import Fd, F


class Operators:
    def __init__(self, variant='mp'):
        self._indices = indices()
        self._variant = variant

    @cached_property
    def h0(self):
        if self._variant == 'mp':
            return self.mp_h0()
        elif self._variant == 're':
            return self.re_h0()
        else:
            raise NotImplementedError(
                f"H0 not implemented for {self._variant}"
            )

    @cached_property
    def h1(self):
        if self._variant == 'mp':
            return self.mp_h1()
        elif self._variant == 're':
            return self.re_h1()
        else:
            raise NotImplementedError(
                f"H1 not implented for {self._variant}"
            )

    @process_arguments
    @cached_member
    def operator(self, opstring):
        """Constructs an arbitrary operator. The amount of creation (c) and
           annihilation (a) operators must be given by opstring. For example
           'ccaa' will return a two particle operator.
           """
        validate_input(opstring=opstring)
        n_create = opstring.count('c')
        idx = self._indices.get_generic_indices(n_g=len(opstring))["general"]
        create = idx[:n_create]
        annihilate = idx[n_create:]

        pref = Rational(1, factorial(len(create)) * factorial(len(annihilate)))
        d = AntiSymmetricTensor('d', create, annihilate)
        op = Mul(*[Fd(s) for s in create]) * \
            Mul(*[F(s) for s in reversed(annihilate)])
        return pref * d * op, Rules()

    @staticmethod
    def mp_h0():
        idx_cls = indices()
        p, q = idx_cls.get_indices('pq')['general']
        f = AntiSymmetricTensor('f', (p,), (q,))
        pq = Fd(p) * F(q)
        h0 = f * pq
        print("H0 = ", latex(h0))
        return h0, Rules()

    @staticmethod
    def mp_h1():
        idx_cls = indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H1
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]
        v1 = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        h1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(h1))
        return h1, Rules()

    @staticmethod
    def re_h0():
        idx_cls = indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]

        f = AntiSymmetricTensor('f', (p,), (q,))
        piqi = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor('V', (p, q), (r, s))
        op_pq = Fd(p) * F(q)
        op_pqsr = Fd(p) * Fd(q) * F(s) * F(r)

        h0 = f * op_pq - piqi * op_pq + Rational(1, 4) * pqrs * op_pqsr
        print("H0 = ", latex(h0))
        # construct the rules for forbidden blocks in H0
        # we are not in a real orbital basis!! -> More canonical blocks
        rules = Rules(forbidden_tensor_blocks={
            'f': ('ov', 'vo'),
            'V': ('ooov', 'oovv', 'ovvv', 'ovoo', 'vvoo', 'vvov')
        })
        return h0, rules

    @staticmethod
    def re_h1():
        idx_cls = indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]

        f = AntiSymmetricTensor('f', (p,), (q,))
        piqi = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor('V', (p, q), (r, s))
        op_pq = Fd(p) * F(q)
        op_pqsr = Fd(p) * Fd(q) * F(s) * F(r)

        h1 = f * op_pq - piqi * op_pq + Rational(1, 4) * pqrs * op_pqsr
        print("H1 = ", latex(h1))
        # construct the rules for forbidden blocks in H1
        rules = Rules(forbidden_tensor_blocks={
            'f': ['oo', 'vv'],
            'V': ['oooo', 'ovov', 'vvvv']
        })
        return h1, rules
