from sympy import Rational, latex, sympify, Mul, S
from sympy.physics.secondquant import AntiSymmetricTensor, NO, F, Fd, wicks
from math import factorial

from indices import indices, n_ov_from_space
from misc import (cached_member, cached_property, Inputerror,
                  process_arguments, transform_to_tuple, validate_input)
from simplify import simplify
from func import gen_term_orders


class Hamiltonian:
    def __init__(self):
        self.indices = indices()

    @cached_property
    def h0(self):
        p, q = self.indices.get_indices('pq')['general']
        f = AntiSymmetricTensor('f', (p,), (q,))
        pq = Fd(p) * F(q)
        h0 = f * pq
        print("H0 = ", latex(h0))
        return h0

    @cached_property
    def h1(self):
        p, q, r, s = self.indices.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H1
        occ = self.indices.get_generic_indices(n_o=1)['occ'][0]
        v1 = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        h1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(h1))
        return h1

    @process_arguments
    @cached_member
    def operator(self, opstring):
        """Constructs an arbitrary operator. The amount of creation (c) and
           annihilation (a) operators must be given by opstring. For example
           'ccaa' will return a two particle operator.
           """
        validate_input(opstring=opstring)
        n_create = opstring.count('c')
        idx = self.indices.get_generic_indices(n_g=len(opstring))["general"]
        create = idx[:n_create]
        annihilate = idx[n_create:]

        pref = Rational(1, factorial(len(create)) * factorial(len(annihilate)))
        d = AntiSymmetricTensor('d', create, annihilate)
        op = Mul(*[Fd(s) for s in create]) * \
            Mul(*[F(s) for s in reversed(annihilate)])
        return pref * d * op


class ground_state:
    def __init__(self, hamiltonian, first_order_singles=False):
        if not isinstance(hamiltonian, Hamiltonian):
            raise Inputerror('Invalid hamiltonian.')
        self.indices = indices()
        self.h = hamiltonian
        self.singles = first_order_singles

    @process_arguments
    @cached_member
    def energy(self, order):
        """Returns the ground state energy of specified order."""

        validate_input(order=order)

        h = self.h.h0 if order == 0 else self.h.h1
        bra = self.psi(order=0, braket="bra")
        ket = self.psi(order=0, braket='ket') if order == 0 else \
            self.psi(order=order-1, braket='ket')
        e = bra * h * ket
        e = wicks(e, keep_only_fully_contracted=True,
                  simplify_kronecker_deltas=True)
        # option 1: return the not simplified energy -> will give a lot more
        #           terms later on
        # option 2: simplify the energy expression and replace the indices with
        #           new, generic indices
        # guess option 2 is nicer, because energy is more readable and shorter
        e = simplify(e)
        e = self.indices.substitute_with_generic(e)
        print(f"E^({order}) = {e}")
        return e.sympy

    def psi(self, order, braket):
        """Retuns the ground state wave function. The type of the wave function
           needs to be specified via the braket string: 'bra' or 'ket'."""
        # Can't cache ground state wave function!
        # Leads to an error for terms of the form:
        #  |1><2|1>... the two |1> need to have different indices!!
        #  |1><1|2>... |1> and |2> can't share indices
        #   -> Therefore, each time a gs wavefunction is requested new indices
        #      need to be used.
        #      But one can still use overlapping indices within a wavefunction
        #      e.g. singles: ia, doubles ijab, triples ijkabc

        validate_input(order=order, braket=braket)

        # catch 0th order wavefunction
        if order == 0:
            print(f"Build gs({order}) {braket} = 1")
            return sympify(1)

        # generalized gs wavefunction generation
        tensor_string = {
            "bra": f"t{order}cc",
            "ket": f"t{order}"
        }
        get_ov = {
            "bra": lambda ov: [other for other in ["occ", "virt"]
                               if other != ov][0],
            "ket": lambda ov: ov
        }
        get_ov = get_ov[braket]
        idx = {'occ': [], 'virt': []}
        psi = 0
        for excitation in range(1, order * 2 + 1):
            # generate 1 additional o/v symbol pair, e.g. singles: ia,
            # doubles: ijab, etc. -> reuse the indices from the lower spaces.
            additional_idx = self.indices.get_generic_indices(n_o=1, n_v=1)
            idx['occ'].extend(additional_idx['occ'])
            idx['virt'].extend(additional_idx['virt'])
            # skip singles for the first order wavefunction if
            # they are not requested
            if order == 1 and not self.singles and excitation == 1:
                continue
            t = AntiSymmetricTensor(
                tensor_string[braket], idx["virt"], idx["occ"]
            )
            operators = Mul(*[Fd(s) for s in idx[get_ov('virt')]]) * \
                Mul(*[F(s) for s in reversed(idx[get_ov('occ')])])
            # prefactor for lifting index restrictions
            prefactor = Rational(1, factorial(excitation) ** 2)
            # For signs: Decided to subtract all Doubles to stay consistent
            #            with existing implementations of the amplitudes.
            #            The remaining amplitudes (S/T/Q...) are added!
            #            (denominator for Triples: i+j+k-a-b-c
            #                             Doubles: a+b-i-j)
            if excitation == 2:
                psi -= prefactor * t * NO(operators)
            else:
                psi += prefactor * t * NO(operators)
        print(f"Build gs({order}) {braket} = ", latex(psi))
        return psi

    @process_arguments
    @cached_member
    def amplitude(self, order, space, indices):
        # TODO: properly implement this (explicit denominator + energies)
        # not working really. The denominator is only represented as symbolic
        # delta without any indices and therefore it is not possible to
        # subtract: - E*t properly
        # atm the subtraction is ommited completely and needs to be done
        # manually afterwards.
        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        validate_input(order=order, space=space, indices=indices)
        space = space[0]
        indices = indices[0]

        n_ov = n_ov_from_space(space)
        if n_ov["n_occ"] != n_ov["n_virt"]:
            raise Inputerror("Invalid space string for a MP-t amplitude: "
                             f"{space}.")
        # if the space is not present at the requested order return 0
        if n_ov["n_occ"] > 2 * order:
            return 0

        idx = self.indices.get_indices(indices)
        if n_ov["n_occ"] != len(idx["occ"]) or \
                n_ov["n_virt"] != len(idx["virt"]):
            raise Inputerror(f"Provided indices {indices} are not adequate for"
                             f" space {space}.")
        bra = 1
        # need to build a symmetric tensor object to represent the denominator
        # properly?
        denom = AntiSymmetricTensor("delta", tuple(), tuple())
        for s in idx["occ"]:
            bra *= Fd(s)
        for s in reversed(idx["virt"]):
            bra *= F(s)

        # construct <k|H1|psi^(n-1)>
        ret = bra * self.h.h1 * self.psi(order-1, "ket")
        ret = wicks(ret, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True)
        # subtract: - sum_{m=1} E_0^(m) * t_k^(n-m)
        terms = gen_term_orders(order=order, term_length=2, min_order=1)
        for term in terms:
            if any(o == 0 for o in term):
                continue
            # possibly also only represent E symbolic?
            # ret -= self.energy(term[0]) * \
            #     AntiSymmetricTensor(f"t{term[1]}", idx["occ"], idx["virt"])
        return (ret * denom).expand()

    def overlap(self, order):
        """Computes the ground state overlap matrix."""
        validate_input(order=order)

        # catch zeroth order
        if order == 0:
            return sympify(1)

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        for term in orders:
            # each wfn is requested only once -> no need to precompute and
            # cache
            i1 = self.psi(order=term[0], braket='bra') * \
                self.psi(order=term[1], braket='ket')
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        # simplify the result by permuting contracted indices
        res = simplify(res)
        print(f"Build GS S^({order}) = {res}")
        return res.sympy

    @process_arguments
    @cached_member
    def expectation_value(self, order, opstring):
        """Computes the expectation value for a given operator. The operator
           is defined by the number of creation and annihilation operators
           that must be provided as string. For example a two particle
           operator is defined as 'ccaa'.
           """
        validate_input(order=order, opstring=opstring)
        # - import all mp wavefunctions. It should be possible here, because
        #   it is not possible to obtain a term |1>*x*|1>.
        wfn = {}
        for o in range(order + 1):
            wfn[o] = {}
            for bk in ["bra", "ket"]:
                wfn[o][bk] = self.psi(order=o, braket=bk)

        # better to generate twice orders for length 2 than once for length 3
        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm factor
            orders_d = gen_term_orders(
                order=norm_term[1], term_length=2, min_order=0
            )
            d = 0
            for term in orders_d:
                i1 = (wfn[term[0]]['bra'] *
                      self.h.operator(opstring=opstring) *
                      wfn[term[1]]['ket'])
                d += wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
            res += (norm * d).expand()
        return simplify(res).sympy

    def norm_factor(self, order):
        """Computes all nth order terms of:
           1 - sum_i S^(i) + (sum_i S^(i))^2 - ...
           This accounts in the one and two particle dm for the
           normalization of higher order terms.
           S = a^2 sum_i=0 S^(i) = 1 -> a^2 = [sum_i=0 S^(i)]^(-1)
           """
        # This can not be cached!
        # in case there is something like a(2)*a(2)*x
        # do the two a(2) need to be different?
        #   all a(n) only consist of t-amplitudes and all indices are
        #   contracted
        # a(2) = 0.25*t_d^(2)
        # a(2)*a(2) = 1/16 * t_d^(2) * t_d'^(2)
        #  -> no caching allowed
        # Then it is also not possible to cache the overlap matrix
        validate_input(order=order)

        taylor_expansion = self.expand_norm_factor(order=order, min_order=2)
        norm_factor = 0
        for pref, termlist in taylor_expansion:
            for term in termlist:
                i1 = pref
                for o in term:
                    i1 *= self.overlap(o)
                    if i1 is S.Zero:
                        break
                norm_factor += i1.expand()
        print(f"norm_factor^({order}): {latex(norm_factor)}")
        return norm_factor

    def expand_norm_factor(self, order, min_order=2):
        """Expands f = (1 + x)^(-1) in a taylor series, where x is defined as
           x = sum_i S^(i) - the sum of overlap matrices of order i.
           The parameter min_order defines the first non_vanishing contribution
           to S. All lower contributions are assumed to give either 1 or 0.
           """
        from sympy import symbols, diff, nsimplify

        validate_input(order=order, min_order=min_order)
        if min_order == 0:
            raise Inputerror("A minimum order of 0 does not make sense here.")

        # below min_order all orders of the overlap matrix should be 0.
        # only the zeroth order contribution should be 1
        # -> obtain 0 or 1 from the overlap function -> handled automatically
        if order < min_order:
            return [(1, [(order,)])]

        x = symbols('x')
        f = (1 + x) ** -1.0
        ret = []
        for exp in range(1, order//min_order + 1):
            f = diff(f, x)
            pref = nsimplify(f.subs(x, 0) / factorial(exp), rational=True)
            orders = gen_term_orders(
                order=order, term_length=exp, min_order=min_order
            )
            ret.append((pref, orders))
        return ret
