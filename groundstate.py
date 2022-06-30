from sympy import nsimplify, Rational, latex, diff, sympify
from sympy.physics.secondquant import AntiSymmetricTensor, NO, F, Fd, wicks
from math import factorial

from indices import indices, n_ov_from_space
from misc import cached_member, cached_property, Inputerror, transform_to_tuple
from isr import gen_order_S, get_order_two
from simplify import simplify


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
        # this symbol is reserved for h1
        o42 = self.indices.get_indices('o42')['occ'][0]
        v1 = AntiSymmetricTensor('V', (p, o42), (q, o42))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        h1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(h1))
        return h1

    @cached_property
    def one_particle(self):
        p, q = self.indices.get_indices('pq')['general']
        pq = Fd(p) * F(q)
        d = AntiSymmetricTensor('d', (p,), (q,))
        return d * pq

    @cached_property
    def two_particle(self):
        p, q, r, s = self.indices.get_indices('pqrs')['general']
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        d = AntiSymmetricTensor('d', (p, q), (r, s))
        return Rational(1, 4) * d * pqsr

    @cached_property
    def ip_transition(self):
        p = self.indices.get_indices('p')['general'][0]
        d = AntiSymmetricTensor('d', tuple(), (p,))
        return d * F(p)

    @cached_property
    def ea_transition(self):
        p = self.indices.get_indices('p')['general'][0]
        d = AntiSymmetricTensor('d', (p,), tuple())
        return d * Fd(p)

    def dip_transition(self):
        p, q = self.indices.get_indices('pq')['general']
        d = AntiSymmetricTensor('d', tuple(), (p, q))
        return Rational(1, 2) * d * F(p) * F(q)


class ground_state:
    def __init__(self, hamiltonian, first_order_singles=False):
        self.indices = indices()
        self.h = hamiltonian
        self.singles = first_order_singles

    @cached_member
    def energy(self, order):
        """Returns the ground state energy of specified order."""

        if not isinstance(order, int):
            raise Inputerror("Order for needs to be a int. "
                             f"{type(order)} is not valid.")

        def H(o): return self.h.h0 if o == 0 else self.h.h1
        h = H(order)
        bra = self.psi(0, "bra")
        def Ket(o): return self.psi(0, "ket") if o == 0 else \
            self.psi(order-1, "ket")
        ket = Ket(order)
        e = bra * h * ket
        e = wicks(e, keep_only_fully_contracted=True,
                  simplify_kronecker_deltas=True)
        # option 1: return the not simplified energy -> will give a lot more
        #           terms later on
        # option 2: simplify the energy expression and replace the indices with
        #           new, generic indices
        # guess option 2 is nicer, because energy more readable and better
        # performance
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

        if not isinstance(order, int):
            raise Inputerror("Order for obtaining ground state wave function"
                             f"needs to of type int. {type(order)} is not "
                             "valid.")
        if braket not in ["ket", "bra"]:
            raise Inputerror("Only possible to build 'bra' or 'ket' gs "
                             f"wave function {braket} is not valid")

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
            operators = 1
            for symbol in idx[get_ov('virt')]:
                operators *= Fd(symbol)
            for symbol in reversed(idx[get_ov('occ')]):
                operators *= F(symbol)
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

    @cached_member
    def amplitude(self, order, space, indices):
        # not working really. The denominator is only represented as symbolic
        # delta without any indices and therefore it is not possible to
        # subtract: - E*t properly
        # atm the subtraction is ommited completely and needs to be done
        # manually afterwards.
        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        if len(space) != 1 or len(indices) != 1:
            raise Inputerror("Expected only 1 space and indice string. "
                             f"Got space {space}, indices {indices}.")
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
        terms = get_order_two(order)
        for term in terms:
            if any(o == 0 for o in term):
                continue
            # possibly also only represent E symbolic?
            # ret -= self.energy(term[0]) * \
            #     AntiSymmetricTensor(f"t{term[1]}", idx["occ"], idx["virt"])
        return (ret * denom).expand()

    def overlap(self, order):
        """Computes the ground state overlap matrix."""

        # catch zeroth order
        if order == 0:
            return sympify(1)
        # import and save gs wave functions to lower workload. At this point
        # a single variant for a bra/ket wave function of order n is fine
        wfn = {}
        for o in range(order + 1):
            if o not in wfn:
                wfn[o] = {}
            for bk in ["bra", "ket"]:
                wfn[o][bk] = self.psi(o, bk)

        orders = get_order_two(order)
        res = 0
        for term in orders:
            i1 = wfn[term[0]]["bra"] * wfn[term[1]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        # simplify the result by permuting contracted indices
        # TODO: This should not introduce an error
        res = simplify(res)
        print(f"Build GS S^({order}) = {res}")
        return res.sympy

    @cached_member
    def one_particle_operator(self, order):
        """Computes the expectation value of a one particle operator
           for the ground state. Opted to implement the expectation value
           instead of the OPDM, beacuse the results are more easily
           readable (and the OPDM may be easily extracted from the result).
           """

        res = 0
        orders = get_order_two(order)
        for term in orders:
            d = self.d_one_particle(term[1])
            a = self.norm_factor(term[0])
            res += (a * d).expand()
        return res

    @cached_member
    def two_particle_operator(self, order):
        """Computes the expectation value of a two particle operator
           for the ground state.
           Did not check any results obtained with that function!
           Also it may be necessary to introduce a prefactor.
           """

        res = 0
        orders = get_order_two(order)
        for term in orders:
            d = self.d_two_particle(term[1])
            a = self.norm_factor(term[0])
            res += (a * d).expand()
        return res

    @cached_member
    def d_one_particle(self, order):
        """Computes the matrix element
           sum_pq d_{pq} <psi|pq|psi>^(n).
           """

        wfn = {}
        for o in range(order + 1):
            wfn[o] = {}
            for bk in ["bra", "ket"]:
                wfn[o][bk] = self.psi(o, bk)

        orders = get_order_two(order)
        res = 0
        for term in orders:
            i1 = (wfn[term[0]]["bra"] * self.h.one_particle *
                  wfn[term[1]]["ket"])
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    @cached_member
    def d_two_particle(self, order):
        """Computes the matrix element
           1/4 sum_{pqrs} d_{pqsr} <psi|pqsr|psi>^(n).
           """

        wfn = {}
        for o in range(order + 1):
            wfn[o] = {}
            for bk in ["bra", "ket"]:
                wfn[o][bk] = self.psi(o, bk)

        orders = get_order_two(order)
        res = 0
        for term in orders:
            i1 = (wfn[term[0]]["bra"] * self.h.two_particle *
                  wfn[term[1]]["ket"])
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

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
        # a(2)*a(2) = 1/16 * t_d^(4) or 1/16 * t_d^(2) * t_d'^(2)
        # I guess the second should be correct -> no caching allowed
        # Then it is also not possible to cache the overlap matrix

        prefactors, orders = self.expand_norm_factor(order, min_order=2)
        norm_factor = 0
        for exponent, termlist in orders.items():
            for o_tuple in termlist:
                i1 = prefactors[exponent]
                for o in o_tuple:
                    i1 *= self.overlap(o)
                norm_factor += i1.expand()
        print(f"norm_factor^({order}): {latex(norm_factor)}")
        return norm_factor

    def expand_norm_factor(self, order, min_order=2):
        """Expands f = (1 + x)^(-1) in a taylor series, where x is defined as
           x = sum_i S^(i) - the sum of overlap matrices of order i.
           The parameter min_order defines the first non_vanishing contribution
           to S. All lower contributions are assumed to give either 1 or 0.
           """
        from sympy import symbols

        x = symbols('x')
        f = (1 + x) ** -1.0
        intermediate = f
        diffs = {0: 1}
        for o in range(1, int(order/min_order) + 1):
            intermediate = diff(intermediate, x)
            diffs[o] = nsimplify(intermediate.subs(x, 0) * 1 / factorial(o))
        orders = gen_order_S(order, min_order=min_order)
        # if order below min_order just return the order, i.e. the overlap
        # matrix of order x will be used (which then sould give either 1 or 0)
        if not orders:
            orders[0] = [(order,)]
        return (diffs, orders)
