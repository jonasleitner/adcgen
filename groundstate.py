from sympy import nsimplify, symbols, Rational, latex, Dummy, diff, sympify
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, wicks, substitute_dummies
)
from math import factorial
from indices import pretty_indices, indices
from misc import cached_member, cached_property, Inputerror

from isr import gen_order_S, get_order_two


class Hamiltonian:
    def __init__(self, canonical=True):
        self.canonical = canonical

    @cached_property
    def h0(self):
        p = symbols('p', cls=Dummy)
        if self.canonical:
            f = AntiSymmetricTensor('f', (p,), (p,))
            pp = Fd(p) * F(p)
            h0 = f * pp
        else:
            q = symbols('q', cls=Dummy)
            f = AntiSymmetricTensor('f', (p,), (q,))
            pq = Fd(p) * F(q)
            h0 = f * pq
        print("H0 = ", latex(h0))
        return h0

    @cached_property
    def h1(self):
        p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        # this symbol is reserved for h1
        o42 = symbols('o42', below_fermi=True, cls=Dummy)
        v1 = AntiSymmetricTensor('V', (p, o42), (q, o42))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        h1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(h1))
        return h1

    @cached_property
    def one_particle(self):
        p, q = symbols('p,q', cls=Dummy)
        pq = Fd(p) * F(q)
        d = AntiSymmetricTensor('d', (p,), (q,))
        return d * pq

    @cached_property
    def two_particle(self):
        p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        d = AntiSymmetricTensor('d', (p, q), (r, s))
        return Rational(1, 4) * d * pqsr

    @cached_property
    def ip_transition(self):
        p = symbols('p', cls=Dummy)
        d = AntiSymmetricTensor('d', tuple(), (p,))
        return d * F(p)

    @cached_property
    def ea_transition(self):
        p = symbols('p', cls=Dummy)
        d = AntiSymmetricTensor('d', (p,), tuple())
        return d * Fd(p)

    def dip_transition(self):
        p, q = symbols('p,q', cls=Dummy)
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
            raise Inputerror("Order for obtaining ground state energy needs to"
                             f" be a int. {type(order)} is not valid.")

        def H(o): return self.h.h0 if o == 0 else self.h.h1
        h = H(order)
        bra = self.psi(0, "bra")
        def Ket(o): return self.psi(0, "ket") if o == 0 else \
            self.psi(order-1, "ket")
        ket = Ket(order)
        e = bra * h * ket
        e = wicks(e, keep_only_fully_contracted=True,
                  simplify_kronecker_deltas=True)
        # new_indices required here
        e = substitute_dummies(e, new_indices=True,
                               pretty_indices=pretty_indices)
        e = self.indices.substitute_with_generic_indices(e)
        print(f"E^({order}) = {latex(e)}")
        return e

    def pretty_energy(self, order):
        return substitute_dummies(self.energy(order), new_indices=True,
                                  pretty_indices=pretty_indices)

    @cached_member
    def psi(self, order, braket):
        """Retuns the ground state wavefunction of the requested order.
           The type (bra or ket) of the requested wavefunction needs to
           be provided as str.
           """

        if not isinstance(order, int):
            raise Inputerror("Order for obtaining ground state wavefunction"
                             f"needs to of type int. {type(order)} is not "
                             "valid.")
        if braket not in ["ket", "bra"]:
            raise Inputerror("Only possible to build 'bra' or 'ket' gs "
                             f"wavefunction {braket} is not valid")

        # catch 0th order wavefunction
        if order == 0:
            print(f"Build gs^({order} {braket} = 1")
            return sympify(1)

        # generalized gs wavefunction generation
        tensor_string = {
            "bra": f"t{order}cc",
            "ket": f"t{order}"
        }
        get_ov = {
            "bra": lambda ov: [other for other in ["occ", "virt"]
                               if other != ov],
            "ket": lambda ov: ov
        }
        psi = 0
        for excitation in range(1, order * 2 + 1):
            # skip singles for the first order wavefunction if
            # they are not desired.
            if order == 1 and not self.singles and excitation == 1:
                continue
            idx = self.indices.get_gs_indices(
                braket, n_occ=excitation, n_virt=excitation
            )
            t = AntiSymmetricTensor(
                tensor_string[braket], tuple(idx["virt"]), tuple(idx["occ"])
            )
            operators = 1
            for symbol in idx["".join(get_ov[braket]('virt'))]:
                operators *= Fd(symbol)
            for symbol in reversed(idx["".join(get_ov[braket]('occ'))]):
                operators *= F(symbol)
            # prefactor for lifting index restrictions
            prefactor = Rational(1, factorial(excitation) ** 2)
            psi += - prefactor * t * NO(operators)
        print(f"Build gs^({order}) {braket} = ", latex(psi))
        return psi

    @cached_member
    def overlap(self, order):
        """Computes the ground state overlap matrix."""

        # catch zeroth order
        if order == 0:
            return sympify(1)
        # import gs wavefunctions
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
        print(f"Build GS S^({order}) = {latex(res)}")
        return res

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
            a = self.account_for_norm(term[0])
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
            a = self.account_for_norm(term[0])
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

    @cached_member
    def account_for_norm(self, order):
        """Computes all nth order terms of:
           1 - sum_i S^(i) + (sum_i S^(i))^2 - ...
           This accounts in the one and two particle dm for the
           normalization of higher order terms.
           S = a^2 sum_i=0 S^(i) = 1 -> a^2 = [sum_i=0 S^(i)]^(-1)
           """

        prefactors, orders = self.expand_norm_factor(order, min_order=2)
        norm_factor = 0
        for exponent, termlist in orders.items():
            for o_tuple in termlist:
                i1 = prefactors[exponent]
                for o in o_tuple:
                    i1 *= self.overlap(o)
                norm_factor += i1.expand()
        return norm_factor

    def expand_norm_factor(self, order, min_order=2):
        """Expands f = (1 + x)^(-1) in a taylor series, where x is defined as
           x = sum_i S^(i) - the sum of overlap matrices of order i.
           The parameter min_order defines the first non_vanishing contribution
           to S. All lower contributions are assumed to give either 1 or 0.
           """

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
