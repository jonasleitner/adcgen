from sympy import Rational, latex, sympify, Mul, S
from sympy.physics.secondquant import NO, F, Fd, wicks
from math import factorial

from .sympy_objects import AntiSymmetricTensor
from .indices import Indices, n_ov_from_space
from .misc import (cached_member, Inputerror,
                   process_arguments, transform_to_tuple, validate_input)
from .simplify import simplify
from .func import gen_term_orders
from .expr_container import Expr
from .operators import Operators


class GroundState:
    def __init__(self, hamiltonian, first_order_singles=False):
        if not isinstance(hamiltonian, Operators):
            raise Inputerror('Invalid hamiltonian.')
        self.indices = Indices()
        self.h = hamiltonian
        self.singles = first_order_singles

    @process_arguments
    @cached_member
    def energy(self, order):
        """Returns the ground state energy of specified order."""

        validate_input(order=order)

        h, rules = self.h.h0 if order == 0 else self.h.h1
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
        e = simplify(Expr(e))
        e = self.indices.substitute_with_generic(e)
        print(f"E^({order}) = {e}")
        return e.sympy

    def psi(self, order, braket):
        """Returns the ground state wave function. The type of the wave
           function needs to be specified via the braket string: 'bra' or
           'ket'."""
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
        """Return the n'th order ground state t-amplitude as defined by
           Rayleigh-Schrödinger perturbation theory."""
        from .intermediates import orb_energy

        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        validate_input(order=order, space=space, indices=indices)
        space = space[0]
        indices = indices[0]

        n_ov = n_ov_from_space(space)
        if n_ov["n_occ"] != n_ov["n_virt"]:
            raise Inputerror("Invalid space string for a MP t-amplitude: "
                             f"{space}.")
        # if the space is not present at the requested order return 0
        if n_ov["n_occ"] > 2 * order:
            return 0

        idx = self.indices.get_indices(indices)
        if n_ov["n_occ"] != len(idx["occ"]) or \
                n_ov["n_virt"] != len(idx["virt"]):
            raise Inputerror(f"Provided indices {indices} are not adequate for"
                             f" space {space}.")

        # build the denominator
        if len(idx['occ']) == 2:  # doubles amplitude: a+b-i-j
            occ_factor = -1
            virt_factor = +1
        else:  # any other amplitude: i-a // i+j+k-a-b-c // ...
            occ_factor = +1
            virt_factor = -1

        denom = 0
        for s in idx['occ']:
            denom += occ_factor * orb_energy(s)
        for s in idx['virt']:
            denom += virt_factor * orb_energy(s)

        # build the bra state: <k|
        bra = 1
        for s in idx["occ"]:
            bra *= Fd(s)
        for s in reversed(idx["virt"]):
            bra *= F(s)

        # construct <k|H1|psi^(n-1)>
        h1, rules = self.h.h1
        ret = bra * h1 * self.psi(order-1, "ket")
        ret = wicks(ret, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True)
        # subtract: - sum_{m=1} E_0^(m) * t_k^(n-m)
        terms = gen_term_orders(order=order, term_length=2, min_order=1)
        for o1, o2 in terms:
            # check if a t-amplitude of order o2 exists with special
            # treatment of the first order singles amplitude
            if (n_ov['n_occ'] > 2 * o2) or \
                    (n_ov['n_occ'] == 1 and o2 == 1 and not self.singles):
                continue
            if n_ov['n_occ'] == 2:  # doubles... special sign
                ret += (
                    self.energy(o1) *
                    AntiSymmetricTensor(f"t{o2}", idx["virt"], idx["occ"])
                )
            else:
                ret -= (
                    self.energy(o1) *
                    AntiSymmetricTensor(f"t{o2}", idx["virt"], idx["occ"])
                )
        return ret / denom

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
        res = simplify(Expr(res))
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
        # get the operator
        op, rules = self.h.operator(opstring=opstring)
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
                i1 = wfn[term[0]]['bra'] * op * wfn[term[1]]['ket']
                d += wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
            res += (norm * d).expand()
        return simplify(Expr(res)).sympy

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
