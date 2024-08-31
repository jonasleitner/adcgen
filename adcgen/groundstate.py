from sympy import Rational, latex, sympify, S
from sympy.physics.secondquant import NO, Dagger
from math import factorial

from .sympy_objects import Amplitude
from .indices import Indices, n_ov_from_space
from .misc import cached_member, Inputerror, validate_input
from .simplify import simplify
from .func import gen_term_orders, wicks
from .expr_container import Expr
from .operators import Operators
from .logger import logger
from .tensor_names import tensor_names


class GroundState:
    """
    Constructs ground state expressions using Rayleigh-Schrödinger
    perturbation theory.

    Parameters
    ----------
    hamiltonian : Operators
        An Operators instance to request the partitioned Hamiltonian and
        other Operators.
    first_order_singles : bool, optional
        If set, the first order wavefunction will contain single amplitudes.
        (Defaults to False)
    """
    def __init__(self, hamiltonian: Operators,
                 first_order_singles: bool = False):
        if not isinstance(hamiltonian, Operators):
            raise Inputerror("Invalid hamiltonian.")
        self.indices = Indices()
        self.h = hamiltonian
        self.singles = first_order_singles

    @cached_member
    def energy(self, order: int):
        """
        Constructs an expression for the n'th-order ground state energy
        contribution.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        """
        # NOTE: this function assumes a block diagonal H0
        # in the general case we have to include <0|H0|n>

        validate_input(order=order)

        h, rules = self.h.h0 if order == 0 else self.h.h1
        bra = self.psi(order=0, braket="bra")
        ket = self.psi(order=0, braket='ket') if order == 0 else \
            self.psi(order=order-1, braket='ket')
        e = bra * h * ket
        e = wicks(e, simplify_kronecker_deltas=True, rules=rules)
        # option 1: return the not simplified energy -> will give a lot more
        #           terms later on
        # option 2: simplify the energy expression and replace the indices with
        #           new, generic indices
        # guess option 2 is nicer, because energy is more readable and shorter
        e = simplify(Expr(e))
        e = self.indices.substitute_with_generic(e)
        logger.debug(f"E^({order}) = {e}")
        return e.sympy

    def psi(self, order: int, braket: str):
        """
        Constructs the n'th-order ground state wavefunction without inserting
        definitions of the respective ground state amplitudes.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        braket: str
            Possible values: 'bra', 'ket'. Defines whether a bra or ket
            wavefunction is constructed.
        """
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
            logger.debug(f"gs({order}) {braket} = 1")
            return sympify(1)

        # generalized gs wavefunction generation
        tensor_name = f"{tensor_names.gs_amplitude}{order}"
        if braket == "bra":
            tensor_name += "cc"
        idx = self.indices.get_generic_indices(n_o=2*order, n_v=2*order)
        psi = 0
        for excitation in range(1, order * 2 + 1):
            # skip singles for the first order wavefunction if
            # they are not requested
            if order == 1 and not self.singles and excitation == 1:
                continue
            # build tensor
            virt: list = idx["virt"][:excitation]
            occ: list = idx["occ"][:excitation]
            t = Amplitude(tensor_name, virt, occ)
            # build operators
            operators = self.h.excitation_operator(creation=virt,
                                                   annihilation=occ,
                                                   reverse_annihilation=True)
            if braket == "bra":
                operators = Dagger(operators)
            # prefactor for lifting index restrictions
            prefactor = Rational(1, factorial(excitation) ** 2)
            # For signs: Decided to subtract all Doubles to stay consistent
            #            with existing implementations of the amplitudes.
            #            The remaining amplitudes (S/T/Q...) are added!
            #            (denominator for Triples: i+j+k-a-b-c
            #                             Doubles: a+b-i-j)
            if excitation == 2:  # doubles
                psi -= prefactor * t * NO(operators)
            else:
                psi += prefactor * t * NO(operators)
        logger.debug(f"gs({order}) {braket} = {latex(psi)}")
        return psi

    def amplitude(self, order: int, space: str, indices: str):
        """
        Constructs the n'th-order expression for the ground state t-amplitudes.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space, e.g., 'ph' or 'pphh' for singly or doubly
            excited configurations.
        indices : str
            The indices the t-amplitude.
        """
        variant = self.h._variant
        if variant == 'mp':
            return self.mp_amplitude(order, space, indices)
        elif variant == 're':
            return self.amplitude_residual(order, space, indices)
        else:
            raise NotImplementedError("Amplitudes not implemented for "
                                      f"{self.h._variant}")

    @cached_member
    def mp_amplitude(self, order: int, space: str, indices: str):
        """
        Constructs the closed n'th-order expression for the MP t-amplitudes.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space, e.g., 'ph' or 'pphh' for singly or doubly
            excited configurations.
        indices : str
            The indices of the constructed t-amplitude.
        """
        from .intermediates import orb_energy

        validate_input(order=order, space=space, indices=indices)

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
        bra = self.h.excitation_operator(creation=idx["occ"],
                                         annihilation=idx["virt"],
                                         reverse_annihilation=True)

        # construct <k|H1|psi^(n-1)>
        h1, rules = self.h.h1
        ret = bra * h1 * self.psi(order-1, "ket")
        ret = wicks(ret, simplify_kronecker_deltas=True, rules=rules)
        # subtract: - sum_{m=1} E_0^(m) * t_k^(n-m)
        terms = gen_term_orders(order=order, term_length=2, min_order=1)
        for o1, o2 in terms:
            # check if a t-amplitude of order o2 exists with special
            # treatment of the first order singles amplitude
            if (n_ov['n_occ'] > 2 * o2) or \
                    (n_ov['n_occ'] == 1 and o2 == 1 and not self.singles):
                continue
            name = f"{tensor_names.gs_amplitude}{o2}"
            contrib = (
                self.energy(o1) * Amplitude(name, idx["virt"], idx["occ"])
            ).expand()
            if n_ov['n_occ'] == 2:  # doubles... special sign
                ret += contrib
            else:
                ret -= contrib
        return ret / denom

    @cached_member
    def amplitude_residual(self, order: int, space: str, indices: str):
        """
        Constructs the n'th-order residual for ground state amplitudes.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space, e.g., 'ph' or 'pphh' for singly or doubly
            excited configurations.
        indices : str
            The indices of the constructed t-amplitude.
        """
        # <Phi_k|0|n> + <Phi_k|1|n-1> - sum_{m=0}^n E^{(m)} t_k^{(n-m)} = 0

        # NOTE: Currently the implementation is general and should work for
        #       arbitrary 0th order Hamiltonians.
        #       Performance can be improved if the block structure
        #       of the RE hamiltonian is taken into account before evaluting
        #       wicks theorem! (Currently its done afterwards)

        # validate the input
        validate_input(order=order, space=space, indices=indices)
        n_ov = n_ov_from_space(space)
        if n_ov['n_occ'] != n_ov['n_virt']:
            raise Inputerror(f"Invalid space for a RE t-amplitude: {space}.")
        if n_ov['n_occ'] > 2 * order:  # space not present at the order
            return 0

        # get the target indices and validate
        idx = self.indices.get_indices(indices)
        if (n_ov['n_occ'] and 'occ' not in idx) or \
                (n_ov['n_virt'] and 'virt' not in idx) or\
                n_ov['n_occ'] != len(idx['occ']) or \
                n_ov['n_virt'] != len(idx['virt']):
            raise Inputerror(f"Indices {indices} are not valid for space "
                             f"{space}.")

        # - build <Phi_k|
        bra = self.h.excitation_operator(creation=idx["occ"],
                                         annihilation=idx["virt"],
                                         reverse_annihilation=True)

        # - compute (<Phi_k|0|n> + <Phi_k|1|n-1>)
        h0, rule = self.h.h0
        term = bra * h0 * self.psi(order, 'ket')
        res = wicks(term, rules=rule, simplify_kronecker_deltas=True)

        h1, rule = self.h.h1
        term = bra * h1 * self.psi(order - 1, 'ket')
        res += wicks(term, rules=rule, simplify_kronecker_deltas=True)

        # - subtract sum_{m=0}^n E^{(m)} t_k^{(n-m)}
        for e_order, t_order in gen_term_orders(order, 2, 0):
            # check if a t amplitude of order t_order exists
            # special treatment of first order singles
            if n_ov['n_occ'] > 2 * t_order or \
                    (n_ov['n_occ'] == 1 and t_order == 1 and not self.singles):
                continue
            name = f"{tensor_names.gs_amplitude}{t_order}"
            contrib = (
                self.energy(e_order) *
                Amplitude(name, idx['virt'], idx['occ'])
            ).expand()
            if n_ov['n_occ'] == 2:  # doubles -> different sign!
                res += contrib
            else:
                res -= contrib
        return res

    def overlap(self, order: int):
        """
        Computes the n'th-order contribution to the ground state overlap
        matrix.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        """
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
            res += wicks(i1, simplify_kronecker_deltas=True)
        # simplify the result by permuting contracted indices
        res = simplify(Expr(res))
        logger.debug(f"gs S^({order}) = {res}")
        return res.sympy

    @cached_member
    def expectation_value(self, order: int, n_particles: int):
        """
        Constructs the n'th-order contribution to the expectation value for
        the given operator.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        n_particles : int
            The number of creation and annihilation operators in the operator
            string.
        """
        validate_input(order=order)
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
        op, rules = self.h.operator(n_create=n_particles,
                                    n_annihilate=n_particles)
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
                d += wicks(i1, simplify_kronecker_deltas=True, rules=rules)
            res += (norm * d).expand()
        return simplify(Expr(res)).sympy

    def norm_factor(self, order: int):
        """
        Constructs the n'th-order contribution of the factor
        that corrects the the norm of the ground state wavefunction:
        1 - sum_i S^(i) + (sum_i S^(i))^2 - ...
        which is the result of a taylor expansion of a^2
        S = a^2 sum_{i=0} S^{(i)} = 1 -> a^2 = [sum_{i=0} S^{(i)}]^{(-1)}.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
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
        logger.debug(f"norm_factor^({order}): {latex(norm_factor)}")
        return norm_factor

    def expand_norm_factor(self, order, min_order=2) -> list:
        """
        Constructs the taylor expansion of the n'th-order contribution to the
        normalization factor a
        f = (1 + x)^(-1),
        where x is defined as x = sum_i S^{(i)}.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        min_order : int, optional
            The lowest order non-vanishing contribution of the overlap matrix S
            excluding the zeroth order contribution which is assumed to have
            a value of 1.

        Returns
        -------
        list
            Iterable containing tuples of prefactors and perturbation
            theoretical orders, for instance with a min_order of 2 the
            5'th order contribution reads
            [(-1, [(5,)]), (1, [(2, 3), (3, 2)])].
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
