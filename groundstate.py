from sympy import symbols, Rational, latex, Dummy
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, wicks, substitute_dummies
)
from math import factorial
from indices import pretty_indices, indices
from misc import cached_member, cached_property


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
        return pq

    @cached_property
    def two_paricle(self):
        p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        return Rational(1, 4) * pqsr


class ground_state:
    def __init__(self, hamiltonian, first_order_singles=False):
        self.indices = indices()
        self.h = hamiltonian
        self.singles = first_order_singles
        # self.energy = {}
        # self.wfn = {}

    @cached_member
    def energy(self, order):
        """Returns the ground state energy of specified order."""

        if not isinstance(order, int):
            print("Order for obtaining ground state wavefunction/energy"
                  f"needs to of type int. {type(order)} is not valid.")
            exit()

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
            print("Order for obtaining ground state wavefunction/energy"
                  f"needs to of type int. {type(order)} is not valid.")
            exit()
        if braket not in ["ket", "bra"]:
            print("Only possible to build 'bra' or 'ket' gs wavefunction"
                  f"{braket} is not valid")
            exit()

        # catch 0th order wavefunction
        if order == 0:
            print(f"Build gs^({order} {braket} = 1")
            return 1

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
            # prefactor correct?
            prefactor = Rational(1, factorial(excitation) ** 2)
            psi += - prefactor * t * NO(operators)
        print(f"Build gs^({order}) {braket} = ", latex(psi))
        return psi
