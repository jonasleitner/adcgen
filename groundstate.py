from sympy import symbols, Rational, latex, Dummy
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, wicks, substitute_dummies
)
from math import factorial
from indices import pretty_indices, indices


class Hamiltonian:
    def __init__(self, canonical=True):
        self.__H0 = 0
        self.__H1 = 0
        self.canonical = canonical

    def __build_H0(self, canonical):
        p = symbols('p', cls=Dummy)
        if canonical:
            f = AntiSymmetricTensor('f', (p,), (p,))
            pp = Fd(p) * F(p)
            self.__H0 = f * pp
        else:
            q = symbols('q', cls=Dummy)
            f = AntiSymmetricTensor('f', (p,), (q,))
            pq = Fd(p) * F(q)
            self.__H0 = f * pq
        print("H0 = ", latex(self.__H0))

    @property
    def get_H0(self):
        if not self.__H0:
            self.__build_H0(self.canonical)
        return self.__H0

    @property
    def get_H1(self):
        if not self.__H1:
            self.__build_H1()
        return self.__H1

    def __build_H1(self):
        p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        o42 = symbols('o42', below_fermi=True, cls=Dummy)
        v1 = AntiSymmetricTensor('V', (p, o42), (q, o42))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        self.__H1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(self.__H1))


class ground_state:
    def __init__(self, hamiltonian, first_order_singles=False):
        self.indices = indices()
        self.h = hamiltonian
        self.singles = first_order_singles
        self.energy = {}
        self.wfn = {}

    def get_energy(self, order):
        """Returns the ground state energy of specified order."""

        if not isinstance(order, int):
            print("Order for obtaining ground state wavefunction/energy",
                  f"needs to of type int. {type(order)} is not valid.")
            exit()

        if order not in self.energy:
            self.__build_E(order)
        return self.energy[order]

    def __build_E(self, order):
        def H(o): return self.h.get_H0 if o == 0 else self.h.get_H1
        h = H(order)
        bra = self.get_psi(0, "bra")
        def Ket(o): return self.get_psi(0, "ket") if o == 0 else \
            self.get_psi(order-1, "ket")
        ket = Ket(order)
        e = bra * h * ket
        e = wicks(e, keep_only_fully_contracted=True,
                  simplify_kronecker_deltas=True)
        # this is problematic, since the substitute_dummies function
        # messes around with the indices and assign them not very nice.
        # But i don't know how to handle the multiple i symbols else.
        # maybe it is necessary to manually build the energy for every order?
        e = substitute_dummies(e, new_indices=True,
                               pretty_indices=pretty_indices)
        e = self.indices.substitute_with_generic_indices(e)
        self.energy[order] = e
        print(f"E^({order}) = {latex(e)}")

    def get_pretty_energy(self, order):
        return substitute_dummies(self.get_energy(order), new_indices=True,
                                  pretty_indices=pretty_indices)

    def get_psi(self, order, braket):
        """Retuns the ground state wavefunction of the requested order.
           The type (bra or ket) of the requested wavefunction needs to
           be provided as str.
           """

        if not isinstance(order, int):
            print("Order for obtaining ground state wavefunction/energy",
                  f"needs to of type int. {type(order)} is not valid.")
            exit()
        if braket not in ["ket", "bra"]:
            print("Only possible to build 'bra' or 'ket' gs wavefunction",
                  f"{braket} is not valid")
            exit()

        if not self.wfn.get(order, False):
            self.wfn[order] = {}

        if braket not in self.wfn[order]:
            if order == 0:
                callback = getattr(self, "_build_psi0")
                callback(braket)
            else:
                self.__build_psi(order, braket)
        return self.wfn[order][braket]

    def __build_psi(self, order, braket):
        # generalized gs wavefunction generation
        tensor_string = {
            "bra": f"t{order}_cc",
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
        self.wfn[order][braket] = psi
        print(f"Build gs^({order}) {braket} = ", latex(psi))

    def _build_psi0(self, *args):
        for braket in args:
            self.wfn[0][braket] = 1
            if braket == "ket":
                print(f"|Psi^(0)> = {self.wfn[0][braket]}\n\n")
            if braket == "bra":
                print(f"<Psi^(0)| = {self.wfn[0][braket]}\n\n")
