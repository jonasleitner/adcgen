from sympy import symbols, Rational, latex, Dummy
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, wicks, evaluate_deltas,
    contraction, substitute_dummies
)
from math import factorial
from indices import pretty_indices, indices


class Hamiltonian:
    def __init__(self):
        self.__H0 = self.build_H0()
        self.__H1 = self.build_H1()

    def build_H0(self):
        p = symbols('p', cls=Dummy)
        q = symbols('q', cls=Dummy)
        f = AntiSymmetricTensor('f', (p,), (q,))
        pq = Fd(p) * F(q)
        H0 = f * pq
        print("H0 = ", latex(H0))
        return H0

    @property
    def get_H0(self):
        return self.__H0

    def build_H1(self):
        p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        m = symbols('m', below_fermi=True, cls=Dummy)
        v1 = AntiSymmetricTensor('V', (p, m), (q, m))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        H1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        print("H1 = ", latex(H1))
        return H1.expand()

    @property
    def get_H1(self):
        return self.__H1


class ground_state:
    def __init__(self, hamiltonian):
        self.indices = indices()
        self.h = hamiltonian
        self.energy = {}
        self.wfn = {}

    def get_energy(self, order):
        if not self.energy.get(order, False):
            try:
                callback = getattr(self, "build_E" + str(order))
                callback()
            except AttributeError:
                print(f"Ground state energy of order {order} not implemented.")
                exit()
        return self.energy[order]

    def get_psi(self, order, braket):
        """Retuns the ground state wavefunction of the requested order.
           The type (Bra or Ket) of the requested wavefunction may be
           provided as ket=True and/or bra=True.
           Returns a dict
           By default only the Ket is build and returned."""

        if braket not in ["ket", "bra"]:
            print("Only possible to build 'bra' or 'ket' gs wavefunction",
                  f"{braket} is not valid")

        if not self.wfn.get(order, False):
            self.wfn[order] = {}

        if braket not in self.wfn[order]:
            if order in [0, 1]:
                callback = getattr(self, "_build_psi" + str(order))
                callback(braket)
            else:
                self.__build_psi(order, braket)
        return self.wfn[order][braket]

    def build_E0(self):
        p, q = symbols('p,q', cls=Dummy)
        # instead use function... does it make any difference?
        f = AntiSymmetricTensor('f', (p,), (q,))
        pq = contraction(Fd(p), F(q))
        E0 = f * pq
        E0 = evaluate_deltas(E0)
        self.energy[0] = E0
        print("E0 = ", latex(E0))

    def build_E1(self):
        # for some reason wicks does not work here... produces 0
        # Therefore, it is necessary to compute the contractions
        # manually. However, the manual contraction only returns
        # "a" or "i" as indices when general symbols (p, q...)
        # are used. Therefore, occupied indices needed to be used.

        # p, q, r, s = symbols('p,q,r,s', cls=Dummy)
        p, q, r, s = symbols('i,j,k,l', below_fermi=True, cls=Dummy)
        pq = contraction(Fd(p), F(q))
        v1 = AntiSymmetricTensor('V', (p, r), (q, r))
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = + contraction(Fd(p), F(r)) * contraction(Fd(q), F(s)) \
            - contraction(Fd(p), F(s)) * contraction(Fd(q), F(r))
        E1 = Rational(1, 4) * v2 * pqsr - v1 * pq
        E1 = evaluate_deltas(E1.expand())
        E1 = substitute_dummies(E1)
        print("E1 = ", latex(E1))
        self.energy[1] = E1

    def build_E2(self):
        psi1 = self.get_psi(1, ket=True)
        psi0 = self.get_psi(0, bra=True)
        H1 = self.h.get_H1
        E2 = psi0["bra"] * H1 * psi1["ket"]
        E2 = wicks(E2, keep_only_fully_contracted=True)
        E2 = evaluate_deltas(E2)
        E2 = substitute_dummies(
            E2, new_indices=True, pretty_indices=pretty_indices
        )
        print("E2 = ", latex(E2))
        self.energy[2] = E2

    def __build_psi(self, order, *args):
        # generalize the gs wavefunction generation
        # currently only used for second oder +
        if "bra" in args:
            psi = 0
            for excitation in range(1, 2 * order + 1):
                idx = self.indices.get_gs_indices(
                    "bra", n_occ=excitation, n_virt=excitation
                )
                t = AntiSymmetricTensor(
                    f"t{order}_cc", tuple(idx["virt"]), tuple(idx["occ"])
                )
                operators = 1
                for symbol in idx["occ"]:
                    operators *= Fd(symbol)
                for symbol in reversed(idx["virt"]):
                    operators *= F(symbol)
                prefactor = Rational(1, factorial(excitation) ** 2)
                psi += prefactor * t * NO(operators)
            self.wfn[order]["bra"] = psi
            print(f"<Psi^({order})| = ", latex(psi))
        if "ket" in args:
            psi = 0
            for excitation in range(1, 2 * order + 1):
                idx = self.indices.get_gs_indices(
                    "ket", n_occ=excitation, n_virt=excitation
                )
                t = AntiSymmetricTensor(
                    f"t{order}", tuple(idx["virt"]), tuple(idx["occ"])
                )
                operators = 1
                for symbol in idx["virt"]:
                    operators *= Fd(symbol)
                for symbol in reversed(idx["occ"]):
                    operators *= F(symbol)
                prefactor = Rational(1, factorial(excitation) ** 2)
                psi += prefactor * t * NO(operators)
            self.wfn[order]["ket"] = psi
            print(f"|Psi^({order})> = {latex(psi)}\n\n")

    def _build_psi0(self, *args):
        for braket in args:
            self.wfn[0][braket] = 1
            if braket == "ket":
                print(f"|Psi^(0)> = {self.wfn[0][braket]}\n\n")
            if braket == "bra":
                print(f"<Psi^(0)| = {self.wfn[0][braket]}\n\n")

    def _build_psi1(self, *args):
        if "bra" in args:
            idx = self.indices.get_gs_indices("bra", n_occ=2, n_virt=2)
            t = AntiSymmetricTensor(
                "t1_cc", tuple(idx["virt"]), tuple(idx["occ"])
            )
            operators = 1
            for symbol in idx["occ"]:
                operators *= Fd(symbol)
            for symbol in reversed(idx["virt"]):
                operators *= F(symbol)
            prefactor = Rational(1, 4)
            psi = prefactor * t * NO(operators)
            self.wfn[1]["bra"] = psi
            print("<Psi^(1)| = ", latex(psi))
        if "ket" in args:
            idx = self.indices.get_gs_indices("ket", n_occ=2, n_virt=2)
            t = AntiSymmetricTensor(
                "t1", tuple(idx["virt"]), tuple(idx["occ"])
            )
            operators = 1
            for symbol in idx["virt"]:
                operators *= Fd(symbol)
            for symbol in reversed(idx["occ"]):
                operators *= F(symbol)
            prefactor = Rational(1, 4)
            psi = prefactor * t * NO(operators)
            self.wfn[1]["ket"] = psi
            print(f"|Psi^(1)>  = {latex(psi)}\n\n")


# h = Hamiltonian()
# h.get_H0
# h.get_H1
# mp = ground_state(h)
# a = mp.get_energy(order=0)
# a = mp.get_energy(order=1)
# a = mp.get_energy(order=2)
# a = mp.get_psi(order=2, ket=True, bra=True)
