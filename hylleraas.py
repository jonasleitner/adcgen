from groundstate import Hamiltonian, ground_state
from indices import pretty_indices
from sympy import latex, Dummy
from sympy.physics.secondquant import wicks, substitute_dummies


def evaluate(expr):
    ret = wicks(expr, keep_only_fully_contracted=True,
                simplify_kronecker_deltas=True)
    ret = substitute_dummies(ret, new_indices=True,
                             pretty_indices=pretty_indices)
    ret = mp.indices.substitute_indices(ret)
    return ret


h = Hamiltonian(canonical=False)
mp = ground_state(h, first_order_singles=True)
e0 = mp.get_energy(0)
e1 = mp.get_energy(1)
e2 = mp.get_energy(2)
ketpsi0 = mp.get_psi(0, "ket")
brapsi0 = mp.get_psi(0, "bra")
brapsi1 = mp.get_psi(1, "bra")
ketpsi1 = mp.get_psi(1, "ket")
brapsi2 = mp.get_psi(2, "bra")
ketpsi2 = mp.get_psi(2, "ket")

term1 = brapsi1 * (h.get_H0 - e0) * ketpsi1
term1 = evaluate(term1)
print(latex(term1), "\n correct!!\n\n")

term2 = brapsi1 * (h.get_H1 - e1) * ketpsi0
term2 = evaluate(term2)
print(latex(term2), "\nCorrect!!!\n\n")

term3 = brapsi0 * (h.get_H1 - e1) * ketpsi1
term3 = evaluate(term3)
print(latex(term3), "\nCorrect!!!\n\n")

res = term1 + term2 + term3
print("second order Hylleraas energy: ", latex(res), "\n\n")


print("THIRD ORDER")

term1 = brapsi2 * (h.get_H1 - e1) * ketpsi0
term1 = evaluate(term1)
print(latex(term1), "\nprobably correct.\n\n")

term2 = brapsi0 * (h.get_H1 - e1) * ketpsi2
term2 = evaluate(term2)
print(latex(term2), "\nprobably correct.\n\n")

term3 = brapsi1 * (h.get_H1 - e1) * ketpsi1
term3 = evaluate(term3)
print(latex(term3), "\n\n")

term4 = brapsi2 * (h.get_H0 - e0) * ketpsi1
term4 = evaluate(term4)
print(latex(term4), "\n\n")

term5 = brapsi1 * (h.get_H0 - e0) * ketpsi2
term5 = evaluate(term5)
print(latex(term5), "\n\n")

term6 = e2 * (brapsi1 * ketpsi0 + brapsi0 * ketpsi1)
term6 = evaluate(term6)
print(latex(term6), "\n\n")

res = term1 + term2 + term3 + term4 + term5 - term6
print("Third order Hylleraas energy: ", latex(res), "\n\n")


print("FOURTH ORDER (PARTIAL)")

term3 = brapsi2 * (h.get_H1 - e1) * ketpsi1
term3 = evaluate(term3)
print(latex(term3), "\n\n")

term7 = brapsi2 * (h.get_H0 - e0) * ketpsi2
term7 = evaluate(term7)
print(latex(term7))
