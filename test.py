import numpy as np
from sympy import symbols, latex, Dummy, Rational, Add, Mul
from sympy.core.function import diff
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas

from indices import pretty_indices, indices
from groundstate import ground_state, Hamiltonian
from isr import intermediate_states
from secular_matrix import secular_matrix

# i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
# a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)
# p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian(canonical=False)
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp)
m = secular_matrix(isr)

a = isr.precursor(1, "pphh", "ket", "ijab")
a_pretty = substitute_dummies(
    a, new_indices=True, pretty_indices=pretty_indices
)
print(latex(a_pretty))
print(mp.indices.substitute_with_generic_indices(a))
print(mp.indices.substitute_indices(a))
