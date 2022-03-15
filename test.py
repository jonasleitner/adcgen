import numpy as np
from sympy import KroneckerDelta, symbols, latex, Dummy, Rational, Add, Mul, S
from sympy.core.function import diff
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas, Dagger

from indices import assign_index, make_pretty, pretty_indices, indices, get_first_missing_index
from groundstate import ground_state, Hamiltonian
from isr import gen_order_S, intermediate_states
from properties import properties
from secular_matrix import secular_matrix
from transformexpr import make_real, change_tensor_name, filter_tensor, remove_tensor, sort_by_n_deltas, sort_by_type_deltas, sort_by_type_tensor, sort_tensor_sum_indices
from misc import cached_member, transform_to_tuple

# i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
# a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)
# p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian(canonical=False)
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp, variant="pp")
m = secular_matrix(isr)
op = properties(isr)

# TODO: CHECK ONE PARTICLE DIFF DM result
a = op.one_particle_operator(adc_order=2)
a = mp.indices.substitute_indices(a)
a = change_tensor_name(a, "Y", "X")
print(latex(a))
a = sort_by_type_tensor(a, "d")
for t, expr in a.items():
    print(f"{len(expr.args)} terms with d {t}:\n{latex(expr)}")
