import numpy as np
from sympy import KroneckerDelta, symbols, latex, Dummy, Rational, Add, Mul, S
from sympy.core.function import diff
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas, Dagger
from itertools import combinations

from indices import assign_index, make_pretty, pretty_indices, indices, get_first_missing_index
from groundstate import ground_state, Hamiltonian
from isr import gen_order_S, intermediate_states
from properties import properties
from secular_matrix import secular_matrix
from transformexpr import make_real, change_tensor_name, filter_tensor, remove_tensor, simplify, sort_by_n_deltas, sort_by_type_deltas, sort_by_type_tensor, sort_tensor_sum_indices
from misc import cached_member, transform_to_tuple

# i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
# a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)
# p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian(canonical=False)
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp, variant="pp")
m = secular_matrix(isr)
op = properties(isr)

# TODO: check canonical hamiltonian result.
a = op.one_particle_operator(adc_order=2)
a = mp.indices.substitute_indices(a)
a = change_tensor_name(a, "Y", "X")
print(latex(a))
a = sort_by_type_tensor(a, "d")
for t, expr in a.items():
    print(f"{len(expr.args)} terms with d {t}:\n{latex(expr)}")
b = a[("ov",)] + a[("vo",)]
b = make_real(b, "d")
b = simplify(b, "d")
b = make_real(b, "d")
b = simplify(b, "d")
print()
print(latex(b))

# v = AntiSymmetricTensor("V", (i,j), (a,b))
# v1 = AntiSymmetricTensor("V", (j,i), (b,a))
# v2 = AntiSymmetricTensor("V", (a,b), (i,j))
# t = AntiSymmetricTensor("t", tuple(), tuple())
# f = AntiSymmetricTensor("f", (i,), (a,))
# f1 = AntiSymmetricTensor("f", (k,), (b,))
# f2 = AntiSymmetricTensor("f", (k,), (a,))
# d = AntiSymmetricTensor("d", tuple(), tuple())

# print(latex(make_real(
#     v*f*t + v2*f*t + v*f1*t + v2*f1*t + v*t + v2*t + f*t + f1*t + v1*t + f2*t + v1*f2*t + d + f*d, "d"
# )))

# print(latex(simplify(v*f + v1*f1)))
# print(KroneckerDelta(i,j).free_symbols)
# print(next(iter(F(i).free_symbols)))
# print(Fd(i).free_symbols)
# print(AntiSymmetricTensor("t", (i,j), (a,b)).upper)
# Rational(1,2).free_symbols
