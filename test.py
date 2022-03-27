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

i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)
# p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian(canonical=False)
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp, variant="pp")
m = secular_matrix(isr)
op = properties(isr)

# TODO: check canonical hamiltonian result.
bla = op.one_particle_operator(2)
bla = mp.indices.substitute_indices(bla)
bla = change_tensor_name(bla, "X", "Y")
# print(latex(bla))
bla = sort_by_type_tensor(bla, "d", symmetric=False)
for t, expr in bla.items():
    s = simplify(expr, True, "d")
    print(f"{len(expr.args)} terms with d {t}:\n{latex(expr)}\n\n",
          latex(s), "\n", len(s.args))

print()
s = simplify(bla[("ov",)] - bla[("vo",)], True, "d")
print(latex(s))
if s == S.Zero:
    print("IT WORKS!!1!!!!!111!!!!")

# v = AntiSymmetricTensor("V", (i,j), (a,b))
# v1 = AntiSymmetricTensor("V", (k,l), (a,b))
# v2 = AntiSymmetricTensor("V", (a,b), (i,j))
# t = AntiSymmetricTensor("t", tuple(), tuple())
# f = AntiSymmetricTensor("f", (i,), (a,))
# f1 = AntiSymmetricTensor("f", (j,), (b,))
# f2 = AntiSymmetricTensor("f", (a,), (i,))
# d = AntiSymmetricTensor("d", (i,), (a,))
# d1 = AntiSymmetricTensor("d", (j,), (b,))

# print(latex(make_real(
#     v*f*t + v2*f*t + v*f2*t + v2*f2*t + v1*f*t + v*t + v2*t + v1*t + f*t + f2*t + f1*t + v
# )))
# print(latex(make_real(
#     v*f*f1 + v*f + f + v + f*f1
# )))
