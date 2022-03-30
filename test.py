import numpy as np
from sympy import KroneckerDelta, symbols, latex, Dummy, Rational, Add, Mul, S, Pow
from sympy.core.function import diff
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas, Dagger
from itertools import combinations

from indices import assign_index, make_pretty, pretty_indices, indices, get_first_missing_index
from groundstate import ground_state, Hamiltonian
from isr import gen_order_S, intermediate_states
from properties import properties
from secular_matrix import secular_matrix
from transformexpr import make_real, change_tensor_name, filter_tensor, remove_tensor, simplify, sort_by_n_deltas, sort_by_type_deltas, sort_by_type_tensor, sort_tensor_contracted_indices
from misc import cached_member, transform_to_tuple
from transformexpr import *
i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)
# p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian()
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp, variant="pp")
m = secular_matrix(isr)
op = properties(isr)

# bla = op.one_particle_operator(2)
# bla = mp.indices.substitute_indices(bla)
# bla = change_tensor_name(bla, "X", "Ycc")
# # print(latex(bla))
# bla = sort_by_type_tensor(bla, "d", symmetric=False)
# for t, expr in bla.items():
#     s = simplify(expr, True, "d")
#     print(f"{len(expr.args)} terms with d {t}:\n{latex(expr)}\n\n",
#           latex(s), "\n", len(s.args))

# print()
# s = simplify(bla[("ov",)] - bla[("vo",)], True, "d")
# print(latex(s))
# if s == S.Zero:
#     print("IT WORKS!!1!!!!!111!!!!")

bla = m.precursor_matrix_block(2, "ph,ph", "ia,jb")
bla = mp.indices.substitute_indices(bla)
print(latex(bla))
print("original number of terms:", len(bla.args))
# orig = sort_by_type_deltas(bla)
# for type, expr in orig.items():
#     print(f"{len(expr.args)} terms with delta {type}:\n{latex(expr)}")
print()
bla = simplify(bla, True)
print(latex(bla))
bla = sort_by_type_deltas(bla)
for type, expr in bla.items():
    print(f"{len(expr.args)} terms with delta {type}:\n{latex(expr)}")

# f = AntiSymmetricTensor("f", (k,), (k,))
# t = AntiSymmetricTensor("t1", (c,d), (l,m))
# tc = AntiSymmetricTensor("t1cc", (c,d), (l,m))

# f1 = AntiSymmetricTensor("f", (m,), (m,))
# t1 = AntiSymmetricTensor("t1", (c,d), (k,l))
# tc1 = AntiSymmetricTensor("t1cc", (c,d), (k,l))

# f2 = AntiSymmetricTensor("f", (c,), (e,))
# t2 = AntiSymmetricTensor("t1", (d,e), (k,l))
# tc2 = AntiSymmetricTensor("t1cc", (c,d), (k,l))

# f3 = AntiSymmetricTensor("f", (d,), (e,))
# t3 = AntiSymmetricTensor("t1", (c,e), (k,l))
# tc3 = AntiSymmetricTensor("t1cc", (c,d), (k,l))

# f4 = AntiSymmetricTensor("f", (m,), (k,))
# t4 = AntiSymmetricTensor("t1", (c,d), (l,m))
# tc4 = AntiSymmetricTensor("t1cc", (c,d), (k,l))

# f5 = AntiSymmetricTensor("f", (m,), (l,))
# t5 = AntiSymmetricTensor("t1", (c,d), (k,m))
# tc5 = AntiSymmetricTensor("t1cc", (c,d), (k,l))

# expr = 0.25*f*t*tc - 0.25*f1*t1*tc1 # - 0.25*f2*t2*tc2 + 0.25*f3*t3*tc3 + 0.25*f4*t4*tc4 - 0.25*f5*t5*tc5
# print(latex(expr))
# print(latex(simplify(expr, True)))
