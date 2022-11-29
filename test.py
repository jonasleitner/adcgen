import numpy as np
from sympy import KroneckerDelta, nsimplify, symbols, latex, Dummy, Rational, Add, Mul, S, Pow, sympify, factor, Function, Expr
from sympy.core.function import diff
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas, Dagger
from itertools import combinations
import sympy
import sys

from indices import indices, get_symbols
from groundstate import ground_state, Hamiltonian
from intermediates import t2_2, t2_1, intermediates, registered_intermediate, eri, orb_energy, t1_2, p0_2_oo, p0_2_vv, \
    t2eri_5, t2eri_4, t2eri_3
from isr import intermediate_states
from properties import properties
from secular_matrix import secular_matrix
from misc import cached_member, transform_to_tuple
import expr_container as e
import simplify as sim
import time
from misc import cached_member, validate_input
import sort_expr as sort
from reduce_expr import reduce_expr
from eri_orbenergy import eri_orbenergy
from func import gen_term_orders
from factor_intermediates import factor_intermediates
from sympy_objects import NonSymmetricTensor
sys.setrecursionlimit(100)

i, j, k, l, m, n, o, i1, j2 = symbols('i,j,k,l,m,n,o,i1,j2', below_fermi=True, cls=Dummy)
a, b, c, d = symbols('a,b,c,d', above_fermi=True, cls=Dummy)
p, q, r, s = symbols('p,q,r,s', cls=Dummy)

h = Hamiltonian()
mp = ground_state(h, first_order_singles=False)
isr = intermediate_states(mp, variant="pp")
m = secular_matrix(isr)
op = properties(isr)

# idx = indices().get_generic_indices(n_g=16)
# i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = idx['general']
# test = (
#     eri((i, m, q, u)) * eri((j, n, r, v)) * eri((k, o, s, w)) *
#     eri((l, p, t, x)) *
#     AntiSymmetricTensor('e', (), (i, j, k, l)) *
#     AntiSymmetricTensor('e', (), (m, n, o, p)) *
#     AntiSymmetricTensor('e', (), (q, r, s, t)) *
#     AntiSymmetricTensor('e', (), (u, v, w, x))
# )
# test = e.expr(test, True, target_idx='')
# test = test.substitute_contracted()
# print(f"Oringal term: {test}\n\n")
# test = test.terms[0].symmetrize
# print(f"Symemtrized term ({len(test)} terms):\n{test}\n\n")
# test = sim.simplify(test, True, 'V')
# print(f"Simplified result:\n{test}")
# exit()

# i, j = get_symbols('ij')
# td2 = t2_2()
# test = Rational(1, 4) * KroneckerDelta(i, j) * eri('klbc') * \
#     td2.expand_itmd(indices='klac').sympy
# test = e.expr(test, True, target_idx='ijab')
# test = reduce_expr(test)
# test = t2_1().factor_itmd(test)
# print(test)
# test = td2.factor_itmd(test, ['t2_1'])
# print(test)
# exit()
start = time.perf_counter()
# bla = mp.expectation_value(3, opstring='ca')
bla = m.isr_matrix_block(3, block='ph,ph', indices='ia,jb')
# bla = m.mvp_block_order(3, mvp_space='ph', block='ph,ph', indices='ia')
# bla = op.expectation_value(2, opstring='ca', subtract_gs=True)
# bla = op.op_block(2, block='ph,phph', opstring='ca', subtract_gs=False)
# bla = op.trans_moment(2, opstring='ca', subtract_gs=False)
# bla = mp.amplitude(3, space="phphphph", indices="iajbkcld")
# bla = e.expr(bla).rename_tensor('X', 'Y')

print(f"RAW RESULT: {len(bla.args)} terms\n")
bla = mp.indices.substitute(bla)
print(f"\nSUBSTITUTED RESULT ({len(bla)} terms)")
bla = sim.simplify(bla, True)
print(f"\nSIMPLIFIED RESULT ({len(bla)} terms):")

sorted = sort.by_delta_types(bla)
# sorted = sort.by_delta_indices(bla)
# sorted = sort.by_tensor_target_idx(bla, 'Y')
# sorted = sort.by_tensor_block(bla, 'd', False)
# print("SORTED RESULT:\n\n")
for t, expr in sorted.items():
    if t != ('oo', ):
        continue
    print(f"{len(expr)} terms in block {t}:\n{expr}\n\n")
    expr = reduce_expr(expr)
    factor_timer = time.perf_counter()
    expr = factor_intermediates(expr)
    print(f"Took {time.perf_counter() - factor_timer} to factor.")
    print(f"{len(expr)} terms in final expression:\n{expr}\n\n\n")
print(f"\nWALLTIME: {time.perf_counter()-start}")
exit()

sorted = sim.extract_dm(bla, False)
print("DM:\n\n")
for t, expr in sorted.items():
    print(f"{len(expr)} terms in block {t}:\n{expr}\n\n{expr.print_latex(3, False)}\n\n\n")
exit()

i, j, k, l = symbols('i,j,k,l', below_fermi=True, cls=Dummy)
a, b, c, d = symbols('a,b,c,d', above_fermi=True, cls=Dummy)


n1 = NO(F(i) * F(j) * Fd(a))
t1 = AntiSymmetricTensor('t1cc', (c, d), (i, l))
t2 = AntiSymmetricTensor('t1', (c, d), (k, l))
t3 = AntiSymmetricTensor('t1', (c, d), (i, k))
v1 = AntiSymmetricTensor('V', (b, c), (j, k))
v2 = AntiSymmetricTensor('V', (a, k), (a, c))
v3 = AntiSymmetricTensor('V', (l, k), (j, i))
v4 = AntiSymmetricTensor('V', (i, j), (a, b))
del1 = KroneckerDelta(i, j)
del2 = KroneckerDelta(a, b)
a1 = F(i)
c1 = Fd(j)
y1 = AntiSymmetricTensor('Y', (a,), (i,))
y2 = AntiSymmetricTensor('Ycc', (a, b), (i, j))
y3 = AntiSymmetricTensor('Y', (b, c), (i, j))
d1 = AntiSymmetricTensor('d', (b,), (c,))
d2 = AntiSymmetricTensor('d', (a,), (c,))
f1 = AntiSymmetricTensor('f', (j,), (k,))
f2 = AntiSymmetricTensor('f', (j,), (l,))
f3 = AntiSymmetricTensor('f', (a,), (i,))
f4 = AntiSymmetricTensor('f', (i,), (a,))
f5 = AntiSymmetricTensor('f', (b,), (c,))
e1 = AntiSymmetricTensor('e', tuple(), (b,))
e2 = AntiSymmetricTensor('e', tuple(), (c,))
e3 = AntiSymmetricTensor('e', tuple(), (j,))
e4 = AntiSymmetricTensor('e', tuple(), (k,))
e5 = AntiSymmetricTensor('e', tuple(), (i,))

bla = f1 * t1 * t2 - f2 * t3 * t2
bla = e.expr(bla, real=True)
print(bla)
bla = sim.simplify(bla, True)
print(bla)
