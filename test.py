from math import factorial
import numpy as np
from sympy import symbols, latex, Dummy, Rational
from sympy.core.function import diff
from indices import pretty_indices
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas

i, j, k, l, m, n, o = symbols('i,j,k,l,m,n,o', below_fermi=True, cls=Dummy)
a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g', above_fermi=True, cls=Dummy)

bra1 = Rational(1, 4) * AntiSymmetricTensor('t1_cc', (c, d), (k, l)) * \
    NO(Fd(k) * Fd(l) * F(d) * F(c))
ket1 = Rational(1, 4) * AntiSymmetricTensor('t1', (e, f), (m, n)) * \
    NO(Fd(e) * Fd(f) * F(n) * F(m))

psi0ket = NO(Fd(b) * F(j))
psi0bra = NO(Fd(i) * F(a))
psi1ket = NO(Fd(b) * F(j)) * ket1
psi1bra = bra1 * NO(Fd(i) * F(a))

for arg in psi1ket.free_symbols:
    print(arg)

S0 = psi0bra * psi0ket
S1 = psi1bra * psi0ket + psi0bra * psi1ket

s0 = wicks(S0, keep_only_fully_contracted=True,
           simplify_kronecker_deltas=True,
           simplify_dummies=True)
print("S0: ", latex(s0))

s1 = wicks(S1, keep_only_fully_contracted=True,
           simplify_kronecker_deltas=True,
           simplify_dummies=True)
print("s1: ", latex(s1))

isr0 = psi0ket * s0
isr0 = evaluate_deltas(isr0)
print("irs0: ", latex(isr0))

isr1 = psi0ket * s1 + s0 * psi1ket
print("before deltas: ", latex(isr1))
isr1 = evaluate_deltas(isr1)
print("isr1: ", latex(isr1))
