from math import factorial
import numpy as np
from sympy import symbols, latex, Dummy
from sympy.core.function import diff
from indices import pretty_indices
from sympy.physics.secondquant import AntiSymmetricTensor, contraction, substitute_dummies, wicks, NO, Fd, F, evaluate_deltas

p, q, r, s = symbols('p,q,r,s', cls=Dummy)

print(latex(wicks(NO(Fd(p) * Fd(q) * F(s) * F(r)), keep_only_fully_contracted=True)))

i, j = symbols('i,j', below_fermi=True, cls=Dummy)
a, b = symbols('a,b', above_fermi=True, cls=Dummy)

S = NO(Fd(i) * F(a)) * NO(Fd(b) * F(j))
psi = Fd(a) * F(i)
isr = S * psi

wick = wicks(isr, keep_only_fully_contracted=True)
wick = evaluate_deltas(wick)
print("gesamt ausdruck auswerten: ", latex(wick))

wick = wicks(S, keep_only_fully_contracted=True)
ges = wick * psi
ges = evaluate_deltas(ges)
ges = substitute_dummies(ges)
print("einzeln wicks fuer S: ", latex(ges))



wick = wicks(S, keep_only_fully_contracted=True)
wick = evaluate_deltas(wick)
wick = substitute_dummies(wick)  # , new_indices=True, pretty_indices=indices)
print(latex(wick))
ges = wick * psi
ges = substitute_dummies(ges, new_indices=True, pretty_indices=pretty_indices)
ges = evaluate_deltas(ges)
print("evaluate_deltas einzeln: ", latex(ges))

test = NO(Fd(j) * F(a))
test = wicks(test, keep_only_fully_contracted=True)
print(latex(test))
test = substitute_dummies(test)
print(latex(test))

delta = contraction(Fd(p), F(q))
delta = evaluate_deltas(delta * AntiSymmetricTensor('f', (p,), (q,)))
print(latex(delta))