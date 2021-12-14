from sympy import symbols, Rational, latex, Dummy
from sympy.physics.secondquant import AntiSymmetricTensor, F, Fd, NO, wicks, evaluate_deltas, substitute_dummies

# build H0 and H1
p, q, r, s = symbols('p,q,r,s', cls=Dummy)
f = AntiSymmetricTensor('f', (p,), (q,))
pq = NO(Fd(p) * F(q))
H0 = f * pq

# sum_i <pi||qi>
m = symbols('m', below_fermi=True, cls=Dummy)
v1 = AntiSymmetricTensor('v1', (p,m), (q,m))
# <pq||rs>
v2 = AntiSymmetricTensor('v2', (p,q), (r,s))
pqsr = NO(Fd(p) * Fd(q) * F(s) * F(r))

H1 = Rational(1, 4) * v2 * pqsr - v1 * pq

print("H0: ", latex(H0))
print("H1: ", latex(H1))

# generate different excitation classes
i,j = symbols('i,j', below_fermi=True, cls=Dummy)
a,b = symbols('a,b', above_fermi=True, cls=Dummy)
singles = NO(Fd(b) * F(j))
singles_d = NO(Fd(i) * F(a))
doubles = NO(Fd(a) * Fd(b) * F(j) * F(i))
doubles_d = NO(F(i) * F(j) * Fd(b) * Fd(a))

# build MP1 wavefunction
t = AntiSymmetricTensor('t', (a,b), (i,j))
t_conj = AntiSymmetricTensor('t_conj', (a,b), (i,j))
mp1 = -Rational(1,4) * t * doubles
mp1_d = -Rational(1,4) * t_conj * doubles_d

# build MP energies
# does evaluate to 0 for some reason
E0 = AntiSymmetricTensor('f', (p,), (q,)) * NO(Fd(p) * F(q))
E0 = wicks(E0)#, keep_only_fully_contracted=True)
E0 = evaluate_deltas(E0)
E0 = substitute_dummies(E0)
#print("E0 = ", latex(wicks(E0, keep_only_fully_contracted=True)))

# MP1 energy
# also evaluates to 0??
E1 = H1
E1 = wicks(E1, keep_only_fully_contracted=True)
E1 = evaluate_deltas(E1)
E1 = substitute_dummies(E1)
print("E1 = ", latex(E1))

# ADC(0)
M_ss_0 = singles_d * H0 * singles
M_ss_0 = wicks(M_ss_0, keep_only_fully_contracted=True)
M_ss_0 = evaluate_deltas(M_ss_0)
M_ss_0 = substitute_dummies(M_ss_0)

print("M_ss_0 = ", latex(M_ss_0))

# ADC(1)
i1 = singles_d * H0 * singles * mp1 - singles_d * singles * mp1
i1 = wicks(i1, keep_only_fully_contracted=True)
i1 = evaluate_deltas(i1)
i1 = substitute_dummies(i1)

print("i1 = ", latex(i1))

i2 = mp1_d * singles_d * H0 * singles - mp1_d * singles_d * singles
i2 = wicks(i2, keep_only_fully_contracted=True)
i2 = evaluate_deltas(i2)
i2 = substitute_dummies(i2)

print("i2 = ", latex(i2))

i3 = singles_d * H1 * singles
i3 = wicks(i3, keep_only_fully_contracted=True)
i3 = evaluate_deltas(i3)
i3 = substitute_dummies(i3)

print("i3 = ", latex(i3))

i4 = singles_d * singles
i4 = wicks(i4, keep_only_fully_contracted=True)
i4 = evaluate_deltas(i4)
i4 = substitute_dummies(i4)

print("i4 = ", latex(i4))

