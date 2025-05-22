from adcgen import Operators, GroundState, ExprContainer, simplify, sort
from sympy import Symbol


order = 1
space = "pphh"
indices = "ijab"


h_mp = Operators(variant="mp")
h_re = Operators(variant="re")

mp = GroundState(h_mp)
re = GroundState(h_re)

# The remp amplitude residual is defined as
# A * MP_residual + (1 - A) * RE_residual.
# -> construct the MP and RE amplitude residuals
mp_residual = mp.amplitude_residual(order=order, space=space, indices=indices)
re_residual = re.amplitude_residual(order=order, space=space, indices=indices)
# the MP mixing ratio (simply a number) can be represented as Symbol
remp_A = Symbol("A")

# build and simplify the remp residual (assuming a real orbital basis)
remp_residual = remp_A * mp_residual + (1 - remp_A) * re_residual
remp_residual = ExprContainer(remp_residual, real=True)
remp_residual.substitute_contracted()
remp_residual = simplify(remp_residual)

# reduce the number of terms by exploiting permutational symmetry:
# X - P_ab X -> (1 - P_ab) X
remp_residual = sort.exploit_perm_sym(remp_residual, target_indices=indices)
print("\n", "#"*80, sep="")
for symmetry, sub_expr in remp_residual.items():
    # symmetry contains tuples of permutation operators and factors (+-1)
    # e.g., (((P_ij,), -1),) corresponds to (1 - P_ij), while
    # (((P_ij,), -1), ((P_ab,), -1), ((P_ij, P_ab), 1)) corresponds to
    # (1 - P_ij - P_ab + P_ij P_ab).
    print(f"\nThe permutations {symmetry} need to be applied to:\n{sub_expr}")
