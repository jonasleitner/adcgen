from adcgen import (
    Operators, GroundState, simplify, ExprContainer, EriOrbenergy
)
from sympy import S

order = 2
space = "pphh"
indices = "ijab"


h = Operators()
mp = GroundState(h)

# - build the raw result
ampl = mp.amplitude(order=order, space=space, indices=indices)
# - Add assumptions and simplify the result.
#   For the mp amplitudes we have to provide the target indices, because they
#   can not be determined with the Einstein sum convention.
ampl = ExprContainer(ampl, real=True, target_idx=indices)
ampl.expand().substitute_contracted()
ampl.use_symbolic_denominators()
ampl = simplify(ampl)
ampl.use_explicit_denominators()

# - remove the common orbital energy denominator from the expression,
#   which sympy should sort as (a + b + c + ... - i - j - ...)
common_denom = None
ampl_without_denom = 0
for term in ampl.terms:
    term = EriOrbenergy(term)
    # term.canonicalize_sign()  # add occ and subtract virt orbital energies
    if common_denom is None:
        common_denom = term.denom
    else:
        assert (common_denom - term.denom).inner is S.Zero
    ampl_without_denom += term.pref * term.num * term.eri

print("\n", "#"*80, sep="")
print(f"Common denominator:\n{common_denom}\n")
print(f"Amplitude definition:\n{ampl_without_denom}")
