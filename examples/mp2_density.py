from adcgen import (
    Operators, GroundState, remove_tensor, Expr, simplify,
    transform_to_spatial_orbitals, reduce_expr
)

h = Operators()
mp = GroundState(h)

# 1 particle operator expectation value: 2nd order contribution
expec = mp.expectation_value(order=2, n_particles=1)
# in a real orbital basis and for a symmetric operator matrix as the density
# is symmetric.
expec = Expr(expec, real=True, sym_tensors=["d"]).substitute_contracted()
expec = simplify(expec)

target_idx = {"oo": "ij", "ov": "ia", "vv": "ab"}

# extract the density matrix from the expectation value by removing the
# operator matrix d from the expression.
dm = remove_tensor(expec, "d")
for dm_block, dm_expr in dm.items():
    assert len(dm_block) == 1
    dm_block = dm_block[0]

    print("\n"*3, f"Processing block {dm_block}...", sep="")

    # in principle we are done at this point.
    # However, it is also possible to express the expression in spatial
    # orbitals (until now spin orbitals have been used).
    # - Since implementations in spatial orbitals often only use ERI and no
    #   t-amplitudes, we expand all the intermediates in the expression and
    #   simplify the result as much as posible.
    dm_expr = reduce_expr(dm_expr)

    # - switch to a symbolic representation of the orbital energy denominators
    dm_expr.use_symbolic_denominators()
    # - integrate the spin, expand the antisymmetric ERI (physicist notation)
    #   into coulomb integrals (chemist notation) assuming a
    #   restricted reference state.
    dm_expr = transform_to_spatial_orbitals(
        dm_expr, target_idx=target_idx[dm_block], target_spin="aa",
        restricted=True, expand_eri=True
    )
    # - simplify the resulting expression again and switch back to explicit
    #   orbital energy denominators
    dm_expr = simplify(dm_expr)
    dm_expr.use_explicit_denominators()

    print("\n", "*"*80, sep="")
    print(f"{len(dm_expr)} terms in dm block {dm_block}:\n{dm_expr}")
    print()
    # or a bit more more readable...
    print(dm_expr.to_latex_str(terms_per_line=1, spin_as_overbar=True))
