from adcgen import (
    Operators, GroundState, IntermediateStates, Properties, Expr, simplify,
    factor_intermediates, remove_tensor, tensor_names
)

# Initialize the class hierarchy to to generate the ADC properties
h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="pp")
prop = Properties(isr)

# build the ADC(2) expectation value for a general one-particle operator
# X_I <I|p^+ q|J> Y_J d^p_q
expec = prop.expectation_value(adc_order=2, n_particles=1)

# Add assumptions:
# - real orbital basis
# - a symmetric operator matrix d (with bra-ket-symmetry)
expec = Expr(expec, real=True, sym_tensors=[tensor_names.operator])
# for the state_diff_dm we have the same eigenvector in the bra (X) and ket (Y)
expec.rename_tensor(current=tensor_names.left_adc_amplitude,
                    new=tensor_names.right_adc_amplitude)

# simplify the result
expec.substitute_contracted()
expec = simplify(expec)

# factor mp densities in the expression. t-amplitudes are also necessary,
# because the expression is expressed by means of t-amplitudes currently
# -> the mp_densitiy intermediates need to 'know' that so that their correct
#    form is used.
expec = factor_intermediates(expec,
                             types_or_names=["t_amplitude", "mp_density"],
                             max_order=2)

# remove the operator tensor d to obtain the density matrix
# (undo the contraction)
density = remove_tensor(expec, tensor_names.operator)
for block, block_expr in density.items():
    assert len(block) == 1
    block = block[0]
    print("\n", "#"*80, sep="")
    print(f"{len(block_expr)} terms in density matrix block {block}:\n"
          f"{block_expr}")
