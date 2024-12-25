from adcgen import (
    Operators, GroundState, IntermediateStates, SecularMatrix, Expr, simplify,
    sort, reduce_expr, factor_intermediates, generate_code
)

# Initialize the class hierarchy to to generate the secular matrix
h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="pp")
m = SecularMatrix(isr)

# Build the raw expression of the 2nd order contribution to the 1p-1h/1p-1h
# PP-ADC secular matrix block in a complex, non-canonical orbital basis
matrix = m.isr_matrix_block(order=2, block="ph,ph", indices="ia,jb")

# Add assumptions to the raw result: real orbital basis
#  -> fock matrix f and ERI V have bra-ket-symmetry
#  -> MP t-amplitudes are real t1cc -> t1
matrix = Expr(matrix, real=True)

# simplify the result:
# - use the lowest available contracted indices (way more readable!)
# - permute contracted indices to reduce the number of terms as much as
#   possible.
matrix.substitute_contracted()
matrix = simplify(matrix)

# print the result and/or write it to a file (can be read in from string again
# with the import_from_sympy_latex function)
for delta_types, expr in sort.by_delta_types(matrix).items():
    print("\n", "#"*80, sep="")
    print(f"{len(expr)} terms with delta {delta_types}:\n{expr}")

# the expression can be further simplified by assuming that we are working
# in a canonical orbital basis:
# - the fock matrix is diagonal
# - insert the definition of the MP t-amplitudes and reduce the number of
#   terms
matrix.diagonalize_fock()
matrix = reduce_expr(matrix)

# the expression now consists only of ERI and orbital energies
# -> factor intermediates in the expression again (of maximum pt order 1)
matrix = factor_intermediates(matrix, max_order=1)

# This generates code using the numpy.einsum syntax
# The naming convention is currently chosen to work for the adc-connect module.
# The scaling is determined by assuming sizes for each space, e.g.,
# 5 core, 20 occupied and 200 virtual orbitals. This can be modified
# through 'adcgen/generate_code/config.json' or by providing a dict with
# the desired sizes to the 'generate_code' function.
code = generate_code(matrix, target_indices="ia,jb", backend="einsum",
                     bra_ket_sym=1, max_tensor_dim=4,
                     optimize_contractions=True)
print(f"\n\nGenerated code:\n{code}")
