from adcgen import (
    Operators, GroundState, ExprContainer, transform_to_spatial_orbitals,
    apply_resolution_of_identity
)

# We first declare the Hamiltonian operator
op = Operators()

# We exemplify this at the MP2 and MP3 energies
# For this, we first define the ground state
gs = GroundState(op)

# Next, we can calculate the MP2 and MP3 energy
energy_mp2 = ExprContainer(gs.energy(2))
energy_mp3 = ExprContainer(gs.energy(3))

# RI is only valid for real orbitals, wherefore we have to make these
# expressions real
energy_mp2.make_real()
energy_mp3.make_real()

# These can now be spin-integrated
energy_mp2 = transform_to_spatial_orbitals(energy_mp2, '', '',
                                           restricted=False)
energy_mp3 = transform_to_spatial_orbitals(energy_mp3, '', '',
                                           restricted=False)

# Lastly, we can apply the resolution of identity approximation
# We will decompose the MP2 energy symmetrically:
energy_mp2 = apply_resolution_of_identity(energy_mp2, factorisation='sym')
# And the MP3 energy asymetrically:
energy_mp3 = apply_resolution_of_identity(energy_mp3, factorisation='asym')

# We can now print the result
print("RI-MP2 Energy:\n")
print(energy_mp2.to_latex_str(spin_as_overbar=True))
print()

print("RI-MP3 Energy:\n")
print(energy_mp3.to_latex_str(spin_as_overbar=True))
print()
