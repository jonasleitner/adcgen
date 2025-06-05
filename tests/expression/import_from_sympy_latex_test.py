from adcgen import import_from_sympy_latex, get_symbols, tensor_names
from adcgen.sympy_objects import (
    Amplitude, AntiSymmetricTensor, KroneckerDelta, NonSymmetricTensor,
    SymmetricTensor
)

from sympy.physics.secondquant import F, Fd, NO
from sympy import Pow, Rational, S, latex, sqrt


class TestImportFromSympyLatex:
    def test_empty(self):
        assert import_from_sympy_latex("").inner is S.Zero

    def test_number(self):
        assert import_from_sympy_latex("2").inner == 2
        assert import_from_sympy_latex(r"\frac{1}{2}").inner == Rational(1, 2)
        assert import_from_sympy_latex(r"\sqrt{2}").inner == sqrt(2)

    def test_delta(self):
        # spin orbitals
        delta = KroneckerDelta(*get_symbols("pq"))
        assert import_from_sympy_latex(latex(delta)).inner - delta is S.Zero
        # spatial orbitals
        delta = KroneckerDelta(*get_symbols("pq", spins="aa"))
        assert import_from_sympy_latex(latex(delta)).inner - delta is S.Zero
        # mixed
        delta = KroneckerDelta(*get_symbols("p", spins="b"), *get_symbols("p"))
        assert import_from_sympy_latex(latex(delta)).inner - delta is S.Zero

    def test_antisymmetric_tensor(self):
        # spin orbitals
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        tensor = AntiSymmetricTensor("x", (i, j), (k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # spatial orbitals
        i, j, k, l = get_symbols("ijkl", spins="abab")  # noqa E741
        tensor = AntiSymmetricTensor("tensor", (i,), (j, k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # exponent > 1
        tensor = Pow(tensor, 2)
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # exponent < 1
        tensor = Pow(tensor, -1)
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero

    def test_symmetric_tensor(self):
        # spin orbitals
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        tensor = SymmetricTensor(tensor_names.coulomb, (i, j), (k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # spatial orbitals
        i, j, k, l = get_symbols("ijkl", spins="abab")  # noqa E741
        tensor = SymmetricTensor(tensor_names.coulomb, (i,), (j, k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero

    def test_amplitude(self):
        # spin orbitals
        i, j, k, l = get_symbols("ijkl")  # noqa E741
        tensor = Amplitude(tensor_names.left_adc_amplitude, (i, j), (k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # spatial orbitals
        i, j, k, l = get_symbols("ijkl", spins="abab")  # noqa E741
        tensor = Amplitude(tensor_names.gs_amplitude, (i,), (j, k, l))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero

    def test_nonsymmetric_tensor(self):
        # spin orbitals
        i, j = get_symbols("ij")  # noqa E741
        tensor = NonSymmetricTensor("bla", (i, j))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero
        # spatial orbitals
        i, j = get_symbols("ij", spins="ab")  # noqa E741
        tensor = NonSymmetricTensor("bla", (i, j))
        assert import_from_sympy_latex(latex(tensor)).inner - tensor is S.Zero

    def test_second_quant_operator(self):
        i, j = get_symbols("ij")
        op = F(i)
        assert import_from_sympy_latex(latex(op)).inner - op is S.Zero
        op = Fd(i)
        assert import_from_sympy_latex(latex(op)).inner - op is S.Zero
        op = NO(F(i) * Fd(j))
        assert import_from_sympy_latex(latex(op)).inner - op is S.Zero

    def test_product(self):
        i, j = get_symbols("ij")
        prod = (
            AntiSymmetricTensor(tensor_names.fock, (i,), (j,))
            * NonSymmetricTensor(tensor_names.orb_energy, (i,))
        )
        assert import_from_sympy_latex(latex(prod)).inner - prod is S.Zero

    def test_sum(self):
        i, j = get_symbols("ij")
        sum = (
            AntiSymmetricTensor(tensor_names.fock, (i,), (j,))
            + Pow(NonSymmetricTensor(tensor_names.orb_energy, (i,)), -42)
        )
        assert import_from_sympy_latex(latex(sum)).inner - sum is S.Zero

    def test_simple_frac(self):
        # t2_1 amplitude
        i, j, a, b = get_symbols("ijab")
        num = AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b))
        denom = (
            NonSymmetricTensor(tensor_names.orb_energy, (a,))
            + NonSymmetricTensor(tensor_names.orb_energy, (b,))
            - NonSymmetricTensor(tensor_names.orb_energy, (i,))
            - NonSymmetricTensor(tensor_names.orb_energy, (j,))
        )
        fraction = num / denom
        imported = import_from_sympy_latex(latex(fraction))
        assert imported.inner - fraction is S.Zero

    def test_complex_frac(self):
        # t2_1 * 1 / singles denom
        i, j, k, a, b, c = get_symbols("ijkabc", "ababab")
        num = AntiSymmetricTensor(tensor_names.eri, (i, j), (a, b))
        denom = (
            NonSymmetricTensor(tensor_names.orb_energy, (a,))
            + NonSymmetricTensor(tensor_names.orb_energy, (b,))
            - NonSymmetricTensor(tensor_names.orb_energy, (i,))
            - NonSymmetricTensor(tensor_names.orb_energy, (j,))
        )
        denom *= (
            - NonSymmetricTensor(tensor_names.orb_energy, (c,))
            + NonSymmetricTensor(tensor_names.orb_energy, (k,))
        )
        fraction = num / denom
        # check the not expanded expression
        imported = import_from_sympy_latex(latex(fraction))
        assert imported.inner - fraction is S.Zero
        # check the expanded expression
        fraction = fraction.expand()
        imported = import_from_sympy_latex(latex(fraction))
        assert imported.inner - fraction is S.Zero
