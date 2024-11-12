from adcgen.generate_code.contraction import (
    Contraction, ScalingComponent, Scaling
)
from adcgen.indices import get_symbols


class TestContraction:
    def test_indices(self):
        i, j, k, b, c = get_symbols("ijkbc")
        indices = ((i, k), (j, k))
        names = ("f_oo", "f_oo")
        target_indices = (i, j)
        contr = Contraction(indices, names, target_indices)
        assert contr.contracted == (k,)
        assert contr.target == (i, j)
        # swap target indices
        indices = ((i, k), (j, k))
        names = ("f_oo", "f_oo")
        target_indices = (j, i)
        contr = Contraction(indices, names, target_indices)
        assert contr.contracted == (k,)
        assert contr.target == (j, i)
        # non einstein target indices
        indices = ((i, k), (j, k), (j, k))
        names = ("f_oo", "f_oo")
        target_indices = (i,)
        contr = Contraction(indices, names, target_indices)
        assert contr.contracted == (j, k)
        assert contr.target == (i,)
        indices = ((j, b), (j, k, b, c))
        names = ("ur1", "t2_1")
        contr = Contraction(indices, names, target_indices)
        assert contr.contracted == (j, b)
        assert contr.target == (k, c)

    def test_scaling(self):
        i, j, k = get_symbols("ijk")
        indices = ((i, k), (j, k))
        names = ("f_oo", "f_oo")
        target_indices = (i, j)
        contr = Contraction(indices, names, target_indices)
        scaling = contr.scaling
        comp = ScalingComponent(3, 0, 0, 3, 0)
        mem = ScalingComponent(2, 0, 0, 2, 0)
        assert scaling.computational == comp
        assert scaling.memory == mem
        assert scaling == Scaling(comp, mem)
