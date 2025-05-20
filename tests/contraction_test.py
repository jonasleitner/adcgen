from adcgen.generate_code.contraction import (
    Contraction, ScalingComponent, Scaling, Sizes
)
from adcgen.indices import get_symbols

from dataclasses import asdict


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
        comp = ScalingComponent(3, 0, 0, 3, 0, 0)
        mem = ScalingComponent(2, 0, 0, 2, 0, 0)
        assert scaling.computational == comp
        assert scaling.memory == mem
        assert scaling == Scaling(comp, mem)

    def test_sizes(self):
        # test the automatic evaluation of the size of the general space
        sizes = {"occ": 1, "virt": 2, "core": 3, "ri": 0}
        res = Sizes.from_dict(sizes)
        sizes["general"] = 6
        assert sizes == asdict(res)
        sizes["general"] = 10
        res = Sizes.from_dict(sizes)
        assert sizes == asdict(res)

    def test_evalute_costs(self):
        sizes = {"occ": 3, "virt": 5, "core": 2, "general": 7, "ri": 8}
        sizes = Sizes.from_dict(sizes)
        comp = ScalingComponent(42, 1, 2, 3, 4, 5)
        mem = ScalingComponent(42, 5, 4, 3, 2, 1)
        scaling = Scaling(comp, mem)
        print(comp.evaluate_costs(sizes))
        print(mem.evaluate_costs(sizes))
        print(scaling.evaluate_costs(sizes))
        assert comp.evaluate_costs(sizes) == 2477260800 
        assert mem.evaluate_costs(sizes) == 9075780000
        assert scaling.evaluate_costs(sizes) == (2477260800, 9075780000)
        # ensure that zero sized spaces are ignored
        sizes = {"occ": 3, "virt": 5, "core": 0, "ri": 0}  # general == 8
        sizes = Sizes.from_dict(sizes)
        assert comp.evaluate_costs(sizes) == 5400
        comp = ScalingComponent(42, 0, 1, 2, 3, 0)
        assert comp.evaluate_costs(sizes) == 45
