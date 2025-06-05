from adcgen.expression import import_from_sympy_latex
from adcgen.generate_code.contraction import Contraction
from adcgen.generate_code.optimize_contractions import (
    _group_objects, optimize_contractions
)
from adcgen.indices import get_symbols
from adcgen.sympy_objects import SymmetricTensor

import pytest


class TestOptimizeContractions:
    sizes = {"core": 5, "occ": 20, "virt": 200}

    def test_factor(self):
        test = "{d^{i}_{a}} {d^{i}_{a}} {d^{j}_{b}} {d^{j}_{b}}"
        test = import_from_sympy_latex(test)
        res = optimize_contractions(test.terms[0], "", None,
                                    space_dims=self.sizes)
        assert len(res) == 3
        i, j, a, b = get_symbols("ijab")
        ref = Contraction(((i, a), (i, a)), ("d_ov", "d_ov"), tuple())
        assert res[0] == ref
        ref = Contraction(((j, b), (j, b)), ("d_ov", "d_ov"), tuple())
        assert res[1] == ref
        ref = Contraction((tuple(), tuple()), ("bla", "bla"), tuple())
        assert res[2].indices == ref.indices
        assert res[2].contracted == ref.contracted
        assert res[2].target == ref.target
        assert res[2].scaling == ref.scaling

    def test_nested_contraction(self):
        test = "{Y^{b}_{j}} {t1^{bc}_{jk}} {t2eri4_{ikac}}"
        test = import_from_sympy_latex(test, convert_default_names=True)
        res = optimize_contractions(test.terms[0], "ia", None,
                                    space_dims=self.sizes)
        assert len(res) == 2
        i, j, k, a, b, c = get_symbols("ijkabc")
        ref = Contraction(((j, b), (j, k, b, c)), ("ur1", "t2_1"), (i, a))
        assert res[0] == ref
        ref = Contraction(((k, c), (i, k, a, c)),
                          (res[0].contraction_name, "t2eri_4"), (i, a))
        assert res[1] == ref

    def test_hypercontraction(self):
        test = "{X^{a}_{i}} {V^{ib}_{ce}} {V^{ja}_{cd}} {Y^{b}_{j}}"
        test = import_from_sympy_latex(test, convert_default_names=True)
        # {D^{ij}_{ec}} {D^{ij}_{cd}}
        i, j, c, d, e = get_symbols("ijcde")
        test *= SymmetricTensor("D", (i, j), (e, c))
        test *= SymmetricTensor("D", (i, j), (c, d))
        test.make_real()

        res = optimize_contractions(test.terms[0], target_indices="de",
                                    max_itmd_dim=4,
                                    max_n_simultaneous_contracted=None,
                                    space_dims=self.sizes)
        assert len(res) == 3
        assert len(res[2].indices) == 4
        assert len(res[1].indices) == 2 and len(res[0].indices) == 2
        res2 = optimize_contractions(test.terms[0], target_indices="de",
                                     max_itmd_dim=4,
                                     max_n_simultaneous_contracted=4,
                                     space_dims=self.sizes)
        assert res[:2] == res2[:2]
        assert (res[2].indices == res2[2].indices and
                res[2].contracted == res2[2].contracted and
                res[2].target == res2[2].target and
                res[2].scaling == res2[2].scaling)
        with pytest.raises(RuntimeError):
            optimize_contractions(test.terms[0], target_indices="de",
                                  max_itmd_dim=4,
                                  max_n_simultaneous_contracted=3,
                                  space_dims=self.sizes)


class TestGroupObjects:
    def test_full_connected(self):
        i, j, k = get_symbols("ijk")
        # 4 connected objects in different order without target indices
        relevant_obj_indices = [(i, j), (j, k), (j, k), (i, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2), (0, 1, 2, 3), (0, 3), (1, 2, 3))
        relevant_obj_indices = [(i, j), (j, k), (i, k), (j, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 3), (0, 1, 2, 3), (0, 2), (1, 2, 3))
        relevant_obj_indices = [(j, k), (j, k), (i, j), (i, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3), (0, 1, 2), (0, 1, 3), (2, 3))
        # with a target index (and a possible outer product)
        relevant_obj_indices = [(j, k), (j, k), (i, j), (i, k)]
        target_indices = (i,)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3), (0, 1, 2), (0, 1, 3), (2, 3))

    def test_multiple_groups(self):
        i, j, k, l, p, q, r, s = get_symbols("ijklpqrs")
        # eri mo transformation
        relevant_obj_indices = [(p, q, r, s), (i, p), (j, q), (k, r), (l, s)]
        target_indices = (i, j, k, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        ref = ((0, 1), (0, 2), (0, 3), (0, 4),
               (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))  # outer
        assert res == ref
        # groups of different size
        relevant_obj_indices = [(p, q, r, s), (i, p), (j, p), (k, r), (l, s)]
        target_indices = (i, j, k, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        ref = ((0, 1, 2), (0, 3), (0, 4),
               (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))  # outer
        assert res == ref
        # with an isolated group
        relevant_obj_indices = [(p, q, p, s), (i, p), (j, p), (k, r), (l, s)]
        target_indices = (i, j, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        ref = ((0, 1, 2), (0, 1, 2, 4),
               (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))  # outer
        assert res == ref
