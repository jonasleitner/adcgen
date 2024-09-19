from adcgen.func import import_from_sympy_latex
from adcgen.generate_code.contraction import Contraction
from adcgen.generate_code.optimize_contractions import (
    _group_objects, optimize_contractions
)
from adcgen.indices import get_symbols


class TestOptimizeContractions:
    def test_factor(self):
        test = "{d^{i}_{a}} {d^{i}_{a}} {d^{j}_{b}} {d^{j}_{b}}"
        test = import_from_sympy_latex(test)
        res = optimize_contractions(test.terms[0], "", None)
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
        test = import_from_sympy_latex(test)
        res = optimize_contractions(test.terms[0], "ia", None)
        assert len(res) == 2
        i, j, k, a, b, c = get_symbols("ijkabc")
        ref = Contraction(((j, b), (j, k, b, c)), ("ur1", "t2_1"), (i, a))
        assert res[0] == ref
        ref = Contraction(((k, c), (i, k, a, c)),
                          (f"contraction_{res[0].id}", "t2eri_4"), (i, a))
        assert res[1] == ref


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
        ref = ((0, 1, 2), (0, 4),
               (0, 3), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))  # outer
        assert res == ref
