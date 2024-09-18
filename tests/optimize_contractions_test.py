from adcgen.generate_code.optimize_contractions import _group_objects
from adcgen.indices import get_symbols


class TestGroupObjects:
    def test_single_group(self):
        i, j, k = get_symbols("ijk")
        # 4 connected objects in different order without target indices
        relevant_obj_indices = [(i, j), (j, k), (j, k), (i, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3),)
        relevant_obj_indices = [(i, j), (j, k), (i, k), (j, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3),)
        relevant_obj_indices = [(j, k), (j, k), (i, j), (i, k)]
        target_indices = tuple()
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3),)
        # with a target index
        relevant_obj_indices = [(j, k), (j, k), (i, j), (i, k)]
        target_indices = (i,)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2, 3),)

    def test_multiple_groups(self):
        i, j, k, l, p, q, r, s = get_symbols("ijklpqrs")
        # eri mo transformation
        relevant_obj_indices = [(p, q, r, s), (i, p), (j, q), (k, r), (l, s)]
        target_indices = (i, j, k, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1), (0, 2), (0, 3), (0, 4))
        # groups of different size
        relevant_obj_indices = [(p, q, r, s), (i, p), (j, p), (k, r), (l, s)]
        target_indices = (i, j, k, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2), (0, 3), (0, 4))
        # with an isolated group
        relevant_obj_indices = [(p, q, p, s), (i, p), (j, p), (k, r), (l, s)]
        target_indices = (i, j, l)
        res = _group_objects(obj_indices=relevant_obj_indices,
                             target_indices=target_indices)
        assert res == ((0, 1, 2), (0, 4), (0, 3), (1, 3), (2, 3), (3, 4))
