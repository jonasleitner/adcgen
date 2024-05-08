from adcgen.indices import Indices


class TestIndices:
    def test_get_indices(self):
        # ensure that the same indices are returned
        idx = Indices()
        assert idx.get_indices("ijk") == idx.get_indices("ijk")
        assert idx.get_indices("ijk", "aba") == idx.get_indices("ijk", "aba")
        assert idx.get_indices("a", "a") != idx.get_indices("a", "b")
        idx.get_generic_indices(n_o=2)

    def test_get_generic_indices(self):
        # ensure that generic indices don't overlap
        # can't explicitly test for names since the class is a singleton
        # and therefore the result would depend on the previous tests
        idx = Indices()
        assert idx.get_generic_indices(n_o=2) != idx.get_generic_indices(n_o=2)
        assert idx.get_generic_indices(n_o_a=2) != \
            idx.get_generic_indices(n_o_a=2)
        assert idx.get_generic_indices(n_o_b=2) != \
            idx.get_generic_indices(n_o_b=2)
        assert idx.get_generic_indices(n_v=2) != idx.get_generic_indices(n_v=2)
        assert idx.get_generic_indices(n_v_a=2) != \
            idx.get_generic_indices(n_v_a=2)
        assert idx.get_generic_indices(n_v_b=2) != \
            idx.get_generic_indices(n_v_b=2)
        assert idx.get_generic_indices(n_g=2) != idx.get_generic_indices(n_g=2)
        assert idx.get_generic_indices(n_g_a=2) != \
            idx.get_generic_indices(n_g_a=2)
        assert idx.get_generic_indices(n_g_b=2) != \
            idx.get_generic_indices(n_g_b=2)
