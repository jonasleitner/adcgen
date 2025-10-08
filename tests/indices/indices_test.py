from . import EnsureClearedSpaces, IsolatedIndexSpace, IsolatedIndices


class TestIndices(EnsureClearedSpaces):
    def setup_method(self):
        # ensure that we have no indices from previous tests
        assert not IsolatedIndices()._index_cache
        assert not IsolatedIndices()._generic_counter
        return super().setup_method()

    def teardown_method(self):
        # Reset and clear the indices cache
        IsolatedIndices().reset()
        return super().teardown_method()

    def test_get_indices(self):
        # ensure that the same indices are returned
        indices = IsolatedIndices()
        occ = IsolatedIndexSpace("occ", "ijk", sort_key=0)
        i = indices.get_indices("i", spaces=[occ]).pop()
        assert i is indices.get_indices("i", [occ]).pop()
        ia = indices.get_indices("i", spaces=[occ], spins=["a"]).pop()
        assert ia is indices.get_indices("i", [occ], spins="a").pop()
        i_ia = indices.get_indices("i", spaces=[occ], subscripts=[[ia]]).pop()
        assert i_ia is not i
        assert (ia,) == i_ia.subscripts
        assert (
            i_ia is
            indices.get_indices("i", spaces=[occ], subscripts=[[ia]]).pop()
        )

    def test_get_generic_indices(self):
        # ensure that generic indices don't overlap and that the first
        # generic index has the _inital_suffix
        indices = IsolatedIndices()
        occ = IsolatedIndexSpace("occ", "ij", sort_key=0)
        first = indices.get_generic_indices(
            count=1, space=occ, spin=None
        ).pop()
        assert first is indices.get_indices(names="i42", spaces=[occ]).pop()
        second, third, fourth = indices.get_generic_indices(
            count=3, space=occ, spin=None
        )
        assert second is indices.get_indices(names="i43", spaces=[occ]).pop()
        assert third is indices.get_indices(names="j43", spaces=[occ]).pop()
        assert fourth is indices.get_indices(names="i44", spaces=[occ]).pop()
        # also ensure everything also works with spin and that we use
        # a separate counter for the same space but different spin
        with_spin = indices.get_generic_indices(
            count=3, space=occ, spin="alpha"
        )
        ref = indices.get_indices(
            names="i42j42i43", spaces=[occ, occ, occ], spins="aaa"
        )
        assert with_spin == ref
        # init a second space and ensure the counter restarts
        # for that space
        virt = IsolatedIndexSpace("virt", "a", sort_key=1)
        first = indices.get_generic_indices(
            count=1, space=virt, spin=None
        ).pop()
        assert first is indices.get_indices(names="a42", spaces=[virt]).pop()
