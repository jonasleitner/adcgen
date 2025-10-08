from adcgen.new_indices import IndexName

from . import EnsureClearedSpaces, IsolatedIndexSpace

import pytest


class TestIndexSpace(EnsureClearedSpaces):
    def test_idx_names(self):
        with pytest.raises(ValueError):  # name with suffix
            IsolatedIndexSpace("bla", "i3jk", sort_key=0)
        with pytest.raises(ValueError):  # duplicate name
            IsolatedIndexSpace("occ", "ii", 0)
        # valid names: ensure they are sorted and imported correctly
        sp = IsolatedIndexSpace("occ", "ji", sort_key=0)
        names = (IndexName(base="i", suffix=0), IndexName(base="j", suffix=0))
        assert names == sp.idx_names

    def test_cache(self):
        # try to create two spaces with the same name
        sp = IsolatedIndexSpace("occ", "ijk", sort_key=0)
        assert sp in IsolatedIndexSpace._available_spaces
        with pytest.raises(ValueError):
            IsolatedIndexSpace("occ", "abc", sort_key=1)
        # try to create another one with intersecting indices
        with pytest.raises(ValueError):
            IsolatedIndexSpace("virt", "iabc", sort_key=1)
        # ensure we obtain the correct space from the cache
        assert sp is IsolatedIndexSpace.get("occ")
        assert sp is IsolatedIndexSpace.get_by_index_name("i")
        IsolatedIndexSpace.remove_from_cache(sp)
        assert sp not in IsolatedIndexSpace._available_spaces
        assert not IsolatedIndexSpace._available_spaces

    def test_is_subspace(self):
        general = IsolatedIndexSpace("general", "pq", 0)
        occ = IsolatedIndexSpace("occ", "ij", sort_key=1, parent=general)
        occ1 = IsolatedIndexSpace("occ1", "k", sort_key=1, parent=occ)
        assert occ.is_subspace_of(general)
        assert occ1.is_subspace_of(occ)
        assert occ1.is_subspace_of(general)
        assert not general.is_subspace_of(occ)
        assert not general.is_subspace_of(occ1)
        assert not general.is_subspace_of(general)

    def test_is_valid_index_name(self):
        occ = IsolatedIndexSpace("occ", "i", 0)
        assert occ.is_valid_index_name("i")
        assert occ.is_valid_index_name("i0")
        assert occ.is_valid_index_name("i42")
        assert not occ.is_valid_index_name("j")
        assert not occ.is_valid_index_name("j42")
