from adcgen.new_indices import Index, Spin

from . import EnsureClearedSpaces, IsolatedIndexSpace

import pytest


class TestIndex(EnsureClearedSpaces):
    def test_name(self):
        # ensure that only names are accepted that belong to the
        # space
        occ = IsolatedIndexSpace("occ", "i", 0)
        i = Index("i", occ)
        assert i.name.base == "i" and i.name.suffix == 0
        i = Index("i42", occ, spin="a")
        assert i.name.base == "i" and i.name.suffix == 42
        assert i.spin is Spin.ALPHA
        with pytest.raises(ValueError):
            Index("j", occ)

    def test_sorting(self):
        # 1) metadata: space, spin
        # 2) name: suffix, base
        # 3) subscripts
        # 4) hash (so the order is always well defined)
        occ = IsolatedIndexSpace("occ", "ij", sort_key=0)
        virt = IsolatedIndexSpace("virt", "ab", sort_key=1)
        i = Index("i", occ)
        a = Index("a", virt)
        assert i.sort_key() < a.sort_key()  # defined in the space def
        ia = Index("i", occ, spin=Spin.ALPHA)
        assert i.sort_key() < ia.sort_key()  # no spin < alpha
        ib = Index("i", occ, spin=Spin.BETA)
        assert ia.sort_key() < ib.sort_key()  # alpha < beta
        ja = Index("j", occ, spin=Spin.ALPHA)
        assert ia.sort_key() < ja.sort_key()  # i < j
        i42a = Index("i42", occ, spin=Spin.ALPHA)
        assert ja.sort_key() < i42a.sort_key()  # j < i42
        i_i42a = Index("i", occ, subscripts=[i42a])
        assert i.sort_key() < i_i42a.sort_key()  # with subscript
        i_i = Index("i", occ, subscripts=[i])
        assert i_i.sort_key() < i_i42a.sort_key()  # with subscript
        # ensure the overall ordering is also correct
        idx = [i, i_i, i_i42a, ia, ja, i42a, ib, a]
        assert idx == sorted(idx, key=Index.sort_key)
        assert i.sort_key() != Index("i", occ)
