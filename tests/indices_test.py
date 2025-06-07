from adcgen.indices import Indices, get_symbols, split_idx_string

import pytest


class TestIndices:
    def test_get_indices(self):
        # ensure that the same indices are returned
        idx = Indices()
        assert idx.get_indices("ijk") == idx.get_indices("ijk")
        assert idx.get_indices("ijk", "aba") == idx.get_indices("ijk", "aba")
        assert idx.get_indices("a", "a") != idx.get_indices("a", "b")
        assert idx.get_indices("PQ") == idx.get_indices("PQ")
        res = idx.get_indices("Ij", "ba")
        I, j = res[("core", "b")].pop(), res[("occ", "a")].pop()
        assert I.space == "core" and I.spin == "b"
        assert j.space == "occ" and j.spin == "a"
        res = idx.get_indices("Pa")
        P, a = res[("aux", "")].pop(), res[("virt", "")].pop()
        assert P.space == "aux" and P.spin == ""
        assert a.space == "virt" and a.spin == ""

    def test_get_generic_indices(self):
        # ensure that generic indices don't overlap
        # can't explicitly test for names since the class is a singleton
        # and therefore the result would depend on the previous tests
        idx = Indices()
        for space in idx.base:
            n_wo_spin = len(idx._symbols[space][""])
            n_alpha = len(idx._symbols[space]["a"])
            n_beta = len(idx._symbols[space]["b"])
            kwargs = {space: 2}
            assert (idx.get_generic_indices(**kwargs) !=
                    idx.get_generic_indices(**kwargs))
            assert n_wo_spin + 4 == len(idx._symbols[space][""])
            kwargs = {f"{space}_a": 2}
            assert (idx.get_generic_indices(**kwargs) !=
                    idx.get_generic_indices(**kwargs))
            assert n_alpha + 4 == len(idx._symbols[space]["a"])
            kwargs = {f"{space}_b": 2}
            assert (idx.get_generic_indices(**kwargs) !=
                    idx.get_generic_indices(**kwargs))
            assert n_beta + 4 == len(idx._symbols[space]["b"])

    def test_get_symbols(self):
        # without spin
        idx = Indices().get_indices("iIa")
        i = idx[("occ", "")].pop()
        I = idx[("core", "")].pop()  # noqa E741
        a = idx[("virt", "")].pop()
        assert i is get_symbols("i").pop()
        assert [i, I, a] == get_symbols("iIa")
        assert [a, i, I] == get_symbols("aiI")
        # with spin
        idx = Indices().get_indices("iIa", "abb")
        i = idx[("occ", "a")].pop()
        I = idx[("core", "b")].pop()  # noqa E741
        a = idx[("virt", "b")].pop()
        assert i is get_symbols("i", "a").pop()
        assert [i, I, a] == get_symbols("iIa", "abb")
        assert [a, i, I] == get_symbols("aiI", "bab")


def test_split_idx_string():
    # single index
    res = split_idx_string("i")
    assert res == ["i"]
    res = split_idx_string("i3")
    assert res == ["i3"]
    res = split_idx_string("i3234235")
    assert res == ["i3234235"]
    # multiple indices without number
    res = split_idx_string("iJa")
    assert res == ["i", "J", "a"]
    # multiple indices with numbers
    res = split_idx_string("i3J11a2")
    assert res == ["i3", "J11", "a2"]
    # some indices with number
    res = split_idx_string("i334Jab")
    assert res == ["i334", "J", "a", "b"]
    res = split_idx_string("i3J33ab")
    assert res == ["i3", "J33", "a", "b"]
    # arbitrary index names
    res = split_idx_string("i⍺β3Ɣ23")
    assert res == ["i", "⍺", "β3", "Ɣ23"]
    # invalid string: starting with a number
    with pytest.raises(ValueError):
        split_idx_string("3Ɣ23")
