from adcgen.indices import Indices, get_symbols


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
