from adcgen.indices import Indices


class TestIndices:
    def test_get_indices(self):
        # ensure that the same indices are returned
        idx = Indices()
        assert idx.get_indices("ijk") == idx.get_indices("ijk")
        assert idx.get_indices("ijk", "aba") == idx.get_indices("ijk", "aba")
        assert idx.get_indices("a", "a") != idx.get_indices("a", "b")

    def test_get_generic_indices(self):
        # ensure that generic indices don't overlap
        # can't explicitly test for names since the class is a singleton
        # and therefore the result would depend on the previous tests
        idx = Indices()
        occ = len(idx._symbols["occ"][""])
        occ_a = len(idx._symbols["occ"]["a"])
        occ_b = len(idx._symbols["occ"]["b"])

        assert idx.get_generic_indices(occ=2) != idx.get_generic_indices(occ=2)
        assert occ + 4 == len(idx._symbols["occ"][""])
        assert idx.get_generic_indices(occ_a=2) != \
            idx.get_generic_indices(occ_a=2)
        assert occ_a + 4 == len(idx._symbols["occ"]["a"])
        assert idx.get_generic_indices(occ_b=2) != \
            idx.get_generic_indices(occ_b=2)
        assert occ_b + 4 == len(idx._symbols["occ"]["b"])

        virt = len(idx._symbols["virt"][""])
        virt_a = len(idx._symbols["virt"]["a"])
        virt_b = len(idx._symbols["virt"]["b"])
        assert idx.get_generic_indices(virt=2) != \
            idx.get_generic_indices(virt=2)
        assert idx.get_generic_indices(virt_a=2) != \
            idx.get_generic_indices(virt_a=2)
        assert idx.get_generic_indices(virt_b=2) != \
            idx.get_generic_indices(virt_b=2)
        assert virt + 4 == len(idx._symbols["virt"][""])
        assert virt_a + 4 == len(idx._symbols["virt"]["a"])
        assert virt_b + 4 == len(idx._symbols["virt"]["b"])

        g = len(idx._symbols["general"][""])
        g_a = len(idx._symbols["general"]["a"])
        g_b = len(idx._symbols["general"]["b"])
        assert idx.get_generic_indices(general=2) != \
            idx.get_generic_indices(general=2)
        assert idx.get_generic_indices(general_a=2) != \
            idx.get_generic_indices(general_a=2)
        assert idx.get_generic_indices(general_b=2) != \
            idx.get_generic_indices(general_b=2)
        assert g + 4 == len(idx._symbols["general"][""])
        assert g_a + 4 == len(idx._symbols["general"]["a"])
        assert g_b + 4 == len(idx._symbols["general"]["b"])
