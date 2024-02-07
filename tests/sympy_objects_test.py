from sympy_adc.sympy_objects import Index


class TestIndex:
    def test_equality(self):
        i = Index("i")
        assert i == i
        assert i != Index("i")

    def test_spin(self):
        i = Index("i", spin="a")
        assert i.assumptions0.get("alpha")
        assert i.spin == "a"
        assert i.assumptions0.get("beta") is None
        i = Index("i")
        assert i.assumptions0.get("beta") is None
        assert i.assumptions0.get("alpha") is None
        assert i.spin is None

    def test_space(self):
        assert Index("i", below_fermi=True).space == "occ"
        assert Index("i", above_fermi=True).space == "virt"
        assert Index("i").space == "general"
