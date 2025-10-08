from adcgen.new_indices import IndexName, Spin

import pytest


class TestIndexName:
    def test_from_str(self):
        # names have to be of the form characer + number
        name = IndexName.from_str("a4")
        assert name.base == "a" and name.suffix == 4
        name = IndexName.from_str("a42")
        assert name.base == "a" and name.suffix == 42
        name = IndexName.from_str("a")
        assert name.base == "a" and name.suffix == 0
        name = IndexName.from_str("π")
        assert name.base == "π" and name.suffix == 0
        name = IndexName.from_str("π42")
        assert name.base == "π" and name.suffix == 42
        with pytest.raises(ValueError):
            name = IndexName.from_str("42a")
        with pytest.raises(ValueError):
            name = IndexName.from_str("42")
        with pytest.raises(ValueError):
            name = IndexName.from_str("ab")

    def test_str(self):
        # ensure that the names are printed correctly
        name = IndexName(base="a", suffix=42)
        assert str(name) == "a42"


class TestSpin:
    def test_from_str(self):
        assert Spin.from_str("n") is Spin.NONE
        assert Spin.from_str("none") is Spin.NONE
        assert Spin.from_str("a") is Spin.ALPHA
        assert Spin.from_str("alpha") is Spin.ALPHA
        assert Spin.from_str("b") is Spin.BETA
        assert Spin.from_str("beta") is Spin.BETA
        with pytest.raises(ValueError):
            Spin.from_str("invalid")

    def test_sorting(self):
        spins = [Spin.NONE, Spin.ALPHA, Spin.BETA]
        assert sorted(spins, key=Spin.sort_key) == spins

    def test_bool(self):
        assert Spin.ALPHA and Spin.BETA
        assert not Spin.NONE
