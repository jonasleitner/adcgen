from adcgen.indices import split_idx_string

import pytest


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
