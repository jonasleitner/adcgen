from sympy_adc.intermediates import Intermediates


class TestAllowedSpinBlocks:
    def test_t2_1(self):
        ref = ["aaaa", "abab", "baba", "bbbb"]
        t2 = Intermediates().available["t2_1"]
        assert t2.allowed_spin_blocks == ref

    def test_t1_2(self):
        ref = ["aa", "bb"]
        t1 = Intermediates().available["t1_2"]
        assert t1.allowed_spin_blocks == ref

    def test_t2_2(self):
        ref = ["aaaa", "abab", "abba", "baab", "baba", "bbbb"]
        t2 = Intermediates().available["t2_2"]
        assert t2.allowed_spin_blocks == ref
