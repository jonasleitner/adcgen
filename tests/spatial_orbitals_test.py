from sympy_adc.intermediates import Intermediates


class TestAllowedSpinBlocks:
    def test_t2_1(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_1"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref

    def test_t1_2(self):
        ref = ("aa", "bb")
        t1 = Intermediates().available["t1_2"]
        assert t1.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t1.allowed_spin_blocks == ref

    def test_t2_2(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_2"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref

    def test_t3_2(self):
        ref = ("aaaaaa", "aabaab", "aababa", "aabbaa", "abaaab", "abaaba",
               "ababaa", "baaaab", "baaaba", "baabaa", "abbabb", "abbbab",
               "abbbba", "bbabba", "bbabab", "bbaabb", "babbba", "babbab",
               "bababb", "bbbbbb")
        ref = tuple(sorted(ref))
        t3 = Intermediates().available["t3_2"]
        assert t3.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t3.allowed_spin_blocks == ref

    def test_t4_2(self):
        t4 = Intermediates().available["t4_2"]
        sb1 = t4.tensor().terms[0].objects[0].allowed_spin_blocks
        sb2 = t4.allowed_spin_blocks
        assert sb1 == sb2

    def test_t1_3(self):
        ref = ("aa", "bb")
        t1 = Intermediates().available["t1_3"]
        assert t1.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t1.allowed_spin_blocks == ref

    def test_t2_3(self):
        ref = ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        t2 = Intermediates().available["t2_3"]
        assert t2.tensor().terms[0].objects[0].allowed_spin_blocks == ref
        assert t2.allowed_spin_blocks == ref
