from sympy_adc.rules import Rules

from sympy import S
import pytest


@pytest.mark.parametrize('variant', ['mp', 're'])
class TestOperators:
    def test_h0(self, variant, cls_instances, reference_data):
        # load the reference data
        ref = reference_data['operators'][variant]['h0']

        h, rules = cls_instances[variant]['op'].h0
        # need to substitute the contracted indices in the operators
        assert (ref - h).substitute_contracted().sympy is S.Zero

        if variant == 'mp':
            assert rules is None
        elif variant == 're':
            ref_rules = Rules(forbidden_tensor_blocks={
                'V': ('ooov', 'ovoo', 'oovv', 'vvoo', 'ovvv', 'vvov'),
                'f': ('ov', 'vo')
            })
            assert rules == ref_rules

    def test_h1(self, variant, cls_instances, reference_data):
        # load the reference data
        ref = reference_data['operators'][variant]['h1']

        h, rules = cls_instances[variant]['op'].h1
        # need to substitute the contracted indices in the operators
        assert (ref - h).substitute_contracted().sympy is S.Zero

        if variant == 'mp':
            assert rules is None
        elif variant == 're':
            ref_rules = Rules(forbidden_tensor_blocks={
                'V': ('oooo', 'ovov', 'vvvv'),
                'f': ('oo', 'vv')
            })
            assert rules == ref_rules

    @pytest.mark.parametrize('opstring', ['ca', 'ccaa', 'cccaaa'])
    def test_operator(self, variant, opstring, cls_instances, reference_data):
        # parametrize variant to ensure that the operators do not depend
        # on the variant

        # load the reference data
        ref = reference_data['operators'][opstring]

        op, rules = cls_instances[variant]['op'].operator(opstring)
        # need to substitute the contracted indices
        assert (ref - op).substitute_contracted().sympy is S.Zero

        assert rules is None
