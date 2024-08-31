from adcgen.rules import Rules
from adcgen.tensor_names import tensor_names

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
                tensor_names.eri: ('ooov', 'ovoo', 'oovv', 'vvoo', 'ovvv',
                                   'vvov'),
                tensor_names.fock: ('ov', 'vo')
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
                tensor_names.eri: ('oooo', 'ovov', 'vvvv'),
                tensor_names.fock: ('oo', 'vv')
            })
            assert rules == ref_rules

    @pytest.mark.parametrize("n_create", [1, 2, 3])
    @pytest.mark.parametrize("n_annihilate", [1, 2, 3])
    def test_operator(self, variant, n_create, n_annihilate, cls_instances,
                      reference_data):
        # parametrize variant to ensure that the operators do not depend
        # on the variant

        # load the reference data
        ref = reference_data['operators'][f"{n_create}_{n_annihilate}"]

        op, rules = cls_instances[variant]['op'].operator(
            n_create=n_create, n_annihilate=n_annihilate
        )
        # need to substitute the contracted indices
        assert (ref - op).substitute_contracted().sympy is S.Zero

        assert rules is None

    @pytest.mark.parametrize("creation", [None, "a", "ab"])
    @pytest.mark.parametrize("annihilation", [None, "i", "ij"])
    def test_excitation_operator(self, variant, creation, annihilation,
                                 cls_instances, reference_data):
        # load the reference data
        ref = reference_data["operators"]["excitation"]
        ref = ref[f"{creation}_{annihilation}"]

        op = cls_instances[variant]["op"].excitation_operator(
            creation=creation, annihilation=annihilation,
            reverse_annihilation=True
        )
        assert op - ref["true"].sympy is S.Zero
        op = cls_instances[variant]["op"].excitation_operator(
            creation=creation, annihilation=annihilation,
            reverse_annihilation=False
        )
        assert op - ref["false"].sympy is S.Zero
