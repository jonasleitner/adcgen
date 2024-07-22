import pytest
import pathlib
import json

from adcgen.func import import_from_sympy_latex
from adcgen.tensor_names import tensor_names


@pytest.fixture(scope='session')
def cls_instances():
    from adcgen.groundstate import Operators, GroundState
    from adcgen.intermediate_states import IntermediateStates
    from adcgen.secular_matrix import SecularMatrix
    from adcgen.properties import Properties

    mp_op = Operators(variant='mp')
    re_op = Operators(variant='re')
    mp = GroundState(mp_op, first_order_singles=False)
    mp_singles = GroundState(mp_op, first_order_singles=True)
    re = GroundState(re_op, first_order_singles=False)
    re_singles = GroundState(re_op, first_order_singles=True)
    isr_pp = IntermediateStates(mp, variant='pp')
    isr_re_pp = IntermediateStates(re, variant='pp')
    m_pp = SecularMatrix(isr_pp)
    prop_pp = Properties(isr_pp)
    return {
        'mp': {
            'op': mp_op,
            'gs': mp,
            'gs_with_singles': mp_singles,
            'isr_pp': isr_pp,
            'm': m_pp,
            'prop_pp': prop_pp,
        },
        're': {
            'op': re_op,
            'gs': re,
            'gs_with_singles': re_singles,
            'isr_pp': isr_re_pp
        }
    }


@pytest.fixture(scope='session')
def reference_data() -> dict[int, dict]:

    def import_data_strings(data_dict: dict) -> dict:
        ret = {}
        for key, val in data_dict.items():
            if isinstance(val, dict):
                ret[key] = import_data_strings(val)
            elif isinstance(val, str):
                # import the expression string and rename tensors to match
                # the currently used tensor names
                ret[key] = tensor_names.rename_tensors(
                    import_from_sympy_latex(val)
                )
            else:
                raise TypeError(f"Unknown type {type(val)}.")
        return ret

    def hook(d): return {int(key) if key.isnumeric() else key: val
                         for key, val in d.items()}

    cache = {}

    path_to_data = pathlib.Path(__file__).parent / 'reference_data'
    for jsonfile in path_to_data.glob('*.json'):
        data = json.load(open(jsonfile), object_hook=hook)
        name = jsonfile.name.split('.json')  # remove .json extension
        assert len(name) == 2
        cache[name[0]] = import_data_strings(data)
    return cache
