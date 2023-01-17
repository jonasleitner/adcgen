import pytest
from collections import namedtuple
import pathlib
import json

from sympy_adc.func import import_from_sympy_latex


@pytest.fixture(scope='session')
def cls_instances():
    from sympy_adc.groundstate import Hamiltonian, ground_state
    from sympy_adc.isr import intermediate_states
    from sympy_adc.secular_matrix import secular_matrix

    instances = namedtuple('instances', ['h', 'gs', 'isr_pp', 'm_pp'])
    h = Hamiltonian()
    gs = ground_state(h, first_order_singles=False)
    isr_pp = intermediate_states(gs, variant='pp')
    m_pp = secular_matrix(isr_pp)
    return instances(h, gs, isr_pp, m_pp)


@pytest.fixture(scope='session')
def reference_data() -> dict[int, dict]:

    def import_data_strings(data_dict: dict) -> dict:
        ret = {}
        for key, val in data_dict.items():
            if isinstance(val, dict):
                ret[key] = import_data_strings(val)
            elif isinstance(val, str):
                ret[key] = import_from_sympy_latex(val)
            else:
                raise TypeError(f"Unknown type {type(val)}.")
        return ret

    def hook(d): return {int(key) if key.isnumeric() else key: val
                         for key, val in d.items()}

    cache = {}

    path_to_data = pathlib.Path(__file__).parent / 'reference_data'
    for jsonfile in path_to_data.glob('*.json'):
        data = json.load(open(jsonfile), object_hook=hook)
        cache[jsonfile.name.rstrip('.json')] = import_data_strings(data)
    return cache
