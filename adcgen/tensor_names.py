from .misc import Singleton
from pathlib import Path
import json


# NOTE: Currently it it possible to rename a tensor at every point in time
# by modifying the appropriate attribute. This might result in weird behaviour
# if a name is modified during a run. This could be prevented by only allowing
# modifications through the 'tensor_names.json' config file, which also means
# that the names can not be changed after after the TensorNames class has
# been initialized the first time. Less flexible but also less error prone.
class TensorNames(metaclass=Singleton):
    """
    Singleton class that that is used to define the names of tensors
    used throughout the code. The names may be changed either by modifying
    'tensor_names.json' or by modifying the appropriate attributes on the
    class instance. By default the following names are used:
    - antisymmetric ERI in physicist notation (eri): V
    - Coulomb integrals in chemist notation (coulomb): v,
    - The fock matrix (fock): f,
    - The arbitrary N-particle operator matrix (operator): d
    In backets the name of the attribute storing the tensor name is given.
    """
    _config_file = "tensor_names.json"

    def __init__(self) -> None:
        config_file = Path(__file__).parent.resolve() / self._config_file
        tensor_names: dict[str, str] = json.load(open(config_file, "r"))
        for key, val in tensor_names.items():
            self.__setattr__(key, val)
