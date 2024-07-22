from .misc import Singleton
from pathlib import Path
from dataclasses import dataclass
import json


_config_file = "tensor_names.json"


# NOTE: Currently it is only possible to modify the values through
# 'tensor_names.json'. It is not possible to adjust them by modifying the
# class attributes. This is less flexible but ensures that throughout
# a run of the program consistent names are used. Especially,
# due to the caching weird behaviour might be possible
# if names are changed in the middle of a run.
@dataclass(slots=True, frozen=True)
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
    The attributes storing the tensor names are given in brackets.
    """
    eri: str = "V"
    coulomb: str = "v"
    fock: str = "f"
    operator: str = "d"

    @staticmethod
    def _from_config() -> 'TensorNames':
        """
        Construct the TensorNames instance with values from the config file
        'tensor_names.json'.
        """
        config_file = Path(__file__).parent.resolve() / _config_file
        tensor_names: dict[str, str] = json.load(open(config_file, "r"))
        return TensorNames(**tensor_names)


# init the TensorNames instance and overwrite the defaults with
# values from the config file
TensorNames._from_config()
